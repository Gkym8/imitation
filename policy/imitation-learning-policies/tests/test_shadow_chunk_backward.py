"""Chunk backwards preserve the full-stream gradient with bounded live graphs."""
import copy
import gc
import unittest
from unittest.mock import Mock, patch

from accelerate import Accelerator
import torch

from imitation_learning.trainers.policy_trainer import PolicyTrainer
from imitation_learning.models.encoders.image_encoders import SharedModelManager
import test_shadow_memory as shadow_fixture


def make_trainer(policy, accelerator=None, model=None):
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    trainer.accelerator = accelerator or Accelerator(cpu=True)
    trainer.model = policy if model is None else model
    trainer.optimizer = torch.optim.SGD(policy.parameters(), lr=0.001)
    trainer.lr_scheduler = Mock()
    trainer.ema_model = Mock()
    trainer.epoch = 0
    trainer.clip_grad_norms = []
    trainer.streaming_chunk_length = 4
    trainer.streaming_chunk_accumulate = 1
    trainer.future_prediction_loss_weight = 0.05
    trainer.gate_selectivity_loss_weight = 0.01
    trainer.critical_action_loss_weights = [0.0]
    return trainer


class _SavedTensor:
    def __init__(self, tensor, tracker):
        self.tensor = tensor.detach()
        self.tracker = tracker
        self.size = tensor.numel() * tensor.element_size()
        tracker.live += self.size
        tracker.peak = max(tracker.peak, tracker.live)

    def __del__(self):
        self.tracker.live -= self.size


class _SavedTensorTracker:
    def __init__(self):
        self.live = self.peak = 0
        self.after_backward = []

    def pack(self, tensor):
        return _SavedTensor(tensor, self)

    @staticmethod
    def unpack(saved):
        return saved.tensor


class ShadowChunkBackwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shadow_fixture.ShadowMemoryPolicyTest.setUpClass()

    def setUp(self):
        self.case = shadow_fixture.ShadowMemoryPolicyTest()
        self.case.setUp()

    def tearDown(self):
        self.case.tearDown()

    def policy(self, mode='tokens', horizon=3):
        cfg = self.case.make_config(mode)
        cfg.workspace.model.denoising_network_partial.max_history_len = 2
        cfg.workspace.model.denoising_network_partial.shadow_horizon = horizon
        cfg.workspace.model.memory_eviction.keep_recent = 0
        return self.case.fixture.instantiate_policy(cfg)

    def test_all_parameter_gradients_and_updates_match_full_sequence(self):
        for mode, horizon in [('tokens', 3), ('pooled', 9)]:
            with self.subTest(mode=mode, horizon=horizon):
                policy = self.policy(mode, horizon)
                reference = copy.deepcopy(policy)
                batch = self.case.fixture.make_batch(anchors=19)
                batch['entire_traj_is_padding'][:, 17:] = True
                full_trainer = make_trainer(reference)
                sequence = dict(batch, _streaming_sequence_chunk_length=4, _streaming_sequence_length=17)
                torch.manual_seed(821)
                full_loss, full_logs = full_trainer.compute_loss(sequence)
                full_loss.backward()
                expected = {name: None if p.grad is None else p.grad.clone()
                            for name, p in reference.named_parameters()}
                full_trainer.optimizer.step()
                trainer = make_trainer(policy)
                actual = {}
                original_step = trainer.optimizer.step

                def capture_step(*args, **kwargs):
                    actual.update({name: None if p.grad is None else p.grad.clone()
                                   for name, p in policy.named_parameters()})
                    return original_step(*args, **kwargs)

                torch.manual_seed(821)
                with patch.object(trainer.optimizer, 'step', side_effect=capture_step) as step:
                    loss, logs = trainer.backward_streaming_batch(batch)
                self.assertEqual(step.call_count, 1)
                torch.testing.assert_close(loss, full_loss.detach(), rtol=2e-6, atol=1e-7)
                for name in expected:
                    self.assertEqual(actual[name] is None, expected[name] is None, name)
                    if expected[name] is not None:
                        torch.testing.assert_close(actual[name], expected[name], rtol=2e-4, atol=2e-7,
                                                   msg=lambda msg: name + ': ' + msg)
                for (name, p), (_, r) in zip(policy.named_parameters(), reference.named_parameters()):
                    torch.testing.assert_close(p, r, rtol=1e-6, atol=1e-7,
                                               msg=lambda msg: name + ': ' + msg)
                self.assertIsNone(policy._streaming_training_state)
                trainer.lr_scheduler.step.assert_called_once()
                trainer.ema_model.step.assert_called_once()
                for name, value in full_logs.items():
                    self.assertAlmostEqual(logs[name], value, places=5)

    def test_outer_gradient_accumulation_keeps_one_update_per_two_streams(self):
        policy = self.policy(horizon=6)
        reference = copy.deepcopy(policy)
        batch = self.case.fixture.make_batch(anchors=13)
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        trainer = make_trainer(policy, accelerator=accelerator)
        trainer.optimizer = accelerator.prepare_optimizer(trainer.optimizer)
        full_trainer = make_trainer(reference, accelerator=accelerator)
        initial = {name: p.detach().clone() for name, p in policy.named_parameters()}
        try:
            for index in range(2):
                sequence = dict(batch, _streaming_sequence_chunk_length=4, _streaming_sequence_length=13)
                torch.manual_seed(900 + index)
                full_loss, _ = full_trainer.compute_loss(sequence)
                (full_loss / 2).backward()
                torch.manual_seed(900 + index)
                trainer.backward_streaming_batch(batch)
                if index == 0:
                    self.assertFalse(accelerator.sync_gradients)
                    for name, p in policy.named_parameters():
                        torch.testing.assert_close(p, initial[name], rtol=0, atol=0)
            self.assertTrue(accelerator.sync_gradients)
            full_trainer.optimizer.step()
            for (name, p), (_, r) in zip(policy.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(p, r, rtol=1e-6, atol=1e-7,
                                           msg=lambda msg: name + ': ' + msg)
            trainer.ema_model.step.assert_called_once()
        finally:
            accelerator.gradient_accumulation_steps = 1

    def test_new_shadow_gates_do_not_depend_on_older_saved_gates(self):
        policy = self.policy(horizon=9)
        batch = self.case.fixture.make_batch(anchors=13)
        policy.compute_loss(self.case.chunk(batch, 0, 4, True, False))
        old = [slot.shadow_log_gate for slot in policy._streaming_training_state[0].slots
               if slot.state == 'shadow']
        old_ids = {id(gate) for gate in old}
        policy.compute_loss(self.case.chunk(batch, 4, 4, False, False))
        new = [slot.shadow_log_gate for slot in policy._streaming_training_state[0].slots
               if slot.state == 'shadow' and id(slot.shadow_log_gate) not in old_ids]
        self.assertTrue(old and new)
        gradients = torch.autograd.grad(sum(g.sum() for g in new), old, allow_unused=True)
        self.assertTrue(all(g is None for g in gradients))
        policy._streaming_training_state = None

    def test_live_saved_activation_peak_plateaus_for_longer_streams(self):
        peaks = []
        for length in [33, 97]:
            policy = self.policy(horizon=3)
            trainer = make_trainer(policy)
            batch = self.case.fixture.make_batch(anchors=length)
            tracker = _SavedTensorTracker()
            backward = trainer.accelerator.backward

            def record_backward(*args, **kwargs):
                backward(*args, **kwargs)
                gc.collect()
                tracker.after_backward.append(tracker.live)

            with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack), \
                 patch.object(trainer.accelerator, 'backward', new=record_backward):
                trainer.backward_streaming_batch(batch)
            gc.collect()
            peaks.append(tracker.peak)
            self.assertEqual(tracker.live, 0)
            self.assertGreater(max(tracker.after_backward), 0)
            del trainer, policy, batch
            SharedModelManager.reset()
        # Three times as many anchors must not retain three times the graph.
        self.assertLessEqual(peaks[1], peaks[0] * 1.15, peaks)
        print('Shadow saved-activation peaks, 33/97 anchors:', peaks)


if __name__ == '__main__':
    unittest.main()
