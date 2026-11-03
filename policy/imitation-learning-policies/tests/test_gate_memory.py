import gc
import unittest
from unittest.mock import patch

import torch
from torch import nn

from imitation_learning.models.memory.gate_memory import (
    GateMemoryBank,
    gate_selectivity_penalty,
)
from imitation_learning.models.encoders.image_encoders import SharedModelManager
from imitation_learning.trainers.policy_trainer import PolicyTrainer
import test_future_prediction_mode as future_fixture


class GateMemoryBankTest(unittest.TestCase):
    def fill(self, bank):
        for i in range(bank.capacity):
            bank.append(torch.full((2, 3), float(i)), i * 10, torch.empty(0))

    def test_eviction_respects_protection_and_keeps_features_aligned(self):
        bank = GateMemoryBank(5, keep_first=True, keep_recent=2)
        self.fill(bank)
        # The first and most recent slots have the lowest gates, but both are
        # protected. Slot 2 is the minimum among the eligible slots.
        removed = bank.append(torch.full((2, 3), 5.0), 50, torch.tensor([0.01, 0.8, 0.2, 0.4, 0.001]))
        self.assertEqual(removed, 2)
        self.assertEqual([slot.slot_id for slot in bank.slots], [0, 1, 3, 4, 5])
        self.assertEqual([slot.frame_index for slot in bank.slots], [0, 10, 30, 40, 50])
        self.assertEqual([float(slot.latent[0, 0]) for slot in bank.slots], [0, 1, 3, 4, 5])
        self.assertEqual(len(bank), bank.capacity)

    def test_ties_evict_oldest_and_new_slot_always_enters(self):
        bank = GateMemoryBank(3, keep_first=False, keep_recent=0)
        self.fill(bank)
        bank.append(torch.ones(2, 3), 30, torch.ones(3))
        self.assertEqual([slot.slot_id for slot in bank.slots], [1, 2, 3])
        # Scores must be supplied freshly for the current read. No stale
        # per-slot utility weights are stored or accumulated by the bank.
        bank.append(torch.ones(2, 3), 40, torch.tensor([0.9, 0.8, 0.1]))
        self.assertEqual([slot.slot_id for slot in bank.slots], [1, 2, 4])

    def test_invalid_protection_is_rejected_without_overriding_it(self):
        with self.assertRaisesRegex(ValueError, "eviction candidate"):
            GateMemoryBank(3, keep_first=True, keep_recent=3)

    def test_keep_recent_two_includes_incoming_slot_with_and_without_shadow(self):
        for horizon in (0, 4):
            with self.subTest(shadow_horizon=horizon):
                bank = GateMemoryBank(8, keep_first=True, keep_recent=2, shadow_horizon=horizon)
                self.fill(bank)
                removed = bank.append(
                    torch.full((2, 3), 8.0), 80,
                    torch.tensor([0.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.01, 0.0]),
                )
                # Slot 7 is protected despite its lower gate. Slot 6 is not:
                # the two recent slots after insertion are 7 and 8.
                self.assertEqual(removed, 6)
                self.assertEqual(
                    [s.slot_id for s in bank.slots if s.state == "active"],
                    [0, 1, 2, 3, 4, 5, 7, 8],
                )
                scores = torch.tensor([
                    0.0 if s.slot_id in (0, 6, 8) else 0.01 if s.slot_id == 7 else 0.5
                    for s in bank.slots
                ])
                # Next insertion protects 8 and 9, so 7 can now be demoted.
                self.assertEqual(bank.append(torch.full((2, 3), 9.0), 90, scores), 7)
                self.assertEqual(
                    [s.slot_id for s in bank.slots if s.state == "active"],
                    [0, 1, 2, 3, 4, 5, 8, 9],
                )
                self.assertEqual(
                    [s.slot_id for s in bank.slots if s.state == "shadow"],
                    [6, 7] if horizon else [],
                )

    def test_recent_protection_boundary_counts_incoming_slot(self):
        for keep_first, keep_recent, evicted in ((True, 2, 1), (False, 3, 0)):
            with self.subTest(keep_first=keep_first, keep_recent=keep_recent):
                bank = GateMemoryBank(3, keep_first=keep_first, keep_recent=keep_recent)
                self.fill(bank)
                self.assertEqual(bank.append(torch.ones(2, 3), 30, torch.zeros(3)), evicted)
                self.assertEqual(bank.active_count, 3)

    def test_detached_features_reproject_with_current_parameters(self):
        bank = GateMemoryBank(2, keep_first=False, keep_recent=0)
        projector = nn.Linear(3, 4, bias=False)
        raw = torch.randn(2, 3, requires_grad=True)
        bank.append(projector(raw), 0, torch.empty(0), raw_features=raw)
        bank.detach()
        self.assertIsNone(bank.slots[0].latent.grad_fn)
        self.assertFalse(bank.slots[0].raw_features.requires_grad)
        with torch.no_grad():
            projector.weight.add_(0.3)
        bank.reproject(projector)
        torch.testing.assert_close(bank.slots[0].latent, projector(raw.detach()))
        bank.slots[0].latent.sum().backward()
        self.assertGreater(float(projector.weight.grad.abs().sum()), 0)
        self.assertIsNone(raw.grad)


class GateSelectivityTest(unittest.TestCase):
    def test_std_is_across_slots_not_across_heads(self):
        # Different heads alone do not satisfy selectivity.
        constant_slots = torch.tensor([[[0.1, 0.9], [0.1, 0.9]]])
        penalty, eligible = gate_selectivity_penalty(
            constant_slots.log(), torch.ones(1, 2, dtype=torch.bool), 2, 0.1,
        )
        self.assertTrue(eligible.all())
        torch.testing.assert_close(penalty, torch.tensor([0.01]))
        # Heads may agree, provided each distinguishes historical slots.
        distinct_slots = torch.tensor([[[0.2, 0.2], [0.8, 0.8]]])
        penalty, _ = gate_selectivity_penalty(
            distinct_slots.log(), torch.ones(1, 2, dtype=torch.bool), 2, 0.1,
        )
        torch.testing.assert_close(penalty, torch.zeros(1))

    def test_population_std_and_padding_mask(self):
        gates = torch.tensor([[[0.4], [0.6], [0.999]]])
        mask = torch.tensor([[True, True, False]])
        penalty, eligible = gate_selectivity_penalty(gates.log(), mask, 2, 0.2)
        self.assertTrue(eligible.all())
        # Population std of [0.4, 0.6] is 0.1, so the penalty is 0.01.
        torch.testing.assert_close(penalty, torch.tensor([0.01]))
        changed = gates.clone()
        changed[:, 2] = 0.001
        other, _ = gate_selectivity_penalty(changed.log(), mask, 2, 0.2)
        torch.testing.assert_close(other, penalty)

    def test_unfilled_and_empty_banks_are_ineligible_and_backward_is_finite(self):
        log_gates = torch.zeros(2, 3, 4, requires_grad=True)
        mask = torch.tensor([[True, False, False], [False, False, False]])
        penalty, eligible = gate_selectivity_penalty(log_gates, mask, 3, 0.1)
        self.assertFalse(eligible.any())
        loss = (penalty * eligible).sum()
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertTrue(torch.isfinite(log_gates.grad).all())

    def test_subthreshold_nonconstant_gates_receive_gradients(self):
        log_gates = torch.tensor([[[0.49, 0.51], [0.51, 0.49]]]).log().requires_grad_()
        penalty, _ = gate_selectivity_penalty(
            log_gates, torch.ones(1, 2, dtype=torch.bool), 2, 0.1,
        )
        penalty.sum().backward()
        self.assertGreater(float(log_gates.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(log_gates.grad).all())


class GateMemoryPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        future_fixture.FuturePredictionModeTest.setUpClass()

    def setUp(self):
        self.fixture = future_fixture.FuturePredictionModeTest()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def chunk(self, batch, start, length, reset, final):
        chunk = {key: value[:, start:start + length + 1] for key, value in batch.items()}
        chunk.update({
            "_streaming_supervised_length": length,
            "_streaming_reset": reset,
            "_streaming_final": final,
        })
        return chunk

    def test_training_and_online_keep_identical_slots_through_evictions(self):
        cfg = self.fixture.make_config("tokens")
        cfg.workspace.model.denoising_network_partial.max_history_len = 4
        cfg.workspace.model.memory_eviction.keep_recent = 2
        policy = self.fixture.instantiate_policy(cfg).eval()
        batch = self.fixture.make_batch(anchors=10)
        policy.compute_loss(batch)
        training_ids = policy.latest_training_slot_ids[0]
        state = policy.state_dict()
        del policy
        policy = self.fixture.instantiate_policy(cfg).eval()
        policy.load_state_dict(state)
        for anchor in range(10):
            with torch.inference_mode():
                policy.predict_action({
                    "head_camera_feature": batch["head_camera_feature"][:, anchor],
                    "agent_pos": batch["agent_pos"][:, anchor],
                    "episode_idx": torch.tensor([0]),
                })
            self.assertLessEqual(len(policy.online_memory_banks[0]), 4)
        self.assertEqual(tuple(slot.slot_id for slot in policy.online_memory_banks[0].slots), training_ids)
        self.assertEqual(training_ids[0], 0)

    def test_future_lookahead_does_not_change_eviction_or_enter_bank(self):
        policy = self.fixture.instantiate_policy(self.fixture.make_config()).eval()
        batch = self.fixture.make_batch(anchors=7)
        losses = policy.compute_loss(self.chunk(batch, 0, 6, True, True))
        ids = policy.latest_training_slot_ids
        changed = {key: value.clone() for key, value in batch.items()}
        changed["head_camera_feature"][:, -1] += 100
        other = policy.compute_loss(self.chunk(changed, 0, 6, True, True))
        self.assertEqual(ids, policy.latest_training_slot_ids)
        self.assertTrue(all(slot_id < 6 for slot_id in ids[0]))
        torch.testing.assert_close(losses["gate_selectivity"], other["gate_selectivity"])

    def test_regularizer_is_invariant_to_chunk_partition(self):
        policy = self.fixture.instantiate_policy(self.fixture.make_config()).eval()
        batch = self.fixture.make_batch(anchors=7)
        full = policy.compute_loss(self.chunk(batch, 0, 6, True, True))["gate_selectivity"]
        first = policy.compute_loss(self.chunk(batch, 0, 3, True, False))["gate_selectivity"]
        second = policy.compute_loss(self.chunk(batch, 3, 3, False, True))["gate_selectivity"]
        torch.testing.assert_close(full, (first + second) / 2)

    def test_selectivity_loss_alone_trains_gate_without_retaining_log_gate_records(self):
        policy = self.fixture.instantiate_policy(self.fixture.make_config())
        loss = policy.compute_loss(self.fixture.make_batch(anchors=6))["gate_selectivity"]
        self.assertGreater(float(loss), 0)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in policy.denoising_network.memory_retriever.slot_gate.parameters()
        ]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
        self.assertNotIn("slot_gate_training_log_gates", policy.denoising_network.recorded_data_dict)

    def test_padding_does_not_advance_bank_or_apply_regularizer(self):
        policy = self.fixture.instantiate_policy(self.fixture.make_config())
        batch = self.fixture.make_batch(anchors=5)
        batch["entire_traj_is_padding"][:, 1:] = True
        losses = policy.compute_loss(batch)
        self.assertEqual(policy.latest_training_slot_ids, ((0,),))
        self.assertEqual(float(losses["gate_selectivity"]), 0)
        total = losses["action"] + losses["future_prediction"] + losses["gate_selectivity"]
        total.backward()
        for parameter in policy.denoising_network.memory_retriever.slot_gate.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_online_never_calls_future_prediction_or_target_encoder(self):
        policy = self.fixture.instantiate_policy(self.fixture.make_config()).eval()
        batch = self.fixture.make_batch(anchors=5)
        with patch.object(policy.future_feature_predictor, "forward", side_effect=AssertionError("future prediction during inference")), \
             patch.object(policy.future_img_feature_encoder, "forward", side_effect=AssertionError("future encoder during inference")):
            for anchor in range(5):
                with torch.inference_mode():
                    output = policy.predict_action({
                        "head_camera_feature": batch["head_camera_feature"][:, anchor],
                        "agent_pos": batch["agent_pos"][:, anchor],
                        "episode_idx": torch.tensor([0]),
                    })
                self.assertTrue(torch.isfinite(output["action"]).all())
        policy.reset()
        self.assertEqual(policy.online_memory_banks, {})


class GateTrainerLossTest(unittest.TestCase):
    def test_weighted_gate_loss_is_logged_and_does_not_mutate_action_loss(self):
        values = {
            "action": torch.tensor(1.0, requires_grad=True),
            "future_prediction": torch.tensor(2.0, requires_grad=True),
            "gate_selectivity": torch.tensor(3.0, requires_grad=True),
        }
        trainer = PolicyTrainer.__new__(PolicyTrainer)
        trainer.model = lambda batch: values
        trainer.future_prediction_loss_weight = 0.05
        trainer.gate_selectivity_loss_weight = 0.01
        trainer.critical_action_loss_weights = [0.0]
        trainer.epoch = 0
        loss, logs = trainer.compute_loss({})
        torch.testing.assert_close(loss, torch.tensor(1.13))
        self.assertEqual(float(values["action"]), 1)
        self.assertEqual(logs["train/gate_selectivity_loss"], 3)
        loss.backward()
        torch.testing.assert_close(values["gate_selectivity"].grad, torch.tensor(0.01))


if __name__ == "__main__":
    unittest.main()
