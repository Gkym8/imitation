"""State-read equation, masks, gradients and retained-frame state alignment."""
import gc
import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from imitation_learning.models.common.learned_query_merger import LearnedQueryPatchMerger
from imitation_learning.models.encoders.image_encoders import SharedModelManager
import test_future_prediction_mode as fixture


class LearnedQueryStateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(120)

    def test_matches_equation_after_image_map_with_multiple_states_and_views(self):
        merger = LearnedQueryPatchMerger(8, 32, 16, 2, 4, num_queries=8).double()
        images = torch.randn(2, 3, 32, 8, dtype=torch.float64)
        states = torch.randn(2, 3, 4, 32, dtype=torch.float64)
        visual = merger._pool_images(images)
        q = F.linear(visual, merger.state_query.weight)
        k = F.linear(states, merger.state_key.weight)
        v = F.linear(states, merger.state_value.weight)
        expected = F.layer_norm(
            ((q @ k.transpose(-2, -1)) / math.sqrt(32)).softmax(-1) @ v + visual,
            (32,), merger.state_norm.weight, merger.state_norm.bias, merger.state_norm.eps,
        )
        with patch.object(merger.state_query, 'forward', wraps=merger.state_query.forward) as query:
            actual = merger(images, states)
        self.assertEqual(actual.shape, (2, 3, 16, 32))
        torch.testing.assert_close(query.call_args.args[0], visual, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        changed = states.clone()
        changed[:, 1, :, 0] += 3
        after = merger(images, changed)
        torch.testing.assert_close(after[:, [0, 2]], actual[:, [0, 2]], rtol=0, atol=0)
        self.assertTrue(((after[:, 1] - actual[:, 1]).abs().amax(-1) > 1e-6).all())

    def test_single_state_is_shared_value_plus_individual_visual_residuals(self):
        merger = LearnedQueryPatchMerger(8, 32, 16, 2, 4, num_queries=8)
        images, states = torch.randn(2, 16, 8), torch.randn(2, 1, 32)
        visual = merger._pool_images(images)
        expected = merger.state_norm(visual + merger.state_value(states))
        torch.testing.assert_close(merger(images, states), expected, rtol=0, atol=0)
        with torch.no_grad():
            merger.state_query.weight.mul_(5)
            merger.state_key.weight.mul_(7)
        torch.testing.assert_close(merger(images, states), expected, rtol=0, atol=0)

    def test_masks_exclude_nan_padding_and_all_masked_states_are_finite(self):
        merger = LearnedQueryPatchMerger(8, 32, 16, 2, 4, num_queries=8)
        images = torch.randn(2, 16, 8)
        states = torch.randn(2, 3, 32)
        mask = torch.tensor([[True, False, True], [False, False, False]])
        padded = states.masked_fill(~mask[..., None], float('nan'))
        actual = merger(images, padded, mask)
        torch.testing.assert_close(actual[:1], merger(images[:1], states[:1, [0, 2]]))
        expected_empty = merger.state_norm(merger._pool_images(images[1:]))
        torch.testing.assert_close(actual[1:], expected_empty, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(actual).all())
        (actual * torch.randn_like(actual)).sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in merger.parameters() if p.grad is not None))

    def test_gradients_reach_image_queries_and_state_qkv(self):
        merger = LearnedQueryPatchMerger(8, 32, 16, 2, 4, num_queries=8)
        images = torch.randn(2, 16, 8).detach()
        states = torch.randn(2, 3, 32, requires_grad=True)
        output = merger(images, states)
        (output * torch.randn_like(output)).sum().backward()
        self.assertIsNone(images.grad)
        for parameter in (merger.query_tokens, merger.state_query.weight, merger.state_key.weight,
                          merger.state_value.weight, merger.state_norm.weight, states):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_bfloat16_autocast_preserves_dtype_and_finite_backward(self):
        merger = LearnedQueryPatchMerger(8, 32, 16, 2, 4, num_queries=8)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = merger(torch.randn(2, 16, 8), torch.randn(2, 2, 32))
            loss = (output * torch.randn_like(output)).sum()
        self.assertEqual(output.dtype, torch.bfloat16)
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in merger.parameters() if p.grad is not None))

    def test_missing_or_misaligned_state_is_rejected(self):
        merger = LearnedQueryPatchMerger(8, 32, 16, 2, 4, num_queries=8)
        images = torch.randn(2, 16, 8)
        with self.assertRaisesRegex(ValueError, 'own state'):
            merger(images)
        for states in (torch.randn(1, 1, 32), torch.randn(2, 32),
                       torch.randn(2, 1, 8), torch.randn(2, 0, 32)):
            with self.subTest(shape=states.shape), self.assertRaisesRegex(ValueError, 'state must match'):
                merger(images, states)
        with self.assertRaisesRegex(ValueError, 'state_mask'):
            merger(images, torch.randn(2, 1, 32), torch.ones(2, 2))


class LearnedQueryStatePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FuturePredictionModeTest.setUpClass()

    def setUp(self):
        self.helper = fixture.FuturePredictionModeTest()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def config(self):
        cfg = self.helper.make_config(cached=True, merge_size=[4, 8])
        cfg.visual_encoder.learned_query_merge = 8
        cfg.workspace.model.denoising_network_partial.state_as_dit_token = True
        return cfg

    def test_streaming_reencodes_retained_states_and_matches_online_shadow_history(self):
        cfg = self.config()
        cfg.workspace.model.denoising_network_partial.max_history_len = 4
        cfg.workspace.model.denoising_network_partial.shadow_horizon = 2
        cfg.workspace.model.memory_eviction.keep_recent = 2
        policy = self.helper.instantiate_policy(cfg)
        batch = self.helper.make_batch(anchors=13)
        for start in (0, 4, 8):
            chunk = {key: value[:, start:start + 5] for key, value in batch.items()}
            chunk.update(_streaming_supervised_length=4, _streaming_reset=start == 0, _streaming_final=False)
            losses = policy.compute_loss(chunk)
            (losses['action'] + 0.05 * losses['future_prediction']).backward()
            bank = policy._streaming_training_state[0]
            for slot in bank.slots:
                torch.testing.assert_close(slot.raw_state['agent_pos'],
                                           batch['agent_pos'][0, slot.frame_index // 10], rtol=0, atol=0)
                self.assertFalse(slot.raw_state['agent_pos'].requires_grad)
                self.assertFalse(slot.latent.requires_grad)
            policy.zero_grad(set_to_none=True)
        expected = [(s.slot_id, s.state, s.expire_at) for s in bank.slots]
        self.assertEqual(bank.shadow_count, 2)
        expected_latents = {s.slot_id: s.latent.clone() for s in bank.slots}
        policy.eval()
        for anchor in range(12):
            with torch.inference_mode():
                policy.predict_action({
                    'head_camera_feature': batch['head_camera_feature'][:, anchor],
                    'agent_pos': batch['agent_pos'][:, anchor], 'episode_idx': torch.tensor([0]),
                })
        online = policy.online_memory_banks[0]
        self.assertEqual([(s.slot_id, s.state, s.expire_at) for s in online.slots], expected)
        for slot in online.slots:
            torch.testing.assert_close(slot.latent, expected_latents[slot.slot_id], rtol=1e-4, atol=1e-5)
            self.assertIsNone(slot.raw_features)
            self.assertIsNone(slot.raw_state)
        before = bank.slots[0].latent.clone()
        with torch.no_grad():
            policy.local_cond_encoder.encoder_dict['agent_pos'].model.model.weight[0].add_(0.1)
        bank.reproject(policy._reproject_state_query_history)
        self.assertGreater(float((bank.slots[0].latent - before).abs().max()), 1e-5)
        sum((s.latent * torch.randn_like(s.latent)).sum() for s in bank.slots).backward()
        grad = policy.local_cond_encoder.encoder_dict['agent_pos'].model.model.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0)

    def test_current_image_state_mask_matches_history_and_hides_padding(self):
        policy = self.helper.instantiate_policy(self.config())
        network = policy.denoising_network
        images = torch.randn(2, 256, 8)
        states = torch.randn(2, 3, 8)
        mask = torch.tensor([[True, False, True], [False, False, False]])
        padded = states.masked_fill(~mask[..., None], float('nan'))
        current = network._project_global_cond(images, padded, mask)
        history = network.project_history_image_features(images, states, mask)
        torch.testing.assert_close(current, history, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(current).all())


if __name__ == '__main__':
    unittest.main()
