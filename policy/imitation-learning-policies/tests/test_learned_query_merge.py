import copy
import gc
import unittest
from unittest.mock import patch

import torch
from omegaconf import open_dict
from transformers import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipMultiheadAttentionPoolingHead

from imitation_learning.models.common.learned_query_merger import LearnedQueryPatchMerger
from imitation_learning.models.encoders.image_encoders import SharedModelManager
import test_future_prediction_mode as fixture


class LearnedQueryMergerTest(unittest.TestCase):
    def test_matches_siglip_map_head_for_each_region(self):
        merger = LearnedQueryPatchMerger(8, 32, 256, [8, 4], 4)
        reference = SiglipMultiheadAttentionPoolingHead(SiglipVisionConfig(
            hidden_size=32, intermediate_size=128, num_attention_heads=4,
            hidden_act='gelu_pytorch_tanh', layer_norm_eps=1e-6,
        ))
        reference.attention.load_state_dict(merger.attention.state_dict())
        reference.layernorm.load_state_dict(merger.layernorm.state_dict())
        reference.mlp.fc1.load_state_dict(merger.mlp[0].state_dict())
        reference.mlp.fc2.load_state_dict(merger.mlp[2].state_dict())
        images = torch.randn(2, 2 * 256, 8)
        with torch.no_grad():
            regions = merger._group(merger.image_projector(merger.image_norm(images)))
            expected = []
            for index in range(regions.shape[1]):
                reference.probe.copy_(merger.query_tokens[index % 8][None, None])
                expected.append(reference(regions[:, index]))
            expected = torch.stack(expected, dim=1)
            actual = merger(images)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_rectangular_groups_masks_and_camera_boundaries(self):
        merger = LearnedQueryPatchMerger(1, 8, 16, [2, 1], 2)
        patches = torch.arange(32).reshape(1, 32, 1)
        expected = torch.tensor([
            [0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15],
            [16, 20], [17, 21], [18, 22], [19, 23], [24, 28], [25, 29], [26, 30], [27, 31],
        ])[None, ..., None]
        torch.testing.assert_close(merger._group(patches), expected)
        mask = torch.ones(1, 32, dtype=torch.bool)
        mask[:, 5] = False
        expected_mask = torch.ones(1, 16, dtype=torch.bool)
        expected_mask[:, 1] = False
        torch.testing.assert_close(merger.merge_mask(mask), expected_mask)

    def test_attention_is_local_and_views_are_independent(self):
        for size, count in (([8, 4], 8), ([4, 8], 8), (4, 16), (8, 4)):
            with self.subTest(size=size):
                merger = LearnedQueryPatchMerger(8, 32, 256, size, 4)
                patches = torch.randn(2, 2 * 256, 8)
                before = merger(patches)
                changed = patches.clone()
                changed[:, 256, 0] += 10
                after = merger(changed)
                difference = (after - before).abs().amax(dim=-1)
                self.assertEqual(after.shape, (2, 2 * count, 32))
                self.assertTrue(torch.equal(difference[:, :count], torch.zeros_like(difference[:, :count])))
                self.assertGreater(float(difference[:, count].max()), 0)
                self.assertTrue(torch.equal(difference[:, count + 1:], torch.zeros_like(difference[:, count + 1:])))

    def test_no_query_residual_and_detached_images_train_pooling(self):
        merger = LearnedQueryPatchMerger(8, 32, 256, [8, 4], 4)
        images = torch.randn(2, 3, 256, 8).detach()
        merger(images).square().mean().backward()
        self.assertIsNone(images.grad)
        self.assertFalse(hasattr(merger, 'state_modulation'))
        for parameter in (merger.query_tokens, merger.image_projector.weight, merger.mlp[0].weight, merger.mlp[2].weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        with torch.no_grad():
            merger.attention.out_proj.weight.zero_()
            merger.attention.out_proj.bias.zero_()
            merger.layernorm.bias.zero_()
            merger.mlp[0].bias.zero_()
            merger.mlp[2].bias.zero_()
        # Q remains nonzero, but with a zero attention output the MAP result
        # is zero. Adding the raw Q as another residual would fail this.
        torch.testing.assert_close(merger(images), torch.zeros(2, 3, 8, 32), rtol=0, atol=0)

    def test_invalid_shapes_and_sizes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'divisible'):
            LearnedQueryPatchMerger(8, 32, 256, [3, 4], 4)
        merger = LearnedQueryPatchMerger(8, 32, 256, [8, 4], 4)
        with self.assertRaisesRegex(ValueError, 'complete'):
            merger(torch.zeros(2, 255, 8))
        with self.assertRaisesRegex(ValueError, 'expects'):
            merger(torch.zeros(256, 8))


class LearnedQueryMergePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FuturePredictionModeTest.setUpClass()

    def setUp(self):
        self.helper = fixture.FuturePredictionModeTest()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def config(self, mode='pooled', cached=True, views=1, size=None):
        cfg = self.helper.make_config(mode, cached, merge_size=[8, 4] if size is None else size)
        cfg.visual_encoder.learned_query_merge = True
        cfg.workspace.train_dataset.image_keys = ['head_camera', 'left_camera', 'right_camera'][:views]
        cfg.workspace.model.denoising_network_partial.state_as_dit_token = True
        return cfg

    def batch(self, cfg, anchors=3):
        cached = cfg.workspace.train_dataset.use_cached_image_features
        batch = self.helper.make_batch(cached=cached, anchors=anchors)
        suffix = '_feature' if cached else ''
        for name in cfg.workspace.train_dataset.image_keys[1:]:
            batch[name + suffix] = batch['head_camera' + suffix].clone() + 0.1
        return batch

    def test_cached_online_multiview_future_modes_train_and_infer(self):
        for cached, mode, views, size, per_view in (
            (True, 'pooled', 1, [8, 4], 8), (False, 'pooled', 1, [4, 8], 8),
            (True, 'tokens', 1, 4, 16), (True, 'pooled', 3, 8, 4),
        ):
            with self.subTest(cached=cached, mode=mode, views=views, size=size):
                cfg = self.config(mode, cached, views, size)
                policy = self.helper.instantiate_policy(cfg)
                network = policy.denoising_network
                count = views * per_view
                self.assertTrue(network.learned_query_merge)
                self.assertEqual(policy.global_cond_encoder.token_num, 256 * views)
                self.assertEqual(network.global_cond_token_num, count)
                self.assertEqual(network.history_img_features_token_num, count)
                self.assertEqual(policy.future_img_feature_encoder.token_num, views if mode == 'pooled' else count)
                self.assertEqual(cfg.workspace.train_dataset.cached_image_feature_kind, 'patches')
                self.assertIsNone(network.history_img_features_projector)
                batch = self.batch(cfg)
                _, target = policy._encode_input_multi_traj(batch)
                target_before = target['future_features'].detach().clone()
                losses = policy.compute_loss(batch)
                (losses['action'] + 0.05 * losses['future_prediction']).backward()
                merger = network.global_cond_projector
                self.assertGreater(float(merger.query_tokens.grad.abs().sum()), 0)
                self.assertGreater(float(merger.mlp[0].weight.grad.abs().sum()), 0)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None))
                self.assertTrue(all(p.grad is None for model in policy.shared_model_manager.vit_models.values() for p in model.parameters()))
                with torch.no_grad():
                    merger.mlp[2].bias.add_(0.25)
                _, target_after = policy._encode_input_multi_traj(batch)
                torch.testing.assert_close(target_before, target_after['future_features'], rtol=0, atol=0)
                policy.eval()
                policy.reset()
                with patch.object(policy.future_feature_predictor, 'forward', side_effect=AssertionError('online future predictor')):
                    for anchor in range(2):
                        suffix = '_feature' if cached else ''
                        obs = {name + suffix: batch[name + suffix][:, anchor] for name in cfg.workspace.train_dataset.image_keys}
                        obs.update(agent_pos=batch['agent_pos'][:, anchor], episode_idx=torch.tensor([0]))
                        with torch.inference_mode():
                            output = policy.predict_action(obs)
                        self.assertEqual(output['action'].shape, (1, 16, 14))
                self.assertTrue(all(slot.latent.shape == (count, 32) and slot.raw_features is None and slot.raw_state is None
                                    for slot in policy.online_memory_banks[0].slots))
                del policy

    def test_current_and_history_share_pooling_and_frame_state(self):
        cfg = self.config(cached=False)
        policy = self.helper.instantiate_policy(cfg)
        batch = self.batch(cfg)
        raw = policy.global_cond_encoder({'head_camera': batch['head_camera']})
        cls = policy.future_img_feature_encoder({'head_camera': batch['head_camera']})
        cached = policy.global_cond_encoder({'head_camera_feature': torch.cat([cls, raw], dim=-2).unsqueeze(-3)})
        torch.testing.assert_close(raw, cached, rtol=0, atol=0)
        network = policy.denoising_network
        states = torch.randn(1, 3, 1, 8)
        current = network._project_global_cond(raw, states)
        changed_state = network._project_global_cond(raw, states + 10)
        history = network.project_history_image_features(cached)
        torch.testing.assert_close(current, changed_state, rtol=0, atol=0)
        torch.testing.assert_close(current, history, rtol=0, atol=0)
        self.assertFalse(any('state_modulation' in k or 'history_img_features_projector.' in k for k in policy.state_dict()))

    def test_streaming_reprojects_retained_patches_shadows_and_frame_states(self):
        cfg = self.config()
        cfg.workspace.model.denoising_network_partial.max_history_len = 3
        cfg.workspace.model.denoising_network_partial.shadow_horizon = 2
        cfg.workspace.model.denoising_network_partial.shadow_gate_scale = 0.5
        cfg.workspace.model.memory_eviction.keep_recent = 0
        policy = self.helper.instantiate_policy(cfg)
        batch = self.batch(cfg, anchors=9)
        for start in (0, 4):
            chunk = {key: value[:, start:start + 5] for key, value in batch.items()}
            chunk.update(_streaming_supervised_length=4, _streaming_reset=start == 0, _streaming_final=False)
            losses = policy.compute_loss(chunk)
            (losses['action'] + 0.05 * losses['future_prediction']).backward()
            bank = policy._streaming_training_state[0]
            self.assertTrue(any(slot.state == 'shadow' for slot in bank.slots))
            for slot in bank.slots:
                self.assertEqual(slot.raw_features.shape, (256, 8))
                self.assertIsNone(slot.raw_state)
                self.assertFalse(slot.raw_features.requires_grad)
            policy.zero_grad(set_to_none=True)
            if start == 0:
                before = bank.slots[0].latent.clone()
                with torch.no_grad():
                    policy.denoising_network.global_cond_projector.mlp[2].bias[0].add_(0.25)
                bank.reproject(policy.denoising_network.project_history_image_features)
                self.assertGreater(float((bank.slots[0].latent - before).abs().max()), 0)
                sum(slot.latent.square().mean() for slot in bank.slots).backward()
                self.assertGreater(float(policy.denoising_network.global_cond_projector.query_tokens.grad.abs().sum()), 0)
                bank.detach()
                policy.zero_grad(set_to_none=True)

    def test_checkpoint_roundtrip(self):
        cfg = self.config()
        policy = self.helper.instantiate_policy(cfg)
        state = copy.deepcopy(policy.state_dict())
        batch = self.batch(cfg)
        torch.manual_seed(1234)
        expected = policy.compute_loss(batch)
        del policy
        restored = self.helper.instantiate_policy(cfg)
        restored.load_state_dict(state, strict=True)
        torch.manual_seed(1234)
        for name, value in restored.compute_loss(batch).items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_rejects_integer_query_counts(self):
        for value in (0, 1, 16):
            cfg = self.config()
            cfg.visual_encoder.learned_query_merge = value
            with self.assertRaisesRegex(Exception, 'must be boolean'):
                self.helper.instantiate_policy(cfg)


if __name__ == '__main__':
    unittest.main()
