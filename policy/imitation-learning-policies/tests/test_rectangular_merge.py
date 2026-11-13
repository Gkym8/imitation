import copy
import gc
import unittest

from omegaconf import OmegaConf
import torch

from imitation_learning.models.common.learned_query_merger import LearnedQueryPatchMerger
from imitation_learning.models.common.spatial_merge import spatial_merge_shape
from imitation_learning.models.encoders.image_encoders import SharedModelManager
import test_future_prediction_mode as fixture


class RectangularMergeGeometryTest(unittest.TestCase):
    def test_integer_pair_and_hydra_list_validation(self):
        for value, expected in ((4, (4, 4)), ([4, 8], (4, 8)), ((8, 4), (8, 4)),
                                (OmegaConf.create([4, 8]), (4, 8))):
            self.assertEqual(spatial_merge_shape(value), expected)
        for value in (True, 0, -1, 2.5, '4,8', [], [4], [4, 8, 2], [4, 0], [True, 8], [4, 2.5]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                spatial_merge_shape(value)

    def test_rectangular_group_order_and_masks_do_not_cross_views(self):
        for shape, indices in (
            ([1, 2], [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]),
            ([2, 1], [[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]]),
        ):
            expected = torch.tensor(indices + [[i + 16 for i in row] for row in indices]).unsqueeze(0)
            for cls in (LearnedQueryPatchMerger,):
                with self.subTest(shape=shape, cls=cls.__name__):
                    args = (1, 8, 16, shape) + (2,)
                    merger = cls(*args)
                    raw = torch.arange(32).reshape(1, 32, 1)
                    torch.testing.assert_close(merger._group(raw).reshape(1, 16, 2), expected)
                    mask = torch.ones(1, 32, dtype=torch.bool)
                    mask[0, 5] = mask[0, 21] = False
                    expected_mask = (expected != 5).all(-1) & (expected != 21).all(-1)
                    torch.testing.assert_close(merger.merge_mask(mask), expected_mask)

    def test_square_integer_and_pair_keep_identical_weights_outputs_and_gradients(self):
        for cls in (LearnedQueryPatchMerger,):
            with self.subTest(cls=cls.__name__):
                tail = (4,)
                torch.manual_seed(10)
                integer = cls(8, 32, 256, 4, *tail)
                torch.manual_seed(10)
                square_pair = cls(8, 32, 256, [4, 4], *tail)
                self.assertEqual(integer.state_dict().keys(), square_pair.state_dict().keys())
                for name, value in integer.state_dict().items():
                    torch.testing.assert_close(square_pair.state_dict()[name], value, rtol=0, atol=0)
                x = torch.randn(2, 512, 8)
                state = torch.randn(2, 1, 32)
                args = ()
                first, second = integer(x, *args), square_pair(x, *args)
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                first.square().sum().backward()
                second.square().sum().backward()
                for left, right in zip(integer.parameters(), square_pair.parameters()):
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_both_sides_must_divide_grid_not_just_the_area(self):
        for shape in ([2, 32], [3, 8], [8, 3]):
            for cls in (LearnedQueryPatchMerger,):
                with self.subTest(shape=shape, cls=cls.__name__), self.assertRaisesRegex(ValueError, 'divisible'):
                    args = (8, 32, 256, shape) + ((4,))
                    cls(*args)


class RectangularMergePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FuturePredictionModeTest.setUpClass()

    def setUp(self):
        self.helper = fixture.FuturePredictionModeTest()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def config(self, merging='learned', shape=(4, 8), cached=True, mode='pooled', views=1):
        cfg = self.helper.make_config(mode, cached, merge_size=list(shape))
        cfg.visual_encoder.learned_query_merge = merging == 'learned'
        cfg.workspace.train_dataset.image_keys = ['head_camera', 'left_camera', 'right_camera'][:views]
        # Rectangular pooling consumes full patch caches for every view.
        cfg.workspace.train_dataset.cached_image_feature_kind = 'patches'
        cfg.workspace.model.denoising_network_partial.state_as_dit_token = True
        return cfg

    def batch(self, cfg, anchors=3):
        cached = cfg.workspace.train_dataset.use_cached_image_features
        batch = self.helper.make_batch(cached=cached, anchors=anchors)
        suffix = '_feature' if cached else ''
        for camera in cfg.workspace.train_dataset.image_keys[1:]:
            batch[camera + suffix] = batch['head_camera' + suffix].clone() + 0.1
        return batch

    def check_policy_case(self, merging, shape, cached, mode, views):
        cfg = self.config(merging, shape, cached, mode, views)
        policy = self.helper.instantiate_policy(cfg)
        network = policy.denoising_network
        batch = self.batch(cfg)
        self.assertEqual(network.global_cond_token_num, 8 * views)
        self.assertEqual(network.history_img_features_token_num, 8 * views)
        self.assertEqual(policy.global_cond_encoder.token_num, (8 if merging == 'mean' else 256) * views)
        self.assertEqual(policy.future_img_feature_encoder.token_num, (8 if mode == 'tokens' else 1) * views)
        if mode == 'tokens':
            _, targets = policy._encode_input_multi_traj(batch)
            raw = batch['head_camera_feature'][..., 1:, :] if cached else None
            if raw is not None:
                grid = raw.reshape(1, 3, 1, 16, 16, 8)
                height, width = shape
                expected = torch.stack([
                    grid[..., r:r + height, c:c + width, :].mean(dim=(-3, -2))
                    for r in range(0, 16, height) for c in range(0, 16, width)
                ], dim=-2).squeeze(2)
                torch.testing.assert_close(targets['future_features'][:, :-1, :8], expected[:, 1:])
        losses = policy.compute_loss(batch)
        total = losses['action'] + 0.05 * losses['future_prediction']
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None))
        if merging == 'learned':
            self.assertGreater(float(network.global_cond_projector.query_tokens.grad.abs().sum()), 0)
        for backbone in policy.shared_model_manager.vit_models.values():
            self.assertTrue(all(p.grad is None for p in backbone.parameters()))
        policy.eval()
        suffix = '_feature' if cached else ''
        for anchor in range(2):
            obs = {camera + suffix: batch[camera + suffix][:, anchor] for camera in cfg.workspace.train_dataset.image_keys}
            obs.update(agent_pos=batch['agent_pos'][:, anchor], episode_idx=torch.tensor([0]))
            with torch.inference_mode():
                output = policy.predict_action(obs)
            self.assertEqual(output['action'].shape, (1, 16, 14))
        self.assertTrue(all(slot.latent.shape == (8 * views, 32) for slot in policy.online_memory_banks[0].slots))

    def test_all_merging_modes_cached_online_targets_orientations_and_views(self):
        cases = (
            ('mean', (4, 8), True, 'tokens', 1),
            ('mean', (8, 4), False, 'pooled', 3),
            ('learned', (4, 8), True, 'tokens', 3),
            ('learned', (8, 4), False, 'pooled', 1),
            ('learned', (4, 8), True, 'pooled', 3),
            ('learned', (8, 4), False, 'tokens', 1),
        )
        for case in cases:
            with self.subTest(case=case):
                self.check_policy_case(*case)

    def test_mean_pooling_uses_exact_rectangles_and_preserves_cls(self):
        for shape in ((4, 8), (8, 4)):
            with self.subTest(shape=shape):
                policy = self.helper.instantiate_policy(self.config('mean', shape))
                patches = torch.arange(256, dtype=torch.float32).reshape(1, 256, 1).expand(-1, -1, 8)
                encoder = policy.global_cond_encoder.encoder_dict['head_camera']
                merged = encoder.merge_spatial_patches(patches)
                grid = patches.reshape(1, 16, 16, 8)
                h, w = shape
                expected = torch.stack([
                    grid[:, r:r+h, c:c+w].mean(dim=(1, 2))
                    for r in range(0, 16, h) for c in range(0, 16, w)
                ], dim=1)
                torch.testing.assert_close(merged, expected, rtol=0, atol=0)
                self.assertEqual(encoder.token_num, 8)
                self.assertEqual(policy.future_img_feature_encoder.token_num, 1)
                del policy

    def test_streaming_and_online_rectangular_history_and_shadows_agree(self):
        for merging in ('mean', 'learned'):
            with self.subTest(merging=merging):
                self.check_streaming_case(merging)

    def check_streaming_case(self, merging):
        cfg = self.config(merging, views=3)
        cfg.workspace.model.denoising_network_partial.max_history_len = 4
        cfg.workspace.model.denoising_network_partial.shadow_horizon = 2
        cfg.workspace.model.memory_eviction.keep_recent = 2
        policy = self.helper.instantiate_policy(cfg)
        batch = self.batch(cfg, anchors=13)
        for start in (0, 4, 8):
            chunk = {name: value[:, start:start + 5] for name, value in batch.items()}
            chunk.update(_streaming_supervised_length=4, _streaming_reset=start == 0, _streaming_final=False)
            losses = policy.compute_loss(chunk)
            (losses['action'] + 0.05 * losses['future_prediction']).backward()
            for slot in policy._streaming_training_state[0].slots:
                self.assertEqual(slot.latent.shape, (24, 32))
                self.assertEqual(slot.raw_features.shape, (3 * (256 if merging == 'learned' else 8), 8))
                self.assertFalse(slot.latent.requires_grad)
                self.assertIsNone(slot.raw_state)
            policy.zero_grad(set_to_none=True)
        expected = [(s.slot_id, s.state, s.expire_at) for s in policy._streaming_training_state[0].slots]
        policy.eval()
        for anchor in range(12):
            obs = {camera + '_feature': batch[camera + '_feature'][:, anchor] for camera in cfg.workspace.train_dataset.image_keys}
            obs.update(agent_pos=batch['agent_pos'][:, anchor], episode_idx=torch.tensor([0]))
            with torch.inference_mode():
                policy.predict_action(obs)
        self.assertEqual([(s.slot_id, s.state, s.expire_at) for s in policy.online_memory_banks[0].slots], expected)

    def test_rectangular_config_checkpoint_roundtrip_and_full_workspace_resolution(self):
        for merging in ('mean', 'learned'):
            with self.subTest(merging=merging):
                self.check_roundtrip_case(merging)

    def check_roundtrip_case(self, merging):
        cfg = self.config(merging, mode='tokens')
        cfg.run_name = 'rectangular-config-roundtrip'
        cfg.project_name = cfg.logger_project_name = 'rmbench'
        cfg.train_server_name = 'localhost'
        saved_yaml = OmegaConf.to_yaml(cfg, resolve=False)
        cfg.workspace.cfg_str_unresolved = saved_yaml
        resolved = copy.deepcopy(cfg)
        OmegaConf.resolve(resolved)
        self.assertEqual(tuple(resolved.visual_encoder.spatial_merge_size), (4, 8))
        expected_encoder_size = (4, 8) if merging == 'mean' else 1
        actual = resolved.workspace.model.history_img_feature_encoder.image_encoder_partial.spatial_merge_size
        self.assertEqual(tuple(actual) if not isinstance(actual, int) else actual, expected_encoder_size)
        policy = self.helper.instantiate_policy(cfg)
        state = {name: value.detach().clone() for name, value in policy.state_dict().items()}
        expected = {name: value.detach().clone() for name, value in policy.compute_loss(self.batch(cfg)).items()}
        del policy
        restored = self.helper.instantiate_policy(OmegaConf.create(saved_yaml))
        restored.load_state_dict(state, strict=True)
        self.assertEqual(restored.denoising_network.global_cond_token_num, 8)
        for name, value in restored.compute_loss(self.batch(cfg)).items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
