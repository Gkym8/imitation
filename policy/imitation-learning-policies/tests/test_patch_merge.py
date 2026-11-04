import copy
import unittest

import torch

from imitation_learning.models.common.patch_merger import SpatialPatchMerger
from imitation_learning.models.encoders.image_encoders import SharedModelManager
import test_future_prediction_mode as fixture


class SpatialPatchMergerTest(unittest.TestCase):
    def test_spatial_groups_preserve_local_order_and_view_boundaries(self):
        merger = SpatialPatchMerger(1, 4, 16, 2)
        x = torch.arange(32).reshape(1, 32, 1)
        grouped = merger._group(x)
        expected = torch.tensor([
            [0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15],
            [16, 17, 20, 21], [18, 19, 22, 23], [24, 25, 28, 29], [26, 27, 30, 31],
        ]).unsqueeze(0)
        torch.testing.assert_close(grouped, expected)
        mask = torch.ones(1, 32, dtype=torch.bool)
        mask[0, 5] = False
        self.assertEqual(merger.merge_mask(mask).tolist(), [[False, True, True, True, True, True, True, True]])
        with self.assertRaisesRegex(ValueError, 'complete'):
            merger(torch.zeros(1, 15, 1))

    def test_detached_backbone_features_still_train_merger(self):
        merger = SpatialPatchMerger(8, 32, 256, 8)
        raw = torch.randn(2, 256, 8).detach()
        y = merger(raw)
        self.assertEqual(y.shape, (2, 4, 32))
        y.square().mean().backward()
        self.assertIsNone(raw.grad)
        self.assertGreater(float(merger.mlp[0].weight.grad.abs().sum()), 0)


class PatchMergePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FuturePredictionModeTest.setUpClass()

    def setUp(self):
        self.fixture = fixture.FuturePredictionModeTest()

    def tearDown(self):
        SharedModelManager.reset()

    def config(self, mode='pooled', cached=True, views=1):
        cfg = self.fixture.make_config(mode, cached, merge_size=8)
        cfg.visual_encoder.patch_merge = True
        cfg.workspace.train_dataset.image_keys = ['head_camera', 'left_camera', 'right_camera'][:views]
        return cfg

    def test_cached_and_online_training_and_both_future_targets(self):
        for cached, mode, views in ((True, 'pooled', 1), (False, 'pooled', 1),
                                    (True, 'tokens', 1), (True, 'pooled', 3)):
            with self.subTest(cached=cached, mode=mode, views=views):
                cfg = self.config(mode, cached, views)
                policy = self.fixture.instantiate_policy(cfg)
                network = policy.denoising_network
                self.assertTrue(network.patch_merge)
                self.assertEqual(policy.global_cond_encoder.token_num, 256 * views)
                self.assertEqual(network.global_cond_token_num, 4 * views)
                self.assertEqual(network.history_img_features_token_num, 4 * views)
                self.assertEqual(policy.future_img_feature_encoder.token_num, views if mode == 'pooled' else 4 * views)
                self.assertEqual(cfg.workspace.train_dataset.cached_image_feature_kind, 'patches')
                self.assertIsNone(network.history_img_features_projector)
                batch = self.fixture.make_batch(cached=cached, anchors=3)
                for camera in cfg.workspace.train_dataset.image_keys[1:]:
                    suffix = '_feature' if cached else ''
                    batch[camera + suffix] = batch['head_camera' + suffix].clone() + 0.1
                _, before_targets = policy._encode_input_multi_traj(batch)
                future_targets = before_targets['future_features'].detach().clone()
                losses = policy.compute_loss(batch)
                total = losses['action'] + 0.05 * losses['future_prediction']
                self.assertTrue(torch.isfinite(total))
                total.backward()
                self.assertGreater(float(network.global_cond_projector.mlp[0].weight.grad.abs().sum()), 0)
                for backbone in policy.shared_model_manager.vit_models.values():
                    self.assertTrue(all(p.grad is None for p in backbone.parameters()))
                with torch.no_grad():
                    network.global_cond_projector.mlp[-1].bias.add_(0.1)
                _, after_targets = policy._encode_input_multi_traj(batch)
                torch.testing.assert_close(future_targets, after_targets['future_features'])
                state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
                self.assertTrue(any('global_cond_projector.mlp.' in k for k in state))
                self.assertFalse(any('history_img_features_projector.' in k for k in state))
                policy.load_state_dict(state, strict=True)
                policy.eval()
                policy.reset()
                for anchor in range(2):
                    obs = {k: v[:, anchor] for k, v in batch.items() if k.endswith('_feature') or k in cfg.workspace.train_dataset.image_keys or k == 'agent_pos'}
                    obs['episode_idx'] = torch.tensor([0])
                    with torch.inference_mode():
                        output = policy.predict_action(obs)
                    self.assertEqual(output['action'].shape, (1, 16, 14))
                bank = policy.online_memory_banks[0]
                self.assertTrue(all(slot.latent.shape == (4 * views, 32) for slot in bank.slots))
                self.assertTrue(all(slot.raw_features is None for slot in bank.slots))
                del policy

    def test_disabled_and_missing_flag_have_identical_weights_and_losses(self):
        cfg = self.fixture.make_config()
        policy = self.fixture.instantiate_policy(cfg)
        state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        losses = {k: v.detach().clone() for k, v in policy.compute_loss(self.fixture.make_batch()).items()}
        self.assertFalse(policy.denoising_network.patch_merge)
        del policy
        legacy = copy.deepcopy(cfg)
        legacy.workspace.train_dataset.cached_image_feature_kind = 'auto'
        del legacy.visual_encoder.patch_merge
        del legacy.workspace.model.denoising_network_partial.patch_merge
        del legacy.workspace.model.denoising_network_partial.patch_merge_size
        policy = self.fixture.instantiate_policy(legacy)
        policy.load_state_dict(state, strict=True)
        for key, value in policy.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        for key, value in policy.compute_loss(self.fixture.make_batch()).items():
            torch.testing.assert_close(value, losses[key], rtol=0, atol=0)

    def test_raw_and_cached_features_produce_identical_merged_tokens(self):
        policy = self.fixture.instantiate_policy(self.config(cached=False))
        batch = self.fixture.make_batch(cached=False)
        images = {'head_camera': batch['head_camera']}
        raw = policy.global_cond_encoder(images)
        cls = policy.future_img_feature_encoder(images)
        cache = torch.cat([cls, raw], dim=-2).unsqueeze(-3)
        cached = policy.global_cond_encoder({'head_camera_feature': cache})
        torch.testing.assert_close(raw, cached, rtol=0, atol=0)
        network = policy.denoising_network
        torch.testing.assert_close(network.global_cond_projector(raw), network.project_history_image_features(cached), rtol=0, atol=0)

    def test_new_checkpoint_roundtrip_preserves_merger_outputs(self):
        cfg = self.config()
        policy = self.fixture.instantiate_policy(cfg)
        merger = policy.denoising_network.global_cond_projector
        with torch.no_grad():
            merger.mlp[-1].bias.add_(0.25)
        raw = torch.randn(2, 256, 8)
        expected = merger(raw).detach()
        state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        del merger, policy
        restored = self.fixture.instantiate_policy(cfg)
        restored.load_state_dict(state, strict=True)
        torch.testing.assert_close(restored.denoising_network.global_cond_projector(raw), expected, rtol=0, atol=0)

    def test_streaming_reprojects_full_detached_history_after_parameter_update(self):
        policy = self.fixture.instantiate_policy(self.config())
        batch = self.fixture.make_batch(anchors=5)
        for start, length, final in ((0, 2, False), (2, 2, True)):
            chunk = {key: value[:, start:start + length + 1] for key, value in batch.items()}
            chunk.update(_streaming_supervised_length=length, _streaming_reset=start == 0, _streaming_final=final)
            losses = policy.compute_loss(chunk)
            (losses['action'] + 0.05 * losses['future_prediction']).backward()
            merger = policy.denoising_network.global_cond_projector
            self.assertGreater(float(merger.mlp[0].weight.grad.abs().sum()), 0)
            if not final:
                bank = policy._streaming_training_state[0]
                for slot in bank.slots:
                    self.assertEqual(slot.raw_features.shape, (256, 8))
                    self.assertFalse(slot.raw_features.requires_grad)
                    self.assertEqual(slot.latent.shape, (4, 32))
                    self.assertFalse(slot.latent.requires_grad)
                with torch.no_grad():
                    merger.mlp[-1].bias.add_(0.2)
                bank.reproject(policy.denoising_network.project_history_image_features)
                self.assertTrue(bank.slots[0].latent.requires_grad)
                torch.testing.assert_close(bank.slots[0].latent, merger(bank.slots[0].raw_features))
                bank.detach()
            policy.zero_grad(set_to_none=True)

    def test_shadow_slots_keep_full_features_during_training(self):
        cfg = self.config()
        cfg.workspace.model.denoising_network_partial.max_history_len = 4
        cfg.workspace.model.denoising_network_partial.shadow_horizon = 2
        policy = self.fixture.instantiate_policy(cfg)
        batch = self.fixture.make_batch(anchors=8)
        batch.update(_streaming_supervised_length=7, _streaming_reset=True, _streaming_final=False)
        losses = policy.compute_loss(batch)
        (losses['action'] + 0.05 * losses['future_prediction']).backward()
        bank = policy._streaming_training_state[0]
        self.assertEqual(bank.active_count, 4)
        self.assertEqual(bank.shadow_count, 2)
        self.assertTrue(all(slot.raw_features.shape == (256, 8) for slot in bank.slots))


if __name__ == '__main__':
    unittest.main()
