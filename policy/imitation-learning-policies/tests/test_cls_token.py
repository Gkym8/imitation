import copy
import gc
import unittest

import torch

from imitation_learning.models.common.learned_query_merger import LearnedQueryPatchMerger
from imitation_learning.models.encoders.image_encoders import SharedModelManager
import test_future_prediction_mode as fixture


class ClsTokenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FuturePredictionModeTest.setUpClass()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def config(self, learned, cached=True, size=4, views=1):
        helper = fixture.FuturePredictionModeTest()
        cfg = helper.make_config(cached=cached, merge_size=size)
        cfg.visual_encoder.learned_query_merge = learned
        cfg.visual_encoder.cls_token = True
        cfg.workspace.train_dataset.image_keys = ['head_camera', 'left_camera'][:views]
        cfg.workspace.model.denoising_network_partial.state_as_dit_token = True
        return helper, cfg

    def batch(self, helper, cfg, anchors=3):
        cached = cfg.workspace.train_dataset.use_cached_image_features
        batch = helper.make_batch(cached=cached, anchors=anchors)
        suffix = '_feature' if cached else ''
        for camera in cfg.workspace.train_dataset.image_keys[1:]:
            batch[camera + suffix] = batch['head_camera' + suffix].clone() + 0.2
        return batch

    def test_cls_bypasses_local_attention_values_masks_and_gradients(self):
        merger = LearnedQueryPatchMerger(8, 32, 256, 4, 4, cls_token=True)
        images = torch.randn(2, 2 * 257, 8, requires_grad=True)
        output = merger(images).reshape(2, 2, 17, 32)
        grouped = images.reshape(2, 2, 257, 8)
        expected_cls = merger.image_projector(merger.image_norm(grouped[..., -1, :]))
        torch.testing.assert_close(output[..., -1, :], expected_cls)
        changed = images.detach().clone().reshape(2, 2, 257, 8)
        changed[:, 0, -1, 0] += 10
        changed_out = merger(changed.reshape(2, 514, 8)).reshape(2, 2, 17, 32)
        torch.testing.assert_close(changed_out[:, 0, :-1], output[:, 0, :-1], rtol=0, atol=0)
        torch.testing.assert_close(changed_out[:, 1], output[:, 1], rtol=0, atol=0)
        self.assertGreater(float((changed_out[:, 0, -1] - output[:, 0, -1]).abs().max()), 0)
        mask = torch.ones(2, 514, dtype=torch.bool)
        mask[:, 256] = False
        mask[:, 257] = False
        expected = torch.ones(2, 34, dtype=torch.bool)
        expected[:, 16] = False
        expected[:, 17] = False
        torch.testing.assert_close(merger.merge_mask(mask), expected)
        output.square().mean().backward()
        grad = images.grad.reshape(2, 2, 257, 8)
        self.assertGreater(float(grad[..., -1, :].abs().sum()), 0)
        self.assertGreater(float(grad[..., :-1, :].abs().sum()), 0)

    def test_raw_and_cached_cls_values_match_for_both_pooling_modes(self):
        for learned in (False, True):
            with self.subTest(learned=learned):
                helper, cfg = self.config(learned, cached=False, views=2)
                policy = helper.instantiate_policy(cfg)
                batch = self.batch(helper, cfg)
                raw = policy.global_cond_encoder(batch)
                obs = {}
                for camera in cfg.workspace.train_dataset.image_keys:
                    encoder = policy.global_cond_encoder.encoder_dict[camera]
                    image = batch[camera].reshape(-1, 3, 224, 224)
                    tokens = policy.shared_model_manager.forward_patches(
                        fixture.MODEL_NAME, encoder._preprocess(image))
                    full = torch.cat([tokens[:, :1], tokens[:, 5:]], dim=-2)
                    obs[camera + '_feature'] = full.reshape(1, 3, 1, 257, 8)
                cached = policy.global_cond_encoder(obs)
                torch.testing.assert_close(raw, cached, rtol=0, atol=0)
                per_view = 257 if learned else 17
                self.assertEqual(raw.shape, (1, 3, 2 * per_view, 8))
                for view, camera in enumerate(cfg.workspace.train_dataset.image_keys):
                    torch.testing.assert_close(raw[..., (view+1)*per_view-1, :], obs[camera+'_feature'][..., 0, 0, :])
                del policy

    def test_both_modes_cached_online_geometry_train_and_infer(self):
        cases = ((False, True, 4, 1, 17), (True, True, 4, 1, 17),
                 (False, False, [4, 8], 2, 9), (True, False, [8, 4], 2, 9),
                 (False, True, 8, 1, 5), (True, True, 8, 1, 5))
        for learned, cached, size, views, per_view in cases:
            with self.subTest(learned=learned, cached=cached, size=size):
                helper, cfg = self.config(learned, cached, size, views)
                tokens_target = learned and not cached
                if tokens_target:
                    cfg.workspace.model.future_prediction_mode = 'tokens'
                policy = helper.instantiate_policy(cfg)
                batch = self.batch(helper, cfg)
                n = views * per_view
                self.assertEqual(policy.denoising_network.global_cond_token_num, n)
                self.assertEqual(policy.denoising_network.history_img_features_token_num, n)
                self.assertEqual(policy.future_img_feature_encoder.token_num, views * (per_view-1) if tokens_target else views)
                positions = policy.denoising_network.global_cond_pos_embedding.reshape(views, per_view, 32)
                torch.testing.assert_close(positions[:, -1], torch.zeros_like(positions[:, -1]), rtol=0, atol=0)
                losses = policy.compute_loss(batch)
                (losses['action'] + .05 * losses['future_prediction']).backward()
                if learned:
                    self.assertGreater(float(policy.denoising_network.global_cond_projector.query_tokens.grad.abs().sum()), 0)
                policy.eval()
                suffix = '_feature' if cached else ''
                for anchor in range(2):
                    obs = {camera+suffix: batch[camera+suffix][:, anchor] for camera in cfg.workspace.train_dataset.image_keys}
                    obs.update(agent_pos=batch['agent_pos'][:, anchor], episode_idx=torch.tensor([0]))
                    with torch.inference_mode():
                        output = policy.predict_action(obs)
                    self.assertEqual(output['action'].shape, (1, 16, 14))
                self.assertTrue(all(slot.latent.shape == (n, 32) for slot in policy.online_memory_banks[0].slots))
                del policy

    def test_streaming_shadow_reprojection_and_checkpoint_roundtrip(self):
        for learned in (False, True):
            with self.subTest(learned=learned):
                helper, cfg = self.config(learned, views=2)
                cfg.workspace.model.denoising_network_partial.max_history_len = 3
                cfg.workspace.model.denoising_network_partial.shadow_horizon = 2
                cfg.workspace.model.memory_eviction.keep_recent = 0
                policy = helper.instantiate_policy(cfg)
                batch = self.batch(helper, cfg, anchors=9)
                for start in (0, 4):
                    chunk = {name: value[:, start:start+5] for name, value in batch.items()}
                    chunk.update(_streaming_supervised_length=4, _streaming_reset=start == 0, _streaming_final=False)
                    losses = policy.compute_loss(chunk)
                    losses['action'].backward()
                    bank = policy._streaming_training_state[0]
                    self.assertTrue(any(slot.state == 'shadow' for slot in bank.slots))
                    self.assertTrue(all(slot.latent.shape == (34, 32) for slot in bank.slots))
                    self.assertTrue(all(slot.raw_features.shape == (2*(257 if learned else 17), 8) for slot in bank.slots))
                    policy.zero_grad(set_to_none=True)
                state = copy.deepcopy(policy.state_dict())
                del policy
                restored = helper.instantiate_policy(cfg)
                restored.load_state_dict(state, strict=True)
                self.assertEqual(restored.denoising_network.global_cond_token_num, 34)
                del restored


if __name__ == '__main__':
    unittest.main()
