"""Global MAP token counts, view isolation and complete policy integration."""
import copy
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from transformers import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipMultiheadAttentionPoolingHead

from imitation_learning.models.common.learned_query_merger import LearnedQueryPatchMerger
from imitation_learning.models.common.spatial_merge import spatial_merge_shape
import test_learned_query_merge as regional


class GlobalQueryMergerTest(unittest.TestCase):
    def test_each_global_query_matches_siglip_map_on_its_whole_view(self):
        merger = LearnedQueryPatchMerger(8, 32, 256, [4, 8], 4, num_queries=3)
        reference = SiglipMultiheadAttentionPoolingHead(SiglipVisionConfig(
            hidden_size=32, intermediate_size=128, num_attention_heads=4,
            hidden_act='gelu_pytorch_tanh', layer_norm_eps=1e-6,
        ))
        reference.attention.load_state_dict(merger.attention.state_dict())
        reference.layernorm.load_state_dict(merger.layernorm.state_dict())
        reference.mlp.fc1.load_state_dict(merger.mlp[0].state_dict())
        reference.mlp.fc2.load_state_dict(merger.mlp[2].state_dict())
        images = torch.randn(2, 512, 8)
        with torch.no_grad():
            views = merger._group(merger.image_projector(merger.image_norm(images)))
            expected = []
            for view in range(2):
                for query in merger.query_tokens:
                    reference.probe.copy_(query[None, None])
                    expected.append(reference(views[:, view]))
            expected = torch.stack(expected, dim=1)
            actual = merger._pool_images(images)
        self.assertEqual(actual.shape, (2, 6, 32))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_all_queries_read_every_patch_without_crossing_views(self):
        merger = LearnedQueryPatchMerger(8, 32, 256, 8, 4, num_queries=8)
        images = torch.randn(2, 3, 512, 8)
        with patch.object(merger.attention, 'forward', wraps=merger.attention.forward) as attention:
            before = merger._pool_images(images).detach()
        query, keys, values = attention.call_args.args
        self.assertEqual(query.shape, (12, 8, 32))
        self.assertEqual(keys.shape, (12, 256, 32))
        self.assertIs(keys, values)
        changed = images.clone()
        # The final patch belongs to a different region than query 0 in the
        # old implementation. Global pooling must expose it to all 8 queries.
        changed[..., -1, 0] += 10
        after = merger._pool_images(changed).detach()
        difference = (after - before).abs().amax(dim=-1)
        self.assertTrue(torch.equal(difference[..., :8], torch.zeros_like(difference[..., :8])))
        self.assertTrue((difference[..., 8:] > 0).all())

    def test_global_count_is_independent_of_spatial_merge_size(self):
        first = LearnedQueryPatchMerger(8, 32, 256, [4, 8], 4, num_queries=16)
        second = LearnedQueryPatchMerger(8, 32, 256, 8, 4, num_queries=16)
        second.load_state_dict(first.state_dict(), strict=True)
        images = torch.randn(2, 256, 8)
        torch.testing.assert_close(first._pool_images(images), second._pool_images(images), rtol=0, atol=0)
        self.assertEqual(first._pool_images(images).shape, (2, 16, 32))
        # Global queries have no dependency on a divisible spatial grid.
        non_square = LearnedQueryPatchMerger(8, 32, 15, [3, 4], 4, num_queries=3)
        self.assertEqual(non_square._pool_images(torch.randn(2, 30, 8)).shape, (2, 6, 32))

    def test_masks_follow_view_validity_for_all_global_queries(self):
        merger = LearnedQueryPatchMerger(8, 32, 256, 8, 4, num_queries=3)
        mask = torch.ones(2, 4, 512, dtype=torch.bool)
        mask[0, 1, 256:] = False
        mask[1, 2, :256] = False
        expected = torch.ones(2, 4, 6, dtype=torch.bool)
        expected[0, 1, 3:] = False
        expected[1, 2, :3] = False
        torch.testing.assert_close(merger.merge_mask(mask), expected)

    def test_global_queries_have_gradients_and_no_raw_query_residual(self):
        merger = LearnedQueryPatchMerger(8, 32, 256, [4, 8], 4, num_queries=8)
        images = torch.randn(2, 3, 256, 8).detach()
        merger._pool_images(images).square().mean().backward()
        self.assertIsNone(images.grad)
        self.assertFalse(hasattr(merger, 'state_modulation'))
        for parameter in (merger.query_tokens, merger.image_projector.weight, merger.mlp[0].weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        with torch.no_grad():
            merger.attention.out_proj.weight.zero_()
            merger.attention.out_proj.bias.zero_()
            merger.layernorm.bias.zero_()
            merger.mlp[0].bias.zero_()
            merger.mlp[2].bias.zero_()
        torch.testing.assert_close(merger._pool_images(images), torch.zeros(2, 3, 8, 32), rtol=0, atol=0)
        for invalid in (0, -1, True, 8.0, '8'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'positive integer'):
                LearnedQueryPatchMerger(8, 32, 256, 8, 4, num_queries=invalid)


class GlobalQueryMergePolicyTest(regional.LearnedQueryMergePolicyTest):
    """Reuse cached/online, multi-view, Shadow and checkpoint lifecycle cases."""

    def config(self, mode='pooled', cached=True, views=1, size=None):
        cfg = super().config(mode, cached, views, size)
        height, width = spatial_merge_shape(cfg.visual_encoder.spatial_merge_size)
        cfg.visual_encoder.learned_query_merge = 256 // (height * width)
        return cfg

    def test_query_count_and_future_target_count_can_differ(self):
        # Explicit counts do not depend on region dimensions. A 3-query
        # current/history representation can still predict 16 future patches.
        cfg = self.config(mode='tokens', cached=True, views=2, size=4)
        cfg.visual_encoder.learned_query_merge = 3
        policy = self.helper.instantiate_policy(cfg)
        network = policy.denoising_network
        self.assertEqual(network.global_cond_token_num, 6)
        self.assertEqual(network.history_img_features_token_num, 6)
        self.assertEqual(policy.future_img_feature_encoder.token_num, 32)
        batch = self.batch(cfg)
        losses = policy.compute_loss(batch)
        (losses['action'] + 0.05 * losses['future_prediction']).backward()
        self.assertTrue(torch.isfinite(network.global_cond_projector.query_tokens.grad).all())
        self.assertGreater(float(network.global_cond_projector.query_tokens.grad.abs().sum()), 0)

    def test_zero_matches_false_and_one_is_distinct_from_legacy_true(self):
        cfg = self.helper.make_config(merge_size=[4, 8])
        cfg.visual_encoder.learned_query_merge = False
        policy = self.helper.instantiate_policy(cfg)
        weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        expected = {k: v.detach().clone() for k, v in policy.compute_loss(self.helper.make_batch()).items()}
        del policy
        cfg.visual_encoder.learned_query_merge = 0
        restored = self.helper.instantiate_policy(cfg)
        restored.load_state_dict(weights, strict=True)
        actual = restored.compute_loss(self.helper.make_batch())
        self.assertEqual(restored.denoising_network.global_cond_token_num, 8)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        del restored
        for setting, count, query_num in ((True, 8, None), (1, 1, 1)):
            cfg.visual_encoder.learned_query_merge = setting
            policy = self.helper.instantiate_policy(cfg)
            self.assertEqual(policy.denoising_network.global_cond_token_num, count)
            self.assertEqual(policy.denoising_network.global_cond_projector.num_queries, query_num)
            del policy

    def test_saved_numeric_config_keeps_global_mode_and_query_count(self):
        cfg = self.config(views=2)
        cfg.visual_encoder.learned_query_merge = 16
        restored_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=False))
        self.assertIs(type(restored_cfg.visual_encoder.learned_query_merge), int)
        policy = self.helper.instantiate_policy(restored_cfg)
        merger = policy.denoising_network.global_cond_projector
        self.assertEqual(merger.query_tokens.shape, (16, 32))
        self.assertEqual(merger.num_queries, 16)
        self.assertEqual(policy.denoising_network.global_cond_token_num, 32)
        changed = copy.deepcopy(cfg)
        changed.visual_encoder.spatial_merge_size = 8
        del policy
        policy = self.helper.instantiate_policy(changed)
        self.assertEqual(policy.denoising_network.global_cond_token_num, 32)
        self.assertEqual(policy.denoising_network.history_img_features_token_num, 32)

    def test_invalid_numeric_settings_are_rejected(self):
        for setting in (-1, 8.0, '8', [8], None):
            cfg = self.config()
            cfg.visual_encoder.learned_query_merge = setting
            with self.subTest(setting=setting), self.assertRaisesRegex(Exception, 'non-negative integer'):
                self.helper.instantiate_policy(cfg)


if __name__ == '__main__':
    unittest.main()
