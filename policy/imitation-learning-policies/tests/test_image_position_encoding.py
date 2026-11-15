import gc
import unittest

import torch
from omegaconf import open_dict

from imitation_learning.models.common.pos_embeddings import get_nd_sincos_pos_embed_from_grid
from imitation_learning.models.denoising_networks.conditional_transformer import ConditionalTransformer
from imitation_learning.models.encoders.image_encoders import SharedModelManager
import test_future_prediction_mode as fixture


class ImagePositionEncodingTest(unittest.TestCase):
    def network(self, count, grid=None):
        return ConditionalTransformer(
            head_num=4, layer_num=1, hidden_dim=32, projector_type='mlp3x_gelu',
            action_dim=14, action_token_num=16, global_cond_dim=8,
            global_cond_token_num=count, local_cond_dim=8,
            local_cond_token_num=1, seed=0, global_cond_grid_shape=grid,
        )

    def test_rectangular_coordinates_preserve_rows_and_columns(self):
        positions = self.network(8, (2, 4)).global_cond_pos_embedding[0]
        # A->C stays in the same row; A->E stays in the same column.
        torch.testing.assert_close(positions[0, :16], positions[2, :16], rtol=0, atol=0)
        self.assertGreater(float((positions[0, 16:] - positions[2, 16:]).abs().sum()), 0)
        torch.testing.assert_close(positions[0, 16:], positions[4, 16:], rtol=0, atol=0)
        self.assertGreater(float((positions[0, :16] - positions[4, :16]).abs().sum()), 0)

    def test_geometry_handles_orientation_views_and_square_token_totals(self):
        for views, height, width in ((1, 2, 4), (1, 4, 2), (1, 2, 8),
                                      (2, 2, 4), (3, 4, 2), (4, 2, 2),
                                      (1, 1, 16), (1, 16, 1), (3, 1, 1)):
            with self.subTest(views=views, height=height, width=width):
                actual = self.network(views * height * width, (height, width))
                expected = torch.from_numpy(get_nd_sincos_pos_embed_from_grid(
                    32, (views, height, width),
                )).float().reshape_as(actual.global_cond_pos_embedding)
                torch.testing.assert_close(actual.global_cond_pos_embedding, expected, rtol=0, atol=0)

    def test_single_spatial_token_has_zero_position(self):
        positions = self.network(1, (1, 1)).global_cond_pos_embedding
        torch.testing.assert_close(positions, torch.zeros_like(positions), rtol=0, atol=0)

    def test_invalid_geometry_is_rejected(self):
        for count, grid in ((8, (0, 4)), (8, (True, 4)), (8, (2,)), (9, (2, 4))):
            with self.subTest(count=count, grid=grid), self.assertRaises(ValueError):
                self.network(count, grid)

    def test_matching_square_routes_keep_weights_outputs_and_gradients(self):
        for views, side in ((1, 2), (2, 2), (3, 2), (1, 4), (2, 4), (3, 4)):
            with self.subTest(views=views, side=side):
                legacy = self.network(views * side * side)
                explicit = self.network(views * side * side, (side, side))
                for key, value in legacy.state_dict().items():
                    torch.testing.assert_close(explicit.state_dict()[key], value, rtol=0, atol=0)
                batch = dict(noisy_action=torch.randn(2, 16, 14),
                             step=torch.tensor([1, 2]),
                             global_cond=torch.randn(2, views * side * side, 8),
                             local_cond=torch.randn(2, 1, 8))
                first = legacy(dict(batch))['action']
                second = explicit(dict(batch))['action']
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                first.square().sum().backward()
                second.square().sum().backward()
                for left, right in zip(legacy.parameters(), explicit.parameters()):
                    if left.grad is None:
                        self.assertIsNone(right.grad)
                    else:
                        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


class ImagePositionPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FuturePredictionModeTest.setUpClass()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def test_pooling_and_query_use_actual_grid_with_cached_and_online_features(self):
        helper = fixture.FuturePredictionModeTest()
        for learned, cached, merge, views in (
            (False, True, [8, 4], 1), (True, True, [8, 4], 1),
            (False, False, [4, 8], 2), (True, False, [4, 8], 2),
            (False, True, [8, 2], 1), (True, True, [8, 2], 1),
        ):
            with self.subTest(learned=learned, cached=cached, merge=merge, views=views):
                cfg = helper.make_config(cached=cached, merge_size=merge)
                cfg.visual_encoder.state_query_merge = learned
                cfg.workspace.train_dataset.image_keys = ['head_camera', 'left_camera'][:views]
                self.assertNotIn('cls_token', cfg.visual_encoder)
                for key in ('global_cond_encoder', 'history_img_feature_encoder', 'denoising_network_partial'):
                    section = cfg.workspace.model[key]
                    if 'image_encoder_partial' in section:
                        section = section.image_encoder_partial
                    self.assertNotIn('cls_token', section)
                policy = helper.instantiate_policy(cfg)
                height, width = 16 // merge[0], 16 // merge[1]
                network = policy.denoising_network
                self.assertEqual(tuple(network.global_cond_grid_shape), (height, width))
                expected = torch.from_numpy(get_nd_sincos_pos_embed_from_grid(
                    32, (views, height, width),
                )).float().reshape(1, views * height * width, 32)
                torch.testing.assert_close(network.global_cond_pos_embedding, expected, rtol=0, atol=0)
                self.assertEqual(network.history_img_features_token_num, views * height * width)
                self.assertFalse(hasattr(policy.global_cond_encoder.encoder_dict['head_camera'], 'cls_token'))
                del policy

    def test_legacy_pooling_config_uses_encoder_geometry(self):
        helper = fixture.FuturePredictionModeTest()
        cfg = helper.make_config(merge_size=[8, 4])
        # Older checkpoints had no merger geometry in the network config.
        with open_dict(cfg.workspace.model.denoising_network_partial):
            del cfg.workspace.model.denoising_network_partial.spatial_merge_size
        policy = helper.instantiate_policy(cfg)
        self.assertEqual(tuple(policy.denoising_network.global_cond_grid_shape), (2, 4))
        self.assertEqual(policy.denoising_network.global_cond_token_num, 8)


if __name__ == '__main__':
    unittest.main()
