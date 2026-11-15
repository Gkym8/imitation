import gc
import unittest

import torch

from imitation_learning.models.common.image_state_fusion import ImageStateFusion
from imitation_learning.models.encoders.image_encoders import SharedModelManager
import test_future_prediction_mode as fixture


class PooledStateFusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FuturePredictionModeTest.setUpClass()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def test_single_state_read_is_the_historical_residual_formula(self):
        fusion = ImageStateFusion(32)
        images = torch.randn(2, 3, 8, 32, requires_grad=True)
        states = torch.randn(2, 3, 1, 32, requires_grad=True)
        actual = fusion(images, states)
        expected = fusion.state_norm(images + fusion.state_value(states))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual[..., 0].sum().backward()
        self.assertGreater(float(images.grad.abs().sum()), 0)
        self.assertGreater(float(states.grad.abs().sum()), 0)

    def test_masked_states_never_enter_fusion_or_gradients(self):
        fusion = ImageStateFusion(32)
        images = torch.randn(2, 8, 32)
        states = torch.randn(2, 2, 32, requires_grad=True)
        mask = torch.tensor([[True, False], [False, False]])
        actual = fusion(images, states, mask)
        changed = fusion(images, states.detach().masked_fill(~mask[..., None], float('nan')), mask)
        torch.testing.assert_close(actual, changed, rtol=0, atol=0)
        torch.testing.assert_close(actual[1], fusion.state_norm(images[1]), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(actual).all())
        actual[..., 0].sum().backward()
        self.assertEqual(float(states.grad[~mask].abs().sum()), 0)

    def test_pooling_fuses_current_and_historical_own_states_before_retrieval(self):
        helper = fixture.FuturePredictionModeTest()
        cfg = helper.make_config(merge_size=[8, 4])
        cfg.workspace.model.denoising_network_partial.state_as_dit_token = True
        policy = helper.instantiate_policy(cfg)
        net = policy.denoising_network
        self.assertFalse(net.state_query_merge)
        self.assertIsNotNone(net.image_state_fusion)
        self.assertTrue(net.memory_retriever.state_already_fused)
        images, states = torch.randn(2, 8, 8), torch.randn(2, 1, 8)
        for projector, call in ((net.global_cond_projector, net._project_global_cond),
                                (net.history_img_features_projector, net.project_history_image_features)):
            actual = call(images, states)
            expected = net.image_state_fusion(projector(images), net.local_cond_projector(states))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertGreater(float((actual - call(images, -states)).abs().max()), 0)
        with self.assertRaisesRegex(ValueError, 'own state'):
            net.project_history_image_features(images)
        batch = helper.make_batch(anchors=3)
        policy.eval()
        for anchor in range(2):
            with torch.inference_mode():
                policy.predict_action(dict(head_camera_feature=batch['head_camera_feature'][:, anchor],
                                           agent_pos=batch['agent_pos'][:, anchor], episode_idx=torch.tensor([0])))
            if anchor == 0:
                first = policy.online_memory_banks[0].slots[0].latent.clone()
        torch.testing.assert_close(policy.online_memory_banks[0].slots[0].latent, first, rtol=0, atol=0)
        self.assertTrue(all(slot.raw_features is None for slot in policy.online_memory_banks[0].slots))
        self.assertEqual(net.blocks[0].input_token_num, 18)

    def test_streaming_pooling_saves_own_states_and_reencodes_after_updates(self):
        helper = fixture.FuturePredictionModeTest()
        cfg = helper.make_config(merge_size=[8, 4])
        netcfg = cfg.workspace.model.denoising_network_partial
        netcfg.max_history_len, netcfg.shadow_horizon = 3, 2
        cfg.workspace.model.memory_eviction.keep_recent = 0
        policy = helper.instantiate_policy(cfg)
        batch = helper.make_batch(anchors=9)
        for start in (0, 4):
            chunk = {name: value[:, start:start+5] for name, value in batch.items()}
            chunk.update(_streaming_supervised_length=4, _streaming_reset=start == 0, _streaming_final=False)
            losses = policy.compute_loss(chunk)
            (losses['action'] + 0.05 * losses['future_prediction']).backward()
            bank = policy._streaming_training_state[0]
            self.assertTrue(any(slot.state == 'shadow' for slot in bank.slots))
            for slot in bank.slots:
                torch.testing.assert_close(slot.raw_state['agent_pos'], batch['agent_pos'][0, slot.frame_index//10], rtol=0, atol=0)
                self.assertEqual(tuple(slot.raw_features.shape), (8, 8))
                self.assertFalse(slot.raw_state['agent_pos'].requires_grad)
            policy.zero_grad(set_to_none=True)
            if start == 0:
                before = bank.slots[0].latent.clone()
                encoder = policy.local_cond_encoder.encoder_dict['agent_pos'].model.model
                with torch.no_grad():
                    encoder.weight[0, 0].add_(0.2)
                bank.reproject(policy._reproject_state_conditioned_history)
                self.assertGreater(float((bank.slots[0].latent - before).abs().max()), 0)
                slot = bank.slots[0]
                expected = policy._reproject_state_conditioned_history(slot.raw_features[None],
                    {name: value[None] for name, value in slot.raw_state.items()})[0]
                # Single-frame and batched GEMMs can round differently.
                torch.testing.assert_close(slot.latent, expected)
                sum(slot.latent[:, 0].sum() for slot in bank.slots).backward()
                self.assertGreater(float(encoder.weight.grad.abs().sum()), 0)
                self.assertGreater(float(policy.denoising_network.image_state_fusion.state_value.weight.grad.abs().sum()), 0)
                bank.detach()
                policy.zero_grad(set_to_none=True)


if __name__ == '__main__':
    unittest.main()
