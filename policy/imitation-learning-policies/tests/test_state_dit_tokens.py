import gc
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf, open_dict
import torch

from imitation_learning.models.denoising_networks.memory_transformer import ObservationDrivenMemoryRetriever
from imitation_learning.models.encoders.image_encoders import SharedModelManager
from imitation_learning.models.common.slot_gate import ObservationConditionedSlotGate
import test_future_prediction_mode as fixture


class StateDiTTokensTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FuturePredictionModeTest.setUpClass()

    def setUp(self):
        self.helper = fixture.FuturePredictionModeTest()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def config(self, mode='pooled', merge=False):
        cfg = self.helper.make_config(mode, cached=True, merge_size=8)
        cfg.visual_encoder.learned_query_merge = merge
        network = cfg.workspace.model.denoising_network_partial
        network.state_as_dit_token = True
        network.max_history_len = 4
        network.shadow_horizon = 4
        network.shadow_gate_scale = 0.5
        cfg.workspace.model.memory_eviction.keep_recent = 2
        return cfg

    def test_default_is_new_route_and_missing_checkpoint_setting_is_legacy(self):
        defaults = OmegaConf.load(fixture.CONFIG_DIR/'workspace/policy/denoising_network/memory_transformer.yaml')
        self.assertTrue(defaults.state_as_dit_token)
        cfg = self.config()
        cfg.workspace.model.denoising_network_partial.state_as_dit_token = False
        policy = self.helper.instantiate_policy(cfg)
        state = policy.state_dict()
        self.assertTrue(any('state_condition_attn' in name for name in state))
        del policy
        with open_dict(cfg.workspace.model.denoising_network_partial):
            del cfg.workspace.model.denoising_network_partial['state_as_dit_token']
        policy = self.helper.instantiate_policy(cfg)
        policy.load_state_dict(state, strict=True)
        self.assertFalse(policy.denoising_network.state_as_dit_token)
        self.assertEqual(policy.denoising_network.blocks[0].input_token_num, 17)

    def test_state_stays_in_dit_and_only_action_tokens_are_decoded(self):
        policy = self.helper.instantiate_policy(self.config()).eval()
        network = policy.denoising_network
        data = {
            'noisy_action':torch.randn(2, 16, 14), 'step':torch.zeros(2, dtype=torch.long),
            'global_cond':torch.randn(2, 4, 8), 'local_cond':torch.randn(2, 1, 8),
            'local_cond_mask':torch.tensor([[True], [False]]),
        }
        inputs = network._project_current_inputs(data)
        self.assertEqual(tuple(inputs['x'].shape), (2, 18, 32))
        torch.testing.assert_close(inputs['x'][:, 1:2], inputs['local_cond'])
        torch.testing.assert_close(inputs['x'][1, 1:2], network.mask_token[0] + network.input_pos_embedding[0, 1:2])
        block_shapes = []
        handles = [network.blocks[0].register_forward_pre_hook(
            lambda module, args, kwargs: block_shapes.append(tuple(kwargs['x'].shape)), with_kwargs=True,
        )]
        with patch.object(network.action_final_layer, 'forward', wraps=network.action_final_layer.forward) as decoder:
            output = network(data)
        for handle in handles:
            handle.remove()
        self.assertEqual(block_shapes, [(2, 18, 32)])
        self.assertEqual(tuple(decoder.call_args.args[0].shape), (2, 16, 32))
        self.assertEqual(tuple(output['action'].shape), (2, 16, 14))
        self.assertFalse(any('state_condition_' in name for name, _ in network.named_parameters()))

    def test_images_query_history_and_gates_use_state_in_single_and_multiview(self):
        for tokens in (4, 12):
            with self.subTest(tokens=tokens):
                retriever = ObservationDrivenMemoryRetriever(
                    hidden_dim=16, head_num=4, decision_memory_token_num=4,
                    history_attention_type='token_wise', ssmax_scaling_param=None,
                    binary_gating=False, straight_through='', retrieval_mode='perception_patches',
                    use_conditioned_slot_gate=True, state_as_dit_token=True,
                ).eval()
                images = torch.randn(2, tokens, 16)
                history = torch.randn(2, 3, tokens, 16)
                gate_inputs, queries = [], []
                handle = retriever.slot_gate.register_forward_pre_hook(
                    lambda module, args, kwargs: gate_inputs.append(kwargs['current_tokens'].clone()), with_kwargs=True,
                )
                query_handle = retriever.perception_retrieval_blocks[0].register_forward_pre_hook(
                    lambda module, args, kwargs: queries.append(kwargs['query'].clone()), with_kwargs=True,
                )
                first = retriever(global_cond=images, local_cond=torch.randn(2, 1, 16), history_latents=history)
                gates = retriever.latest_slot_gates.clone()
                second = retriever(global_cond=images, local_cond=10 * torch.randn(2, 1, 16), history_latents=history)
                handle.remove()
                query_handle.remove()
                self.assertGreater(float((gates - retriever.latest_slot_gates).abs().max()), 1e-6)
                self.assertGreater(float((first - second).abs().max()), 0)
                self.assertEqual(retriever.slot_gate.mlp[0].in_features, 3 * 16)
                for value in gate_inputs:
                    torch.testing.assert_close(value, images, rtol=0, atol=0)
                for value in queries:
                    torch.testing.assert_close(value, images, rtol=0, atol=0)
                torch.testing.assert_close(retriever(global_cond=images, local_cond=torch.randn(2, 1, 16)), images)

    def test_state_gate_masking_and_state_gradients(self):
        gate = ObservationConditionedSlotGate(16, 4, use_state_condition=True)
        history, images = torch.randn(2, 3, 4, 16), torch.randn(2, 4, 16)
        states = torch.randn(2, 2, 16, requires_grad=True)
        mask = torch.tensor([[True, False], [False, False]])
        original = gate(history, images, state_tokens=states, state_mask=mask)
        changed = states.detach().masked_fill(~mask[:, :, None], float('nan'))
        masked = gate(history, images, state_tokens=changed, state_mask=mask)
        torch.testing.assert_close(original, masked, rtol=0, atol=0)
        original.sum().backward()
        self.assertGreater(float(states.grad[0, 0].abs().sum()), 0)
        self.assertEqual(float(states.grad[~mask].abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, 'state_mask'):
            gate(history, images, state_tokens=states.detach(), state_mask=torch.ones(2, 3))

    def test_chunked_training_and_online_share_new_route_and_shadow_clock(self):
        for mode in ('pooled', 'tokens'):
            for merge in (False, True):
                with self.subTest(mode=mode, merge=merge):
                    policy = self.helper.instantiate_policy(self.config(mode, merge))
                    batch = self.helper.make_batch(anchors=13)
                    batch['agent_pos'].requires_grad_()
                    for start in (0, 4, 8):
                        chunk = {name:value[:, start:start + 5] for name,value in batch.items()}
                        chunk.update(_streaming_supervised_length=4, _streaming_reset=start == 0, _streaming_final=False)
                        losses = policy.compute_loss(chunk)
                        action_gradient = torch.autograd.grad(losses['action'], batch['agent_pos'], retain_graph=True)[0]
                        self.assertGreater(float(action_gradient.abs().sum()), 0)
                        future_gradient = torch.autograd.grad(losses['future_prediction'], batch['agent_pos'], retain_graph=True, allow_unused=True)[0]
                        self.assertIsNotNone(future_gradient)
                        self.assertGreater(float(future_gradient.abs().sum()), 0)
                        total = losses['action'] + 0.05 * losses['future_prediction'] + 0.01 * losses['gate_selectivity']
                        total.backward()
                        self.assertGreater(float(policy.denoising_network.local_cond_projector.model[0].weight.grad.abs().sum()), 0)
                        self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None))
                        policy.zero_grad(set_to_none=True)
                    def snapshot(bank):
                        return [(s.slot_id, s.state, s.expire_at) for s in bank.slots]
                    expected = snapshot(policy._streaming_training_state[0])
                    policy.eval()
                    with patch.object(policy.future_feature_predictor, 'forward', side_effect=AssertionError('online future predictor')):
                        for anchor in range(12):
                            with torch.inference_mode():
                                result = policy.predict_action({
                                    'head_camera_feature':batch['head_camera_feature'][:, anchor],
                                    'agent_pos':batch['agent_pos'][:, anchor], 'episode_idx':torch.tensor([0]),
                                })
                            self.assertEqual(tuple(result['action'].shape), (1, 16, 14))
                            self.assertTrue(torch.isfinite(result['action']).all())
                    self.assertEqual(snapshot(policy.online_memory_banks[0]), expected)
                    self.assertEqual(policy.denoising_network.blocks[0].input_token_num, 18)
                    del policy

    def test_new_architecture_roundtrip_loads_strictly(self):
        cfg = self.config()
        policy = self.helper.instantiate_policy(cfg)
        state = policy.state_dict()
        del policy
        restored = self.helper.instantiate_policy(cfg)
        restored.load_state_dict(state, strict=True)
        self.assertTrue(restored.denoising_network.state_as_dit_token)
        self.assertFalse(any('state_condition_' in name for name in state))


if __name__ == '__main__':
    unittest.main()
