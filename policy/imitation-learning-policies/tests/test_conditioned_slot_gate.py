import gc
import unittest
from unittest.mock import patch

import hydra
import torch
from torch import nn

from imitation_learning.models.common.slot_gate import ObservationConditionedSlotGate
from imitation_learning.models.denoising_networks.memory_transformer import (
    ObservationDrivenMemoryRetriever,
)
from imitation_learning.models.denoising_networks.modules import HistoryCrossAttention
from imitation_learning.models.encoders.image_encoders import SharedModelManager
import test_future_prediction_mode as future_fixture


def make_retriever():
    return ObservationDrivenMemoryRetriever(
        hidden_dim=16,
        head_num=4,
        decision_memory_token_num=4,
        history_attention_type="token_wise",
        ssmax_scaling_param=None,
        binary_gating=False,
        straight_through="",
        retrieval_mode="perception_patches",
        perception_retrieval_layers=2,
        use_conditioned_slot_gate=True,
    )


class ObservationConditionedSlotGateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.gate = ObservationConditionedSlotGate(16, 4, hidden_dim=32)
        self.history = torch.randn(2, 3, 4, 16)
        self.current = torch.randn(2, 4, 16)

    def test_current_and_history_both_affect_scores(self):
        original = self.gate(self.history, self.current)
        changed_current = self.gate(self.history, self.current.flip(-1))
        changed_history = self.gate(self.history.flip(-1), self.current)
        self.assertEqual(original.shape, (2, 3, 4))
        self.assertGreater(float((original - changed_current).abs().max()), 1e-5)
        self.assertGreater(float((original - changed_history).abs().max()), 1e-5)
        self.assertTrue(((original.exp() > 0) & (original.exp() < 1)).all())

    def test_slot_permutation_only_permutes_gate_outputs(self):
        order = torch.tensor([2, 0, 1])
        original = self.gate(self.history, self.current)
        permuted = self.gate(self.history[:, order], self.current)
        torch.testing.assert_close(permuted, original[:, order])

    def test_masked_slots_and_observation_tokens_cannot_affect_scores(self):
        history_mask = torch.tensor([[True, True, False], [False, False, False]])
        current_mask = torch.tensor([[True, True, False, False], [False] * 4])
        original = self.gate(self.history, self.current, history_mask, current_mask)
        history = self.history.masked_fill(~history_mask[:, :, None, None], float("nan"))
        current = self.current.masked_fill(~current_mask[:, :, None], float("nan"))
        masked = self.gate(history, current, history_mask, current_mask)
        torch.testing.assert_close(masked, original)
        self.assertTrue(torch.isfinite(masked).all())

    def test_gradients_reach_current_and_history_and_ignore_padding(self):
        history = self.history.requires_grad_()
        current = self.current.requires_grad_()
        mask = torch.tensor([[True, True, False], [False, False, False]])
        self.gate(history, current, history_mask=mask).sum().backward()
        self.assertGreater(float(history.grad[0, :2].abs().sum()), 0)
        self.assertEqual(float(history.grad[0, 2].abs().sum()), 0)
        self.assertEqual(float(history.grad[1].abs().sum()), 0)
        self.assertGreater(float(current.grad[0].abs().sum()), 0)
        self.assertEqual(float(current.grad[1].abs().sum()), 0)
        for parameter in self.gate.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_log_sigmoid_keeps_tiny_gate_gradients(self):
        with torch.no_grad():
            self.gate.mlp[-1].bias.fill_(-1000)
        output = self.gate(self.history, self.current)
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertGreater(float(self.gate.mlp[-1].bias.grad.abs().min()), 0)

    def test_slot_count_can_change_without_changing_parameters(self):
        for slots in (0, 1, 8, 16):
            with self.subTest(slots=slots):
                output = self.gate(torch.randn(2, slots, 4, 16), self.current)
                self.assertEqual(output.shape, (2, slots, 4))

    def test_invalid_masks_and_dimensions_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "history_mask"):
            self.gate(self.history, self.current, torch.ones(2, 2))
        with self.assertRaisesRegex(ValueError, "current_tokens"):
            self.gate(self.history, torch.randn(1, 4, 16))
        with self.assertRaisesRegex(ValueError, "positive"):
            ObservationConditionedSlotGate(16, 0)


class HeadwiseSlotAttentionTest(unittest.TestCase):
    def make_attention(self, kind):
        attention = HistoryCrossAttention(
            dim=8, head_num=2, qkv_bias=True, qk_norm=False,
            norm_layer=nn.LayerNorm, attention_type=kind,
            ssmax_scaling_param=None,
        )
        with torch.no_grad():
            attention.q.weight.zero_()
            attention.q.bias.zero_()
            attention.kv.weight.zero_()
            attention.kv.bias.zero_()
        return attention

    def test_each_head_uses_its_gate_and_broadcasts_over_patches(self):
        for kind in ("token_wise", "block_wise"):
            with self.subTest(kind=kind):
                attention = self.make_attention(kind)
                gates = torch.tensor([[[0.9, 0.2], [0.1, 0.8]]])
                records = {"history_cross_attention": [], "history_cross_attention_slot_weights": []}
                attention(
                    torch.randn(1, 2, 8), torch.randn(1, 2, 2, 8),
                    history_log_gate=gates.log(), record_data_dict=records,
                )
                expected_slots = gates.permute(0, 2, 1)
                slot_weights = records["history_cross_attention_slot_weights"][0]
                expected = expected_slots[:, :, None, :].expand_as(slot_weights)
                torch.testing.assert_close(slot_weights, expected)
                if kind == "token_wise":
                    patches = records["history_cross_attention"][0].reshape(1, 2, 2, 2, 2)
                    torch.testing.assert_close(
                        patches, (expected / 2).unsqueeze(-1).expand_as(patches)
                    )

    def test_masked_rows_have_zero_attention_and_finite_gate_backward(self):
        attention = self.make_attention("token_wise")
        log_gate = torch.randn(2, 3, 2, requires_grad=True)
        records = {"history_cross_attention_slot_weights": []}
        output = attention(
            torch.randn(2, 2, 8), torch.randn(2, 3, 2, 8),
            history_mask=torch.tensor([[True, True, False], [False] * 3]),
            history_log_gate=log_gate, record_data_dict=records,
        )
        self.assertTrue(torch.isfinite(output).all())
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(log_gate.grad).all())
        self.assertEqual(float(log_gate.grad[1].abs().sum()), 0)
        weights = records["history_cross_attention_slot_weights"][0]
        self.assertEqual(float(weights[1].abs().sum()), 0)
        self.assertEqual(float(weights[0, :, :, 2].abs().sum()), 0)


class ConditionedRetrieverTest(unittest.TestCase):
    def test_decision_memory_fuses_both_inputs_and_empty_history_keeps_current(self):
        retriever = make_retriever().eval()
        with torch.no_grad():
            retriever.perception_gate.proj.weight.zero_()
            retriever.perception_gate.proj.bias.copy_(torch.linspace(-2, 2, 16))
        image = torch.randn(2, 4, 16)
        state = torch.randn(2, 1, 16)
        current = retriever.condition_current_tokens(image, state)
        layer_outputs = []
        handle = retriever.perception_retrieval_blocks[-1].register_forward_hook(
            lambda module, args, output: layer_outputs.append(output),
        )
        records = {"decision_memory": []}
        decision = retriever(
            global_cond=image, local_cond=state,
            history_latents=torch.randn(2, 3, 4, 16),
            history_mask=torch.tensor([[True, True, False], [False, False, False]]),
            record_data_dict=records,
        )
        handle.remove()
        self.assertEqual(tuple(decision.shape), (2, 4, 16))
        gate = retriever.perception_gate.proj.bias.sigmoid()
        expected = gate * current[0] + (1 - gate) * layer_outputs[0][0]
        torch.testing.assert_close(decision[0], expected)
        self.assertGreater(float((decision[0] - current[0]).abs().max()), 1e-4)
        self.assertGreater(float((decision[0] - layer_outputs[0][0]).abs().max()), 1e-4)
        torch.testing.assert_close(decision[1], current[1])
        torch.testing.assert_close(records["decision_memory"][0], decision)
        self.assertIsNone(records["decision_memory"][0].grad_fn)
        without_history = retriever(global_cond=image, local_cond=state)
        torch.testing.assert_close(without_history, current)

    def test_fusion_weights_are_restored_and_required_during_strict_nested_load(self):
        original = nn.Module()
        original.memory_retriever = make_retriever()
        state = original.state_dict()
        self.assertIn("memory_retriever.perception_gate.proj.weight", state)
        self.assertIn("memory_retriever.perception_gate.proj.bias", state)
        state["memory_retriever.perception_gate.proj.weight"] = torch.randn(16, 32)
        state["memory_retriever.perception_gate.proj.bias"] = torch.randn(16)
        restored = nn.Module()
        restored.memory_retriever = make_retriever()
        result = restored.load_state_dict(state, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        for key, value in state.items():
            torch.testing.assert_close(restored.state_dict()[key], value)
        without_fusion = {key: value for key, value in state.items() if "perception_gate" not in key}
        with self.assertRaisesRegex(RuntimeError, "Missing key.*perception_gate"):
            restored.load_state_dict(without_fusion, strict=True)
        state["memory_retriever.unrelated_weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            restored.load_state_dict(state, strict=True)

    def test_one_gate_evaluation_is_reused_in_all_retrieval_layers(self):
        retriever = make_retriever()
        calls = []
        biases = []
        handle = retriever.slot_gate.register_forward_hook(
            lambda module, args, output: calls.append(output)
        )
        handles = [block.history_cross_attn.register_forward_pre_hook(
            lambda module, args, kwargs: biases.append(kwargs["history_log_gate"]),
            with_kwargs=True,
        ) for block in retriever.perception_retrieval_blocks]
        records = {"history_slot_gates": []}
        retriever(
            global_cond=torch.randn(1, 4, 16),
            local_cond=torch.randn(1, 1, 16),
            history_latents=torch.randn(1, 3, 4, 16),
            history_mask=torch.tensor([[True, True, False]]),
            record_data_dict=records,
        )
        handle.remove()
        for hook in handles:
            hook.remove()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(biases), 2)
        self.assertTrue(all(bias is calls[0] for bias in biases))
        self.assertEqual(retriever.latest_slot_gates.shape, (1, 3, 4))
        self.assertIsNone(retriever.latest_slot_gates.grad_fn)
        self.assertEqual(float(retriever.latest_slot_gates[:, 2].abs().sum()), 0)
        torch.testing.assert_close(records["history_slot_gates"][0], retriever.latest_slot_gates)

    def test_proprioception_changes_gate_for_fixed_image_and_history(self):
        torch.manual_seed(4)
        retriever = make_retriever().eval()
        current = torch.randn(1, 4, 16)
        history = torch.randn(1, 3, 4, 16)
        retriever(global_cond=current, history_latents=history, local_cond=torch.randn(1, 1, 16))
        first = retriever.latest_slot_gates.clone()
        retriever(global_cond=current, history_latents=history, local_cond=torch.randn(1, 1, 16))
        self.assertGreater(float((first - retriever.latest_slot_gates).abs().max()), 1e-6)

    def test_no_history_sample_keeps_gate_parameters_in_backward_graph(self):
        retriever = make_retriever()
        output = retriever(
            global_cond=torch.randn(2, 4, 16),
            local_cond=torch.randn(2, 1, 16),
            history_latents=torch.randn(2, 3, 4, 16),
            history_mask=torch.zeros(2, 3, dtype=torch.bool),
        )
        self.assertTrue(torch.isfinite(output).all())
        output.square().sum().backward()
        for parameter in retriever.slot_gate.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


class ConditionedSlotGatePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        future_fixture.FuturePredictionModeTest.setUpClass()

    def setUp(self):
        self.fixture = future_fixture.FuturePredictionModeTest()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def test_streaming_projection_omits_unused_sliding_windows(self):
        for cached in (True, False):
            with self.subTest(cached=cached):
                policy = self.fixture.instantiate_policy(self.fixture.make_config("tokens", cached))
                try:
                    batch = self.fixture.make_batch(cached)
                    data, _ = policy._encode_input_multi_traj(batch)
                    network = policy.denoising_network
                    legacy = network.prepare_parallel_forward(dict(data))
                    with patch.object(
                        network, "add_history_key_encoding",
                        side_effect=AssertionError("streaming must not build sliding-window keys"),
                    ):
                        streaming = network.prepare_parallel_forward({**data, "_streaming_training": True})
                    for name in ("history_latents", "history_key_latents", "history_mask"):
                        self.assertIn(name, legacy)
                        self.assertNotIn(name, streaming)
                    self.assertEqual(tuple(streaming["all_history_latents"].shape), (1, 3, 16, 32))
                    for name in ("all_history_latents", "x", "global_cond", "local_cond", "step"):
                        torch.testing.assert_close(streaming[name], legacy[name])

                    # Recreate the redundant-window path under identical
                    # randomness to verify losses and retained slots are unchanged.
                    prepare = network.prepare_parallel_forward
                    torch.manual_seed(7)
                    policy.torch_rng.manual_seed(7)
                    streaming_losses = policy.compute_loss(batch)
                    retained_slots = policy.latest_training_slot_ids
                    torch.manual_seed(7)
                    policy.torch_rng.manual_seed(7)
                    with patch.object(
                        network, "prepare_parallel_forward",
                        side_effect=lambda data: prepare({**data, "_streaming_training": False}),
                    ):
                        legacy_losses = policy.compute_loss(batch)
                    for name, value in streaming_losses.items():
                        torch.testing.assert_close(value, legacy_losses[name], rtol=0, atol=0)
                    self.assertEqual(policy.latest_training_slot_ids, retained_slots)
                finally:
                    del policy

    def test_action_and_future_losses_each_train_both_gates_in_both_feature_modes(self):
        for mode in ("pooled", "tokens"):
            for cached in (True, False):
                with self.subTest(mode=mode, cached=cached):
                    cfg = self.fixture.make_config(mode, cached)
                    self.assertEqual(cfg.workspace.model.gate_selectivity_min_std, 0.1)
                    policy = self.fixture.instantiate_policy(cfg)
                    retriever = policy.denoising_network.memory_retriever
                    gate = retriever.slot_gate
                    self.assertIsNotNone(gate)
                    fusion = retriever.perception_gate
                    # The network-wide Xavier initialization must not overwrite
                    # the original small fusion initialization (g starts near .5).
                    self.assertLess(float(fusion.proj.weight.std()), 0.003)
                    batch = self.fixture.make_batch(cached)
                    losses = policy.compute_loss(batch)
                    parameters = tuple(gate.parameters())
                    for name in ("action", "future_prediction"):
                        for module in (gate, fusion):
                            gradients = torch.autograd.grad(
                                losses[name], tuple(module.parameters()), retain_graph=True,
                            )
                            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
                    (losses["action"] + 0.05 * losses["future_prediction"]).backward()
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in parameters), 0)
                    del policy

    def test_gate_trains_across_chunk_and_optimizer_boundaries(self):
        policy = self.fixture.instantiate_policy(self.fixture.make_config("tokens"))
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.001)
        batch = self.fixture.make_batch(anchors=7)
        for start in (0, 3):
            chunk = {key: value[:, start:start + 4] for key, value in batch.items()}
            chunk["_streaming_supervised_length"] = 3
            chunk["_streaming_reset"] = start == 0
            chunk["_streaming_final"] = start == 3
            optimizer.zero_grad(set_to_none=True)
            losses = policy.compute_loss(chunk)
            total = losses["action"] + 0.05 * losses["future_prediction"]
            self.assertTrue(torch.isfinite(total))
            total.backward()
            gate = policy.denoising_network.memory_retriever.slot_gate
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in gate.parameters()), 0)
            optimizer.step()
        self.assertIsNone(policy._streaming_training_state)

    def test_online_gate_is_reused_across_diffusion_steps(self):
        cfg = self.fixture.make_config("tokens")
        cfg.workspace.model.noise_scheduler.inference_step_num = 4
        policy = self.fixture.instantiate_policy(cfg).eval()
        gate = policy.denoising_network.memory_retriever.slot_gate
        calls = []
        handle = gate.register_forward_hook(lambda module, args, output: calls.append(output.detach()))
        batch = self.fixture.make_batch()
        for anchor in range(3):
            before = len(calls)
            with torch.no_grad():
                output = policy.predict_action({
                    "head_camera_feature": batch["head_camera_feature"][:, anchor],
                    "agent_pos": batch["agent_pos"][:, anchor],
                    "episode_idx": torch.tensor([0]),
                })
            self.assertEqual(len(calls) - before, 1)
            self.assertEqual(output["action"].shape, (1, 16, 14))
            self.assertTrue(torch.isfinite(output["action"]).all())
            self.assertEqual(policy.denoising_network.memory_retriever.latest_slot_gates.shape, (1, 2, 4))
        handle.remove()

    def test_memory_policy_requires_conditioned_gate(self):
        cfg = self.fixture.make_config()
        network_cfg = cfg.workspace.model.denoising_network_partial
        network_cfg.use_conditioned_slot_gate = False
        with self.assertRaisesRegex(hydra.errors.InstantiationException, "requires use_conditioned_slot_gate"):
            self.fixture.instantiate_policy(cfg)


if __name__ == "__main__":
    unittest.main()
