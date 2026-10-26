import gc
import inspect
import unittest
from unittest.mock import patch

from omegaconf import open_dict
import torch

from imitation_learning.models.denoising_networks.memory_transformer import (
    MemoryTransformer, ObservationDrivenMemoryRetriever,
    PerceptionHistoryRetrievalBlock, TokenGateFusion,
)
from imitation_learning.models.encoders.image_encoders import SharedModelManager
import test_future_prediction_mode as future_fixture


class FusionGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        future_fixture.FuturePredictionModeTest.setUpClass()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def retriever(self, enabled=True):
        return ObservationDrivenMemoryRetriever(
            hidden_dim=16, head_num=4,
            history_attention_type="token_wise", ssmax_scaling_param=None,
            binary_gating=False, straight_through="",
            retrieval_mode="perception_patches", use_conditioned_slot_gate=True,
            fusion_gate=enabled,
        )

    def test_gate_can_select_current_retrieved_or_their_average(self):
        gate = TokenGateFusion(16)
        current, retrieved = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
        with torch.no_grad():
            gate.proj.weight.zero_()
            gate.proj.bias.zero_()
            torch.testing.assert_close(gate(current, retrieved), .5 * (current + retrieved))
            gate.proj.bias.fill_(30)
            torch.testing.assert_close(gate(current, retrieved), current)
            gate.proj.bias.fill_(-30)
            torch.testing.assert_close(gate(current, retrieved), retrieved)

    def test_retrieval_retains_query_residual_even_when_attention_read_is_fixed(self):
        block = PerceptionHistoryRetrievalBlock(16, 4, "token_wise", None)
        query = torch.randn(2, 4, 16, requires_grad=True)
        fixed_read = torch.randn_like(query)
        history = torch.randn(2, 3, 4, 16)
        with patch.object(block.history_cross_attn, "forward", return_value=fixed_read):
            output = block(query, history, None, None)
            changed = block(query + torch.randn_like(query), history, None, None)
        self.assertGreater(float((output - changed).abs().max()), 1e-4)
        (output * torch.randn_like(output)).sum().backward()
        self.assertGreater(float(query.grad.abs().sum()), 1e-4)

    def test_enabled_gate_uses_current_for_empty_masked_and_skipped_history(self):
        retriever = self.retriever()
        current, history = torch.randn(2, 4, 16), torch.randn(2, 3, 4, 16)
        for inputs in ({}, {"history_latents": history[:, :0]},
                       {"history_latents": history, "skip_history": True},
                       {"history_latents": history, "history_mask": torch.zeros(2, 3, dtype=torch.bool)}):
            torch.testing.assert_close(retriever(global_cond=current, **inputs), current)
        captured = []
        handle = retriever.perception_gate.register_forward_pre_hook(
            lambda module, args: captured.append(args)
        )
        mixed = retriever(global_cond=current, history_latents=history,
                          history_mask=torch.tensor([[True] * 3, [False] * 3]))
        handle.remove()
        torch.testing.assert_close(captured[0][1][1], current[1])
        torch.testing.assert_close(mixed[1], current[1])
        self.assertTrue(all(p.requires_grad for p in retriever.perception_gate.parameters()))

    def test_retired_option_is_absent_and_saved_metadata_cannot_reenable_it(self):
        for cls in (MemoryTransformer, ObservationDrivenMemoryRetriever, PerceptionHistoryRetrievalBlock):
            self.assertNotIn("history_only", inspect.signature(cls).parameters)
        fixture = future_fixture.FuturePredictionModeTest()
        for task in ("rmbench", "observe_and_pickup"):
            cfg = fixture.make_config(task=task)
            self.assertNotIn("history_only", cfg.workspace.model.denoising_network_partial)
            self.assertTrue(cfg.workspace.model.denoising_network_partial.fusion_gate)
        cfg = fixture.make_config()
        policy = fixture.instantiate_policy(cfg)
        state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        del policy
        with open_dict(cfg.workspace.model.denoising_network_partial):
            cfg.workspace.model.denoising_network_partial.history_only = True
        policy = fixture.instantiate_policy(cfg)
        policy.load_state_dict(state, strict=True)
        self.assertFalse(hasattr(policy.denoising_network, "history_only"))
        self.assertFalse(hasattr(policy.denoising_network.memory_retriever, "history_only"))
        self.assertTrue(all(p.requires_grad for p in policy.denoising_network.memory_retriever.perception_gate.parameters()))

    def test_dit_and_future_share_fused_memory_in_training_and_online(self):
        fixture = future_fixture.FuturePredictionModeTest()
        for mode in ("pooled", "tokens"):
            for cached in (False, True):
                for projected in (False, True):
                    with self.subTest(mode=mode, cached=cached, projected=projected):
                        cfg = fixture.make_config(mode, cached, merge_size=8)
                        cfg.workspace.model.denoising_network_partial.shadow_horizon = 2
                        cfg.workspace.model.denoising_network_partial.query_tokens = 8 if projected else None
                        cfg.workspace.model.denoising_network_partial.history_time_encoding = (
                            "absolute" if projected else "relative"
                        )
                        policy = fixture.instantiate_policy(cfg)
                        batch = fixture.make_batch(cached, anchors=4)
                        suffix = "_feature" if cached else ""
                        for i, camera in enumerate(("left_camera", "right_camera"), 1):
                            batch[camera + suffix] = batch["head_camera" + suffix] + i * .1
                        memories, retrieval_inputs, dit_inputs, future_inputs, token_counts = [], [], [], [], []
                        retriever = policy.denoising_network.memory_retriever
                        handles = [
                            retriever.register_forward_pre_hook(
                                lambda module, args, kwargs: retrieval_inputs.append(kwargs), with_kwargs=True),
                            retriever.register_forward_hook(lambda module, args, output: memories.append(output)),
                            policy.denoising_network.blocks[0].register_forward_pre_hook(
                                lambda module, args, kwargs: dit_inputs.append(kwargs), with_kwargs=True),
                            policy.future_feature_predictor.register_forward_pre_hook(
                                lambda module, args, kwargs: future_inputs.append(kwargs), with_kwargs=True),
                            policy.future_feature_predictor.transformer.register_forward_pre_hook(
                                lambda module, args: token_counts.append(args[0].shape[1])),
                        ]
                        try:
                            losses = policy.compute_loss(batch)
                            training_memory = torch.cat(memories)
                            current = retriever.condition_current_tokens(
                                retrieval_inputs[0]["global_cond"],
                                retrieval_inputs[0].get("local_cond"),
                                retrieval_inputs[0].get("local_cond_mask"),
                            )
                            current = retriever.project_current_query(current)
                            torch.testing.assert_close(training_memory[:1], current)
                            self.assertNotIn("decision_memory_mask", dit_inputs[0])
                            self.assertEqual(dit_inputs[0]["global_cond"].shape[1], 4)
                            torch.testing.assert_close(dit_inputs[0]["decision_memory"], training_memory)
                            self.assertEqual(len(future_inputs), 1)
                            self.assertNotIn("memory_context_tokens", future_inputs[0])
                            expected = training_memory[1:3]
                            torch.testing.assert_close(future_inputs[0]["current_observation_tokens"], expected)
                            self.assertEqual(future_inputs[0]["action_chunk"].shape[1], 10)
                            self.assertEqual(token_counts, [expected.shape[1] + 1 + policy.future_feature_predictor.future_token_num])
                            losses["future_prediction"].backward()
                            self.assertGreater(float(retriever.perception_gate.proj.weight.grad.abs().sum()), 0)
                            slot_grads = [p.grad for p in retriever.slot_gate.parameters()]
                            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in slot_grads))
                            if cached:
                                self.assertGreater(sum(float(g.abs().sum()) for g in slot_grads), 0)
                            # Inference hard-deletes. Compare its numerical path
                            # to an Active-only training reference, while the
                            # preceding gradient checks still exercise Shadow.
                            memories.clear()
                            policy.eval()
                            with patch.object(policy.denoising_network, "effective_shadow_horizon", 0), torch.no_grad():
                                policy.compute_loss(batch)
                            expected_online = torch.cat(memories).detach().clone()
                            memories.clear()
                            future_inputs.clear()
                            policy.reset()
                            for anchor in range(4):
                                obs = {key: value[:, anchor] for key, value in batch.items()
                                       if key not in {"action", "traj_idx", "entire_traj_is_padding"}}
                                obs["episode_idx"] = torch.tensor([0])
                                with torch.inference_mode():
                                    output = policy.predict_action(obs)
                                self.assertTrue(torch.isfinite(output["action"]).all())
                                self.assertNotIn("decision_memory_mask", dit_inputs[-1])
                            self.assertEqual(future_inputs, [])
                            torch.testing.assert_close(torch.cat(memories), expected_online, rtol=2e-5, atol=2e-5)
                        finally:
                            for handle in handles:
                                handle.remove()
                        del policy, losses

    def test_disabled_gate_uses_current_plus_history_residual_without_double_counting(self):
        retriever = self.retriever()
        captured = []
        handle = retriever.perception_gate.register_forward_pre_hook(
            lambda module, args: captured.append(args),
        )
        inputs = dict(global_cond=torch.randn(1, 4, 16),
                      history_latents=torch.randn(1, 3, 4, 16))
        enabled = retriever(**inputs)
        handle.remove()
        current, retrieved = captured[0]
        torch.testing.assert_close(enabled, retriever.perception_gate(current, retrieved))
        retriever.fusion_gate = False
        disabled = retriever(**inputs)
        torch.testing.assert_close(disabled, retrieved)
        self.assertGreater(float((disabled - current).abs().max()), 1e-5)
        self.assertGreater(float((disabled - (current + retrieved)).abs().max()), 1e-5)

    def test_disabled_gate_keeps_current_observation_when_history_is_empty_or_masked(self):
        retriever = self.retriever(False)
        current = torch.randn(2, 4, 16)
        torch.testing.assert_close(retriever(global_cond=current), current)
        output = retriever(global_cond=current, history_latents=torch.randn(2, 3, 4, 16),
                           history_mask=torch.zeros(2, 3, dtype=torch.bool))
        torch.testing.assert_close(output, current)
        self.assertTrue(all(not p.requires_grad for p in retriever.perception_gate.parameters()))
        with self.assertRaisesRegex(ValueError, "fusion_gate"):
            self.retriever("false")

    def test_disabled_gate_trains_action_future_and_slot_gate_in_both_modes(self):
        fixture = future_fixture.FuturePredictionModeTest()
        for mode in ("pooled", "tokens"):
            with self.subTest(mode=mode):
                cfg = fixture.make_config(mode)
                cfg.workspace.model.denoising_network_partial.fusion_gate = False
                cfg.workspace.model.denoising_network_partial.shadow_horizon = 3
                policy = fixture.instantiate_policy(cfg)
                losses = policy.compute_loss(fixture.make_batch(anchors=8))
                total = losses["action"] + .05 * losses["future_prediction"] + .01 * losses["gate_selectivity"]
                self.assertTrue(torch.isfinite(total))
                total.backward()
                retriever = policy.denoising_network.memory_retriever
                self.assertFalse(retriever.fusion_gate)
                self.assertTrue(all(p.grad is None for p in retriever.perception_gate.parameters()))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in retriever.slot_gate.parameters()), 0)
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in policy.future_feature_predictor.parameters()
                                       if p.grad is not None), 0)
                del policy

    def test_checkpoint_without_flag_defaults_to_enabled_and_preserves_state_shapes(self):
        fixture = future_fixture.FuturePredictionModeTest()
        cfg = fixture.make_config()
        policy = fixture.instantiate_policy(cfg)
        state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        del policy
        with open_dict(cfg.workspace.model.denoising_network_partial):
            del cfg.workspace.model.denoising_network_partial["fusion_gate"]
        policy = fixture.instantiate_policy(cfg)
        policy.load_state_dict(state, strict=True)
        self.assertTrue(policy.denoising_network.fusion_gate)
        self.assertTrue(policy.denoising_network.memory_retriever.fusion_gate)
        del policy
        with open_dict(cfg.workspace.model.denoising_network_partial):
            cfg.workspace.model.denoising_network_partial.fusion_gate = False
        policy = fixture.instantiate_policy(cfg)
        policy.load_state_dict(state, strict=True)
        self.assertFalse(policy.denoising_network.memory_retriever.fusion_gate)


if __name__ == "__main__":
    unittest.main()
