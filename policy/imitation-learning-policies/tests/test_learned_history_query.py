import gc
import unittest

from omegaconf import open_dict
import torch

from imitation_learning.models.denoising_networks.memory_transformer import (
    ObservationDrivenMemoryRetriever,
)
from imitation_learning.models.encoders.image_encoders import SharedModelManager
import test_future_prediction_mode as future_fixture


def make_retriever(tokens=4, **overrides):
    options = dict(
        hidden_dim=16, head_num=4, decision_memory_token_num=4,
        history_attention_type="token_wise", ssmax_scaling_param=None,
        binary_gating=False, straight_through="",
        retrieval_mode="perception_patches", use_conditioned_slot_gate=True,
        use_learned_history_query=True, perception_token_num=tokens,
    )
    options.update(overrides)
    return ObservationDrivenMemoryRetriever(**options)


class LearnedHistoryQueryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_learned_initialization_enters_history_but_not_slot_gate_or_fusion(self):
        retriever = make_retriever().eval()
        image, state = torch.randn(2, 4, 16), torch.randn(2, 1, 16)
        current = retriever.condition_current_tokens(image, state)
        initial = retriever.history_query_tokens.detach().clone()
        reads, attention, gates, fusion = [], [], [], []
        handles = [
            retriever.history_query_cross_attn.register_forward_hook(
                lambda module, args, output: attention.append((args, output))
            ),
            retriever.perception_retrieval_blocks[0].register_forward_pre_hook(
                lambda module, args, kwargs: reads.append(kwargs["query"]),
                with_kwargs=True,
            ),
            retriever.slot_gate.register_forward_pre_hook(
                lambda module, args, kwargs: gates.append(kwargs["current_tokens"]),
                with_kwargs=True,
            ),
            retriever.perception_gate.register_forward_pre_hook(
                lambda module, args: fusion.append(args)
            ),
        ]
        try:
            for _ in range(2):
                output = retriever(
                    global_cond=image, local_cond=state,
                    history_latents=torch.randn(2, 3, 4, 16),
                )
                self.assertEqual(tuple(output.shape), (2, 4, 16))
                args, update = attention[-1]
                torch.testing.assert_close(args[0], initial.expand(2, -1, -1))
                torch.testing.assert_close(args[1], torch.cat([image, state], dim=1))
                torch.testing.assert_close(reads[-1], args[0] + update)
                torch.testing.assert_close(gates[-1], current)
                torch.testing.assert_close(fusion[-1][0], current)
                torch.testing.assert_close(output, retriever.perception_gate(*fusion[-1]))
            torch.testing.assert_close(reads[0], reads[1])
            torch.testing.assert_close(retriever.history_query_tokens, initial)
        finally:
            for handle in handles:
                handle.remove()
        query = retriever.initialize_history_query(image, state)
        for changed_image, changed_state in ((image.flip(-1), state), (image, state.flip(-1))):
            changed = retriever.initialize_history_query(changed_image, changed_state)
            self.assertGreater(float((query - changed).abs().max()), 1e-5)

    def test_context_masks_exclude_padded_image_and_state_tokens(self):
        retriever = make_retriever()
        image, state = torch.randn(2, 4, 16), torch.randn(2, 2, 16)
        image_mask = torch.tensor([[True, True, False, False], [True] * 4])
        state_mask = torch.tensor([[True, False], [False, True]])
        for masks in ((image_mask, state_mask), (image_mask, None), (None, state_mask)):
            changed = [image.clone(), state.clone()]
            for tokens, mask in zip(changed, masks):
                if mask is not None:
                    tokens.masked_fill_(~mask[..., None], float("nan"))
            expected = retriever.initialize_history_query(image, state, *masks)
            actual = retriever.initialize_history_query(*changed, *masks)
            torch.testing.assert_close(actual, expected)
            self.assertTrue(torch.isfinite(actual).all())

    def test_no_history_and_disabled_fusion_preserve_current_fallback(self):
        for fusion_gate in (True, False):
            retriever = make_retriever(fusion_gate=fusion_gate)
            image, state = torch.randn(2, 4, 16), torch.randn(2, 1, 16)
            current = retriever.condition_current_tokens(image, state)
            torch.testing.assert_close(retriever(global_cond=image, local_cond=state), current)
            output = retriever(
                global_cond=image, local_cond=state,
                history_latents=torch.randn(2, 3, 4, 16),
                history_mask=torch.tensor([[True] * 3, [False] * 3]),
            )
            torch.testing.assert_close(output[1], current[1])
            output.square().sum().backward()
            for parameter in (retriever.history_query_tokens,
                              *retriever.history_query_cross_attn.parameters()):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_query_count_validation_and_enabled_state_roundtrip(self):
        for tokens in (1, 4, 12):
            retriever = make_retriever(tokens)
            restored = make_retriever(tokens)
            restored.load_state_dict(retriever.state_dict(), strict=True)
            inputs = dict(global_cond=torch.randn(1, tokens, 16),
                          history_latents=torch.randn(1, 3, tokens, 16))
            torch.testing.assert_close(restored(**inputs), retriever(**inputs))
            self.assertEqual(tuple(retriever.history_query_tokens.shape), (1, tokens, 16))
        initialized = make_retriever().initialize_history_query(torch.randn(1, 12, 16))
        self.assertEqual(tuple(initialized.shape), (1, 4, 16))
        with self.assertRaisesRegex(ValueError, "use_learned_history_query"):
            make_retriever(use_learned_history_query="false")
        with self.assertRaisesRegex(ValueError, "perception_patches"):
            make_retriever(retrieval_mode="learned_queries")
        with self.assertRaisesRegex(ValueError, "perception_token_num"):
            make_retriever(perception_token_num=None)

    def test_explicit_query_count_controls_initialization_and_rejects_invalid_values(self):
        image, state = torch.randn(2, 4, 16), torch.randn(2, 1, 16)
        for count in (1, 2, 4, 8, 16):
            with self.subTest(count=count):
                retriever = make_retriever(query_tokens=count)
                self.assertEqual(tuple(retriever.history_query_tokens.shape), (1, count, 16))
                self.assertEqual(tuple(retriever.initialize_history_query(image, state).shape),
                                 (2, count, 16))
        for count in (0, -1, True, 2.5, "4"):
            with self.subTest(invalid=count):
                with self.assertRaisesRegex(ValueError, "query_tokens"):
                    make_retriever(query_tokens=count)

    def test_different_query_count_aligns_before_fusion_and_preserves_empty_rows(self):
        image, state = torch.randn(2, 4, 16), torch.randn(2, 1, 16)
        for count in (1, 2, 8, 16):
            for fusion_gate in (True, False):
                with self.subTest(count=count, fusion_gate=fusion_gate):
                    retriever = make_retriever(query_tokens=count, fusion_gate=fusion_gate)
                    current = retriever.condition_current_tokens(image, state)
                    reads, aligned = [], []
                    handles = [
                        retriever.perception_retrieval_blocks[0].register_forward_pre_hook(
                            lambda module, args, kwargs: reads.append(kwargs["query"]),
                            with_kwargs=True,
                        ),
                        retriever.history_query_output_attn.register_forward_hook(
                            lambda module, args, output: aligned.append((args, output))
                        ),
                    ]
                    try:
                        inputs = dict(global_cond=image, local_cond=state,
                                      history_latents=torch.randn(2, 3, 4, 16),
                                      history_mask=torch.tensor([[True] * 3, [False] * 3]))
                        result = retriever(**inputs)
                        self.assertEqual(tuple(reads[0].shape), (2, count, 16))
                        args, update = aligned[0]
                        torch.testing.assert_close(args[0], current)
                        self.assertEqual(tuple(args[1].shape), (2, count, 16))
                        retrieved = current + update
                        expected = retriever.perception_gate(current, retrieved) if fusion_gate else retrieved
                        torch.testing.assert_close(result[0], expected[0])
                        torch.testing.assert_close(result[1], current[1])
                        self.assertEqual(tuple(result.shape), (2, 4, 16))
                        result.square().sum().backward()
                        self.assertGreater(float(retriever.history_query_tokens.grad.abs().sum()), 0)
                        self.assertGreater(sum(float(p.grad.abs().sum()) for p in
                                               retriever.history_query_output_attn.parameters()), 0)
                        zero_gated = retriever(**inputs, memory_gate_val=torch.zeros(2))
                        torch.testing.assert_close(zero_gated, current)
                        torch.testing.assert_close(retriever(global_cond=image, local_cond=state), current)
                    finally:
                        for handle in handles:
                            handle.remove()

    def test_query_count_checkpoint_roundtrip_and_equal_count_uses_original_parameters(self):
        default = make_retriever()
        same = make_retriever(query_tokens=4)
        same.load_state_dict(default.state_dict(), strict=True)
        self.assertFalse(hasattr(same, "history_query_output_attn"))
        inputs = dict(global_cond=torch.randn(1, 4, 16),
                      history_latents=torch.randn(1, 3, 4, 16))
        torch.testing.assert_close(same(**inputs), default(**inputs), rtol=0, atol=0)
        different = make_retriever(query_tokens=8)
        restored = make_retriever(query_tokens=8)
        restored.load_state_dict(different.state_dict(), strict=True)
        torch.testing.assert_close(restored(**inputs), different(**inputs), rtol=0, atol=0)
        disabled = make_retriever(use_learned_history_query=False, query_tokens=8)
        self.assertFalse(hasattr(disabled, "history_query_tokens"))
        self.assertFalse(hasattr(disabled, "history_query_output_attn"))


class LearnedHistoryQueryPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        future_fixture.FuturePredictionModeTest.setUpClass()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def config(self, mode="tokens", cached=True):
        fixture = future_fixture.FuturePredictionModeTest()
        cfg = fixture.make_config(mode, cached, merge_size=8)
        cfg.workspace.model.denoising_network_partial.use_learned_history_query = True
        cfg.workspace.model.denoising_network_partial.shadow_horizon = 3
        return cfg

    def test_token_count_follows_cameras_and_spatial_merge_automatically(self):
        fixture = future_fixture.FuturePredictionModeTest()
        for cameras, merge, expected in ((["head_camera"], 8, 4),
                                         (["head_camera"], 4, 16),
                                         (["head_camera", "left_camera", "right_camera"], 8, 12)):
            cfg = self.config()
            cfg.visual_encoder.spatial_merge_size = merge
            cfg.workspace.train_dataset.image_keys = cameras
            policy = fixture.instantiate_policy(cfg)
            network = policy.denoising_network
            self.assertEqual(network.decision_memory_token_num, expected)
            self.assertEqual(tuple(network.memory_retriever.history_query_tokens.shape), (1, expected, 32))
            del policy

    def test_action_and_future_each_train_query_for_cached_and_raw_features(self):
        fixture = future_fixture.FuturePredictionModeTest()
        for mode in ("pooled", "tokens"):
            for cached in (True, False):
                with self.subTest(mode=mode, cached=cached):
                    policy = fixture.instantiate_policy(self.config(mode, cached))
                    retriever = policy.denoising_network.memory_retriever
                    losses = policy.compute_loss(fixture.make_batch(cached, anchors=5))
                    parameters = (retriever.history_query_tokens,
                                  *retriever.history_query_cross_attn.parameters())
                    for loss_name in ("action", "future_prediction"):
                        gradients = torch.autograd.grad(losses[loss_name], parameters, retain_graph=True)
                        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
                        self.assertGreater(float(gradients[0].abs().sum()), 0)
                        self.assertGreater(sum(float(grad.abs().sum()) for grad in gradients[1:]), 0)
                    del policy

    def test_streaming_optimizer_boundaries_and_once_per_observation_inference(self):
        fixture = future_fixture.FuturePredictionModeTest()
        cfg = self.config()
        cfg.workspace.model.denoising_network_partial.query_tokens = 8
        cfg.workspace.model.noise_scheduler.inference_step_num = 4
        policy = fixture.instantiate_policy(cfg)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.001)
        batch = fixture.make_batch(anchors=7)
        for start in (0, 3):
            chunk = {key: value[:, start:start + 4] for key, value in batch.items()}
            chunk.update(_streaming_supervised_length=3, _streaming_reset=start == 0,
                         _streaming_final=start == 3)
            optimizer.zero_grad(set_to_none=True)
            losses = policy.compute_loss(chunk)
            total = losses["action"] + .05 * losses["future_prediction"] + .01 * losses["gate_selectivity"]
            self.assertTrue(torch.isfinite(total))
            total.backward()
            self.assertGreater(float(policy.denoising_network.memory_retriever.history_query_tokens.grad.abs().sum()), 0)
            optimizer.step()
        calls = []
        retriever = policy.denoising_network.memory_retriever
        handle = retriever.history_query_cross_attn.register_forward_hook(
            lambda module, args, output: calls.append(output.detach())
        )
        policy.eval()
        try:
            for anchor in range(3):
                before = len(calls)
                with torch.no_grad():
                    output = policy.predict_action({
                        "head_camera_feature": batch["head_camera_feature"][:, anchor],
                        "agent_pos": batch["agent_pos"][:, anchor],
                        "episode_idx": torch.tensor([0]),
                    })
                self.assertEqual(len(calls) - before, 1)
                self.assertEqual(tuple(output["action"].shape), (1, 16, 14))
                self.assertTrue(torch.isfinite(output["action"]).all())
        finally:
            handle.remove()

    def test_checkpoint_without_flag_strictly_loads_original_path(self):
        fixture = future_fixture.FuturePredictionModeTest()
        cfg = fixture.make_config()
        self.assertFalse(cfg.workspace.model.denoising_network_partial.use_learned_history_query)
        policy = fixture.instantiate_policy(cfg)
        state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        self.assertFalse(any("history_query" in key for key in state))
        del policy
        with open_dict(cfg.workspace.model.denoising_network_partial):
            del cfg.workspace.model.denoising_network_partial["use_learned_history_query"]
        policy = fixture.instantiate_policy(cfg)
        policy.load_state_dict(state, strict=True)
        self.assertFalse(policy.denoising_network.memory_retriever.use_learned_history_query)

    def test_explicit_query_count_trains_with_both_time_and_prediction_modes(self):
        fixture = future_fixture.FuturePredictionModeTest()
        for count, mode, cached, time_mode in ((2, "pooled", True, "relative"),
                                               (8, "tokens", False, "absolute")):
            with self.subTest(count=count, mode=mode, cached=cached, time=time_mode):
                cfg = self.config(mode, cached)
                cfg.workspace.model.denoising_network_partial.query_tokens = count
                cfg.workspace.model.denoising_network_partial.history_time_encoding = time_mode
                policy = fixture.instantiate_policy(cfg)
                network = policy.denoising_network
                retriever = network.memory_retriever
                self.assertEqual(network.decision_memory_token_num, 4)
                self.assertEqual(tuple(retriever.history_query_tokens.shape), (1, count, 32))
                losses = policy.compute_loss(fixture.make_batch(cached, anchors=5))
                parameters = (retriever.history_query_tokens,
                              *retriever.history_query_output_attn.parameters())
                for name in ("action", "future_prediction"):
                    gradients = torch.autograd.grad(losses[name], parameters, retain_graph=True)
                    self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                    self.assertGreater(float(gradients[0].abs().sum()), 0)
                    self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients[1:]), 0)
                del policy


if __name__ == "__main__":
    unittest.main()
