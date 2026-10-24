import gc
import unittest

from omegaconf import open_dict
import torch

from imitation_learning.models.denoising_networks.memory_transformer import ObservationDrivenMemoryRetriever
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
            hidden_dim=16, head_num=4, decision_memory_token_num=4,
            history_attention_type="token_wise", ssmax_scaling_param=None,
            binary_gating=False, straight_through="",
            retrieval_mode="perception_patches", use_conditioned_slot_gate=True,
            fusion_gate=enabled,
        )

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
