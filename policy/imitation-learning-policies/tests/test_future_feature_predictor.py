import unittest

import torch

from imitation_learning.models.future_prediction.future_feature_predictor import (
    FutureFeaturePredictor,
)


def make_predictor(use_action_condition: bool = True, **kwargs) -> FutureFeaturePredictor:
    torch.manual_seed(0)
    return FutureFeaturePredictor(
        hidden_dim=32,
        output_feature_dim=24,
        action_dim=8,
        head_num=4,
        layer_num=2,
        mlp_ratio=2.0,
        dropout=0.0,
        future_token_num=2,
        max_action_length=16,
        use_action_condition=use_action_condition,
        **kwargs,
    )


class FutureFeaturePredictorTest(unittest.TestCase):
    def test_chunk_condition_has_one_token_at_nondivisible_width(self):
        predictor = FutureFeaturePredictor(
            hidden_dim=32, predictor_hidden_dim=512, output_feature_dim=24,
            action_dim=14, head_num=8, layer_num=1, max_action_length=16,
            action_condition_mode="chunk", action_chunk_length=10,
        )
        seen = []
        handle = predictor.transformer.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].shape)
        )
        current = torch.randn(2, 1, 32, requires_grad=True)
        actions = torch.randn(2, 10, 14, requires_grad=True)
        prediction = predictor(current, action_chunk=actions)
        handle.remove()
        self.assertEqual(seen, [torch.Size([2, 4, 512])])  # current + action + 2 targets
        self.assertIsNone(predictor.action_position_embedding)
        prediction.square().mean().backward()
        self.assertTrue(torch.isfinite(actions.grad).all())
        self.assertTrue(bool((actions.grad.abs().sum(dim=-1) > 0).all()))
        self.assertGreater(float(current.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is not None for p in predictor.action_projector.parameters()))

    def test_chunk_embedding_preserves_action_order(self):
        predictor = FutureFeaturePredictor(
            hidden_dim=4, output_feature_dim=4, action_dim=1, head_num=1,
            layer_num=1, max_action_length=2, action_condition_mode="chunk",
        )
        with torch.no_grad():
            predictor.action_type_embedding.zero_()
            encoder = predictor.action_projector
            encoder.patch_embed.weight.copy_(torch.tensor([[[1.]], [[2.]]]))
            encoder.patch_embed.bias.zero_()
            encoder.out_project.weight.copy_(torch.eye(4))
            encoder.out_project.bias.zero_()
        actions = torch.tensor([[[1.], [2.]]])
        forward, _ = predictor._encode_actions(actions, None, 1)
        reverse, _ = predictor._encode_actions(actions.flip(1), None, 1)
        torch.testing.assert_close(forward, torch.tensor([[[1., 2., 2., 4.]]]))
        self.assertFalse(torch.equal(forward, reverse))

    def test_chunk_mask_ignores_nan_padding_and_preserves_short_input_positions(self):
        predictor = make_predictor(action_condition_mode="chunk", action_chunk_length=10).eval()
        current = torch.randn(2, 1, 32)
        actions = torch.randn(2, 6, 8, requires_grad=True)
        mask = torch.tensor([[True, True, False, True, False, False],
                             [False, True, True, False, True, False]])
        full = torch.nn.functional.pad(actions.detach(), (0, 0, 0, 4))
        full_mask = torch.nn.functional.pad(mask, (0, 4), value=False)
        full[~full_mask] = float("nan")
        original = predictor(current, action_chunk=actions, action_mask=mask)
        modified = predictor(current, action_chunk=full, action_mask=full_mask)
        torch.testing.assert_close(original, modified, rtol=0, atol=0)
        original.square().mean().backward()
        self.assertTrue(torch.isfinite(actions.grad).all())
        torch.testing.assert_close(actions.grad[~mask], torch.zeros_like(actions.grad[~mask]))
        self.assertTrue(bool((actions.grad[mask].abs().sum(dim=-1) > 0).all()))
        invalid = torch.full((2, 10, 8), float("nan"))
        none_valid = torch.zeros(2, 10, dtype=torch.bool)
        missing = predictor(current, action_chunk=invalid, action_mask=none_valid)
        empty = predictor(current, action_chunk=torch.empty(2, 0, 8))
        self.assertTrue(torch.isfinite(missing).all())
        torch.testing.assert_close(missing, empty, rtol=0, atol=0)

    def test_legacy_action_default_matches_explicit_tokens_and_loads_strictly(self):
        original = make_predictor()
        restored = make_predictor(action_condition_mode="tokens")
        restored.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(original.action_position_embedding.shape, (1, 16, 32))
        self.assertIn("action_projector.weight", original.state_dict())
        self.assertFalse(any("patch_embed" in key for key in original.state_dict()))
        current, actions = torch.randn(2, 3, 32), torch.randn(2, 10, 8)
        torch.testing.assert_close(original(current, action_chunk=actions),
                                   restored(current, action_chunk=actions), rtol=0, atol=0)

    def test_chunk_settings_validate_length_and_support_disabled_conditioning(self):
        for kwargs in ({"action_condition_mode": "mean"},
                       {"action_chunk_length": 0}, {"action_chunk_length": True},
                       {"action_chunk_length": 17}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_predictor(**kwargs)
        predictor = make_predictor(action_condition_mode="chunk", action_chunk_length=10)
        with self.assertRaisesRegex(ValueError, "action_chunk_length"):
            predictor(torch.randn(1, 32), action_chunk=torch.randn(1, 11, 8))
        disabled = make_predictor(False, action_condition_mode="chunk", action_chunk_length=10)
        self.assertIsNone(disabled.action_projector)
        self.assertEqual(disabled(torch.randn(1, 32)).shape, (1, 2, 24))

    def test_mse_loss_preserves_magnitude_and_can_backpropagate_to_target(self):
        predicted = torch.tensor([[[2.0, 4.0]]], requires_grad=True)
        target = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
        loss = FutureFeaturePredictor.mse_prediction_loss(predicted, target, detach_target=False)
        torch.testing.assert_close(loss, torch.tensor(2.5))
        cosine = FutureFeaturePredictor.cosine_prediction_loss(predicted, target)
        torch.testing.assert_close(cosine, torch.tensor(0.0), atol=1e-6, rtol=0)
        loss.backward()
        torch.testing.assert_close(predicted.grad, torch.tensor([[[1.0, 2.0]]]))
        torch.testing.assert_close(target.grad, -predicted.grad)

    def test_mse_loss_masks_before_error_and_detaches_targets_by_default(self):
        predicted = torch.tensor([
            [[1.0, 3.0], [float("nan"), 1e6]], [[2.0, 4.0], [3.0, 5.0]],
        ], requires_grad=True)
        target = torch.zeros_like(predicted, requires_grad=True)
        with torch.no_grad():
            target[0, 1] = float("nan")
        mask = torch.tensor([[True, False], [True, True]])
        loss = FutureFeaturePredictor.mse_prediction_loss(predicted, target, mask)
        torch.testing.assert_close(loss, torch.tensor(64.0 / 6.0))
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(predicted.grad).all())
        torch.testing.assert_close(predicted.grad[0, 1], torch.zeros(2))
        torch.testing.assert_close(predicted.grad[mask], predicted.detach()[mask] / 3)

    def test_mse_loss_without_valid_targets_is_finite_differentiable_zero(self):
        predicted = torch.full((2, 3, 4), float("nan"), requires_grad=True)
        target = torch.full_like(predicted, float("nan"), requires_grad=True)
        loss = FutureFeaturePredictor.mse_prediction_loss(
            predicted, target, torch.zeros(2, 3, dtype=torch.bool),
        )
        self.assertEqual(float(loss), 0)
        loss.backward()
        torch.testing.assert_close(predicted.grad, torch.zeros_like(predicted))
        self.assertIsNone(target.grad)

    def test_mse_loss_accumulates_low_precision_features_in_float32(self):
        predicted = torch.tensor([[[256.0, 512.0]]], dtype=torch.bfloat16, requires_grad=True)
        loss = FutureFeaturePredictor.mse_prediction_loss(predicted, torch.zeros_like(predicted))
        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, torch.tensor(163840.0))
        loss.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        torch.testing.assert_close(predicted.grad, predicted.detach())

    def test_mse_loss_rejects_mismatched_features_and_masks(self):
        predicted = torch.randn(1, 2, 3)
        with self.assertRaisesRegex(ValueError, "same shape"):
            FutureFeaturePredictor.mse_prediction_loss(predicted, torch.randn(1, 2, 4))
        with self.assertRaisesRegex(ValueError, "target_mask"):
            FutureFeaturePredictor.mse_prediction_loss(predicted, predicted, torch.ones(1, 2, 1))
        with self.assertRaisesRegex(ValueError, "batch and feature"):
            FutureFeaturePredictor.mse_prediction_loss(torch.ones(3), torch.ones(3))

    def test_internal_transformer_can_be_wider_than_context(self) -> None:
        predictor = FutureFeaturePredictor(
            hidden_dim=16,
            predictor_hidden_dim=32,
            output_feature_dim=24,
            action_dim=8,
            head_num=4,
            layer_num=2,
            mlp_ratio=2.0,
            dropout=0.0,
            future_token_num=1,
            max_action_length=16,
        )
        current = torch.randn(2, 1, 16, requires_grad=True)
        actions = torch.randn(2, 10, 8, requires_grad=True)

        predicted = predictor(
            current_observation_tokens=current,
            action_chunk=actions,
        )

        self.assertEqual(predictor.input_feature_dim, 16)
        self.assertEqual(predictor.hidden_dim, 32)
        self.assertEqual(predicted.shape, (2, 1, 24))
        self.assertEqual(
            predictor.context_projector(current).shape,
            (2, 1, 32),
        )
        self.assertEqual(
            predictor.transformer.layers[0].self_attn.embed_dim,
            32,
        )
        predicted.mean().backward()
        self.assertIsNotNone(current.grad)
        self.assertIsNotNone(actions.grad)

    def test_default_width_keeps_parameter_free_identity_projection(self) -> None:
        predictor = make_predictor()

        self.assertEqual(predictor.input_feature_dim, predictor.hidden_dim)
        self.assertIsInstance(predictor.context_projector, torch.nn.Identity)

    def test_forward_accepts_unconstrained_token_layouts_and_backpropagates(
        self,
    ) -> None:
        predictor = make_predictor()
        current = torch.randn(2, 3, 32, requires_grad=True)
        memory = torch.randn(2, 6, 2, 32, requires_grad=True)
        actions = torch.randn(2, 10, 8, requires_grad=True)

        predicted = predictor(
            current_observation_tokens=current,
            memory_context_tokens=memory,
            action_chunk=actions,
        )

        self.assertEqual(predicted.shape, (2, 2, 24))
        self.assertTrue(torch.isfinite(predicted).all())

        predicted.square().mean().backward()
        self.assertIsNotNone(current.grad)
        self.assertIsNotNone(memory.grad)
        self.assertIsNotNone(actions.grad)

    def test_forward_supports_empty_memory_and_single_observation_token(
        self,
    ) -> None:
        predictor = make_predictor()
        current = torch.randn(2, 32)
        actions = torch.randn(2, 10, 8)

        predicted = predictor(
            current_observation_tokens=current,
            memory_context_tokens=None,
            action_chunk=actions,
        )

        self.assertEqual(predicted.shape, (2, 2, 24))

    def test_invalid_memory_tokens_are_ignored(self) -> None:
        predictor = make_predictor().eval()
        current = torch.randn(2, 32)
        actions = torch.randn(2, 10, 8)
        memory = torch.randn(2, 3, 2, 32)
        memory_mask = torch.ones(2, 3, 2, dtype=torch.bool)
        memory_mask[:, -1] = False

        modified_memory = memory.clone()
        modified_memory[:, -1] = 1_000.0

        with torch.no_grad():
            original_output = predictor(
                current_observation_tokens=current,
                memory_context_tokens=memory,
                memory_context_mask=memory_mask,
                action_chunk=actions,
            )
            modified_output = predictor(
                current_observation_tokens=current,
                memory_context_tokens=modified_memory,
                memory_context_mask=memory_mask,
                action_chunk=actions,
            )

        torch.testing.assert_close(original_output, modified_output)

    def test_cosine_loss_masks_targets_and_detaches_target_by_default(
        self,
    ) -> None:
        predictor = make_predictor()
        predicted = torch.randn(2, 2, 24, requires_grad=True)
        target = torch.randn(2, 2, 24, requires_grad=True)
        target_mask = torch.tensor([[True, False], [True, True]])

        loss = predictor.cosine_prediction_loss(
            predicted,
            target,
            target_mask=target_mask,
        )
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertIsNotNone(predicted.grad)
        self.assertIsNone(target.grad)
        self.assertEqual(torch.count_nonzero(predicted.grad[0, 1]), 0)

    def test_action_condition_can_be_disabled(self) -> None:
        predictor = make_predictor(use_action_condition=False)
        current = torch.randn(2, 4, 32)

        predicted = predictor(current_observation_tokens=current)
        self.assertEqual(predicted.shape, (2, 2, 24))

        with self.assertRaisesRegex(ValueError, "action input"):
            predictor(
                current_observation_tokens=current,
                action_chunk=torch.randn(2, 10, 8),
            )


if __name__ == "__main__":
    unittest.main()
