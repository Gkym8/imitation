import unittest

import torch

from imitation_learning.models.future_prediction.future_feature_predictor import (
    FutureFeaturePredictor,
)


def make_predictor(use_action_condition: bool = True) -> FutureFeaturePredictor:
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
    )


class FutureFeaturePredictorTest(unittest.TestCase):
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
