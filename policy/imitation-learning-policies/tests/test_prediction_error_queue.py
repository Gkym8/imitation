import unittest

import torch

from imitation_learning.models.memory.prediction_error_queue import (
    PredictionErrorQueue,
    PredictionErrorQueueConfig,
    apply_recent_slot_attention_floor,
)


def make_config(**overrides) -> PredictionErrorQueueConfig:
    values = {
        "enabled": True,
        "capacity": 3,
        "warmup_min_samples": 2,
        "warmup_min_std": 0.05,
        "epsilon": 1.0e-6,
        "weight_min": -0.5,
        "weight_max": 0.5,
        "first_slot_max_frame_distance": 5,
        "first_slot_weight": 0.25,
        "first_slot_frozen": True,
        "recent_slot_attention_floor": 0.2,
        "record_training_stats": True,
        "record_slot_weights": True,
    }
    values.update(overrides)
    return PredictionErrorQueueConfig(**values)

class PredictionErrorQueueTest(unittest.TestCase):
    def test_all_configuration_parameters_are_required(self) -> None:
        config = {
            "enabled": True,
            "capacity": 3,
            "warmup_min_samples": 2,
            "warmup_min_std": 0.05,
            "epsilon": 1.0e-6,
            "weight_min": -0.5,
            "weight_max": 0.5,
        }
        with self.assertRaisesRegex(ValueError, "record_training_stats"):
            PredictionErrorQueueConfig.from_mapping(config)

    def test_first_slot_weight_and_first_error_based_second_slot_weight(
        self,
    ) -> None:
        queue = PredictionErrorQueue(make_config())
        reference = torch.tensor(0.0)

        # Omitting an index is the online-inference path at the true episode start.
        self.assertEqual(float(queue.first_slot_weight(reference)), 0.25)
        frame_indices = torch.tensor([0, 5, 6])
        torch.testing.assert_close(
            queue.first_slot_weight(torch.zeros(3), frame_indices),
            torch.tensor([0.25, 0.25, 0.0]),
        )
        record = queue.score_and_push(torch.tensor(0.025))

        self.assertEqual(int(record.queue_size_before), 0)
        torch.testing.assert_close(record.queue_mean, torch.tensor(0.0))
        torch.testing.assert_close(record.queue_std, torch.tensor(0.0))
        torch.testing.assert_close(record.effective_std, torch.tensor(0.05))
        torch.testing.assert_close(record.z_score, torch.tensor(0.5))
        expected_weight = 0.5 * (2.0 * torch.sigmoid(torch.tensor(0.5)) - 1.0)
        torch.testing.assert_close(record.initial_weight, expected_weight)
        self.assertEqual(len(queue), 1)

    def test_new_configuration_fields_have_checkpoint_compatible_defaults(
        self,
    ) -> None:
        values = make_config().__dict__.copy()
        values.pop("first_slot_weight")
        values.pop("first_slot_frozen")
        values.pop("recent_slot_attention_floor")
        values.pop("record_slot_weights")

        config = PredictionErrorQueueConfig.from_mapping(values)

        self.assertEqual(config.first_slot_weight, config.weight_max)
        self.assertTrue(config.first_slot_frozen)
        self.assertIsNone(config.recent_slot_attention_floor)
        self.assertFalse(config.record_slot_weights)

    def test_first_slot_frozen_must_be_boolean(self) -> None:
        with self.assertRaisesRegex(ValueError, "first_slot_frozen"):
            make_config(first_slot_frozen=1)

    def test_first_slot_weight_must_be_inside_weight_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "first_slot_weight"):
            make_config(first_slot_weight=0.6)

    def test_recent_attention_floor_must_be_inside_weight_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "recent_slot_attention_floor"):
            make_config(recent_slot_attention_floor=0.6)

    def test_recent_attention_floor_only_changes_query_copy(self) -> None:
        stored = torch.tensor([0.4, 0.25, -0.1, 0.45, 0.03, -0.2])
        original = stored.clone()

        attention = apply_recent_slot_attention_floor(
            stored,
            attention_bias_mode="standardized",
            recent_slot_protection_num=2,
            recent_slot_attention_floor=0.4,
            fixed_slot_mask=torch.tensor(
                [True, False, False, False, False, False]
            ),
        )

        torch.testing.assert_close(stored, original)
        torch.testing.assert_close(
            attention,
            torch.tensor([0.4, 0.25, -0.1, 0.45, 0.4, 0.4]),
        )

    def test_fixed_first_slot_takes_precedence_in_short_history(self) -> None:
        attention = apply_recent_slot_attention_floor(
            torch.tensor([0.25]),
            attention_bias_mode="standardized",
            recent_slot_protection_num=2,
            recent_slot_attention_floor=0.4,
            fixed_slot_mask=torch.tensor([True]),
        )

        torch.testing.assert_close(attention, torch.tensor([0.25]))

    def test_recent_attention_floor_is_disabled_in_additive_mode(self) -> None:
        stored = torch.tensor([0.4, -0.2, 0.1])

        attention = apply_recent_slot_attention_floor(
            stored,
            attention_bias_mode="additive",
            recent_slot_protection_num=2,
            recent_slot_attention_floor=0.5,
        )

        torch.testing.assert_close(attention, stored)
        self.assertIsNot(attention, stored)

    def test_warmup_uses_minimum_standard_deviation(self) -> None:
        queue = PredictionErrorQueue(make_config())
        queue.score_and_push(torch.tensor(0.7))
        record = queue.score_and_push(torch.tensor(0.8))

        torch.testing.assert_close(record.queue_std, torch.tensor(0.0))
        torch.testing.assert_close(record.effective_std, torch.tensor(0.05))
        torch.testing.assert_close(record.z_score, torch.tensor(2.0))
        expected_weight = 0.5 * (2.0 * torch.sigmoid(torch.tensor(2.0)) - 1.0)
        torch.testing.assert_close(record.initial_weight, expected_weight)

    def test_after_warmup_uses_observed_standard_deviation(self) -> None:
        queue = PredictionErrorQueue(make_config())
        queue.score_and_push(torch.tensor(0.0))
        queue.score_and_push(torch.tensor(2.0))
        record = queue.score_and_push(torch.tensor(3.0))

        torch.testing.assert_close(record.queue_mean, torch.tensor(1.0))
        torch.testing.assert_close(record.queue_std, torch.tensor(1.0))
        torch.testing.assert_close(
            record.effective_std, torch.tensor(1.0 + 1.0e-6)
        )

    def test_full_queue_evicts_oldest_error(self) -> None:
        queue = PredictionErrorQueue(make_config())
        for value in [1.0, 2.0, 3.0, 4.0]:
            queue.score_and_push(torch.tensor(value))

        self.assertEqual(len(queue), 3)
        torch.testing.assert_close(
            torch.stack(queue.values), torch.tensor([2.0, 3.0, 4.0])
        )


if __name__ == "__main__":
    unittest.main()
