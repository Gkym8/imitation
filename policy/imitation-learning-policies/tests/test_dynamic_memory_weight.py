import unittest

import torch

from imitation_learning.models.memory.dynamic_memory_weight import (
    DynamicMemoryWeightUpdateConfig,
    apply_dynamic_memory_weight_update,
    select_memory_slot_eviction_index,
    select_random_training_slot_eviction_index,
)


class DynamicMemoryWeightTest(unittest.TestCase):
    def test_all_configuration_parameters_are_required(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "evict_lowest_weight_when_full"
        ):
            DynamicMemoryWeightUpdateConfig.from_mapping(
                {"enabled": True, "step_size": 0.05}
            )

    def test_old_configuration_defaults_to_no_recent_slot_protection(self) -> None:
        config = DynamicMemoryWeightUpdateConfig.from_mapping(
            {
                "enabled": True,
                "step_size": 100,
                "evict_lowest_weight_when_full": True,
            }
        )

        self.assertEqual(config.recent_slot_protection_num, 0)

    def test_explicit_update_stops_gradient_masks_and_clips(self) -> None:
        weights = torch.tensor([0.4, -0.4, 0.1], requires_grad=True)
        gradients = torch.tensor([-4.0, 4.0, 2.0], requires_grad=True)
        valid_mask = torch.tensor([True, True, False])

        updated, record = apply_dynamic_memory_weight_update(
            weights,
            gradients,
            valid_mask,
            step_size=0.1,
            weight_min=-0.5,
            weight_max=0.5,
        )

        torch.testing.assert_close(updated, torch.tensor([0.5, -0.5, 0.1]))
        self.assertFalse(updated.requires_grad)
        torch.testing.assert_close(record.gradient, torch.tensor([-4.0, 4.0]))
        torch.testing.assert_close(
            record.was_clipped, torch.tensor([True, True])
        )

    def test_minimum_weight_eviction_uses_oldest_tie_break(self) -> None:
        weights = [
            torch.tensor(0.2),
            torch.tensor(-0.3),
            torch.tensor(-0.3),
            torch.tensor(0.4),
        ]
        self.assertEqual(
            select_memory_slot_eviction_index(
                weights, evict_lowest_weight_when_full=True
            ),
            1,
        )

    def test_fifo_eviction_removes_oldest(self) -> None:
        weights = [torch.tensor(0.4), torch.tensor(-0.5)]
        self.assertEqual(
            select_memory_slot_eviction_index(
                weights, evict_lowest_weight_when_full=False
            ),
            0,
        )

    def test_recent_slots_are_excluded_from_weighted_eviction(self) -> None:
        weights = [
            torch.tensor(0.4),
            torch.tensor(-0.3),
            torch.tensor(0.2),
            torch.tensor(-0.5),
        ]

        self.assertEqual(
            select_memory_slot_eviction_index(
                weights,
                evict_lowest_weight_when_full=True,
                recent_slot_protection_num=2,
            ),
            1,
        )

    def test_recent_slot_protection_must_leave_an_eviction_candidate(self) -> None:
        weights = [torch.tensor(0.4), torch.tensor(-0.5)]
        with self.assertRaisesRegex(ValueError, "at least one slot"):
            select_memory_slot_eviction_index(
                weights,
                evict_lowest_weight_when_full=True,
                recent_slot_protection_num=2,
            )

    def test_random_training_eviction_only_selects_existing_slots(self) -> None:
        torch.manual_seed(7)
        selected = {
            select_random_training_slot_eviction_index(4)
            for _ in range(200)
        }

        self.assertEqual(selected, {0, 1, 2, 3})

    def test_random_training_eviction_rejects_empty_memory(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty training memory"):
            select_random_training_slot_eviction_index(0)


if __name__ == "__main__":
    unittest.main()
