import csv
import os
import tempfile
import unittest

import torch

from imitation_learning.models.memory.slot_weight_table import (
    build_active_slot_weight_table,
    write_active_slot_weight_csv,
)


class SlotWeightTableTest(unittest.TestCase):
    def test_build_and_write_only_active_retained_slots(self) -> None:
        table = build_active_slot_weight_table(
            slot_weights=torch.tensor(
                [[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4]]
            ),
            retained_source_slot_indices=torch.tensor(
                [[-1, 0, 2, 3], [-1, -1, 1, 3]]
            ),
            slot_frame_indices=torch.tensor(
                [[0, 10, 20, 30], [5, 15, 25, 35]]
            ),
        )

        torch.testing.assert_close(
            table["weight"],
            torch.tensor(
                [[0.1, 0.1, 0.3, 0.4], [-0.1, -0.1, -0.2, -0.4]]
            ),
        )
        torch.testing.assert_close(
            table["valid_mask"],
            torch.tensor(
                [[False, True, True, True], [False, False, True, True]]
            ),
        )

        record = {
            "global_step": 17,
            **{key: value.cpu() for key, value in table.items()},
            "rank": torch.tensor([0, 1]),
            "sample_index": torch.tensor([3, 4]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "epoch_0002.csv")
            write_active_slot_weight_csv(path, epoch=2, records=[record])
            with open(path, newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 5)
        self.assertEqual(
            list(rows[0]),
            [
                "epoch",
                "global_step",
                "rank",
                "sample_index",
                "buffer_position",
                "source_slot_index",
                "anchor_frame_index",
                "weight",
            ],
        )
        self.assertEqual(
            {
                key: rows[0][key]
                for key in (
                    "epoch",
                    "global_step",
                    "rank",
                    "sample_index",
                    "buffer_position",
                    "source_slot_index",
                    "anchor_frame_index",
                )
            },
            {
                "epoch": "2",
                "global_step": "17",
                "rank": "0",
                "sample_index": "3",
                "buffer_position": "1",
                "source_slot_index": "0",
                "anchor_frame_index": "0",
            },
        )
        self.assertAlmostEqual(float(rows[0]["weight"]), 0.1)
        self.assertEqual(rows[-1]["source_slot_index"], "3")
        self.assertEqual(rows[-1]["anchor_frame_index"], "35")
        self.assertAlmostEqual(float(rows[-1]["weight"]), -0.4)

    def test_rejects_out_of_range_source_slot(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_active_slot_weight_table(
                slot_weights=torch.zeros(1, 2),
                retained_source_slot_indices=torch.tensor([[0, 2]]),
                slot_frame_indices=torch.zeros(1, 2, dtype=torch.long),
            )


if __name__ == "__main__":
    unittest.main()
