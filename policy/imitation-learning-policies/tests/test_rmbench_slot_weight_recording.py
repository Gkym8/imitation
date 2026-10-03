from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from rmbench_model import RMBenchMemoryPolicy


class _FakePolicy:
    def __init__(self, snapshot, attention):
        self.latest_online_memory_query_dict = {0: snapshot}
        self.recorded_data_dicts = {
            0: [{"history_cross_attention_slot_weights": attention}]
        }


class RMBenchSlotWeightRecordingTest(unittest.TestCase):
    def make_adapter(self, record_path: Path, snapshot) -> RMBenchMemoryPolicy:
        adapter = RMBenchMemoryPolicy.__new__(RMBenchMemoryPolicy)
        adapter.slot_weight_record_path = record_path
        adapter._eval_episode_index = 2
        adapter._eval_decision_index = 3
        adapter._memory_attention_record_key = (
            "history_cross_attention_slot_weights"
        )
        active_slot_count = len(snapshot["slot_ids"])
        attention = torch.empty(1, 1, 0) if active_slot_count == 0 else torch.tensor(
            [[[[[0.25, 0.75], [0.50, 0.50]]]]], dtype=torch.float32
        )
        adapter.policy = _FakePolicy(snapshot, attention)
        adapter._initialize_slot_weight_recording()
        return adapter

    def test_records_one_row_per_active_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory_slot_weights.csv"
            adapter = self.make_adapter(
                path,
                {
                    "query_frame_index": 30,
                    "slot_ids": (0, 2),
                    "slot_frame_indices": (0, 20),
                    "slot_weights": (0.4, -0.125),
                },
            )

            adapter._record_memory_query()

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["episode_index"], "2")
            self.assertEqual(rows[0]["decision_index"], "3")
            self.assertEqual(rows[0]["query_frame_index"], "30")
            self.assertEqual(rows[0]["slot_id"], "0")
            self.assertEqual(rows[0]["slot_frame_index"], "0")
            self.assertEqual(rows[0]["slot_age_frames"], "30")
            self.assertEqual(rows[0]["slot_weight"], "0.4")
            self.assertEqual(rows[0]["attention_mean"], "0.375")
            self.assertEqual(rows[0]["attention_std"], "0.125")
            self.assertEqual(rows[0]["attention_max"], "0.5")
            self.assertEqual(rows[1]["slot_id"], "2")
            self.assertEqual(rows[1]["slot_frame_index"], "20")
            self.assertEqual(rows[1]["slot_age_frames"], "10")
            self.assertEqual(rows[1]["slot_weight"], "-0.125")
            self.assertEqual(rows[1]["attention_mean"], "0.625")
            self.assertEqual(rows[1]["attention_std"], "0.125")
            self.assertEqual(rows[1]["attention_max"], "0.75")

    def test_records_empty_first_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory_slot_weights.csv"
            adapter = self.make_adapter(
                path,
                {
                    "query_frame_index": 0,
                    "slot_ids": (),
                    "slot_frame_indices": (),
                    "slot_weights": (),
                },
            )

            adapter._record_memory_query()

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["active_slot_count"], "0")
            self.assertEqual(rows[0]["slot_id"], "")


if __name__ == "__main__":
    unittest.main()
