from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

from rmbench_model import RMBenchMemoryPolicy


class _FakePolicy:
    def __init__(self, snapshot, attention):
        self.latest_online_memory_query_dict = {0: snapshot}
        self.recorded_data_dicts = {
            0: [{"history_cross_attention_slot_weights": attention}]
        }


class RMBenchSlotGateRecordingTest(unittest.TestCase):
    def make_adapter(self, record_path: Path, snapshot) -> RMBenchMemoryPolicy:
        count = len(snapshot["slot_ids"])
        snapshot.setdefault("query_anchor_index", 3)
        snapshot.setdefault("slot_states", ("active",) * count)
        snapshot.setdefault("slot_expire_at", (None,) * count)
        snapshot.setdefault("slot_effective_gates", snapshot["slot_gates"])
        snapshot.setdefault("active_slot_count", snapshot["slot_states"].count("active"))
        snapshot.setdefault("shadow_slot_count", snapshot["slot_states"].count("shadow"))
        adapter = RMBenchMemoryPolicy.__new__(RMBenchMemoryPolicy)
        adapter.slot_gate_record_path = record_path
        adapter._gate_head_num = 2
        adapter._eval_episode_index = 2
        adapter._memory_attention_record_key = (
            "history_cross_attention_slot_weights"
        )
        # Include padding and distinct head/query weights to verify that only
        # real slots are written and their attention averages all heads/queries.
        attention = torch.empty(1, 1, 0) if count == 0 else torch.tensor(
            [[[[[0.0, 0.0, 1.0], [0.0, 0.75, 0.25]],
               [[0.0, 0.5, 0.5], [0.0, 0.25, 0.75]]]]],
            dtype=torch.float32,
        )
        adapter.policy = _FakePolicy(snapshot, attention)
        adapter._initialize_slot_gate_recording()
        return adapter

    def test_records_one_row_per_active_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory_slot_gates.csv"
            adapter = self.make_adapter(
                path,
                {
                    "query_frame_index": 30,
                    "slot_ids": (0, 2),
                    "slot_frame_indices": (0, 20),
                    "slot_gates": ((0.2, 0.6), (0.1, 0.3)),
                },
            )

            adapter._record_memory_query()

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["episode_index"], "2")
            self.assertEqual(list(rows[0]), [
                "episode_index", "query_anchor_index", "slot_id", "slot_state",
                "gate_mean", "gate_head_0", "gate_head_1", "attention_mean",
            ])
            self.assertEqual(rows[0]["query_anchor_index"], "3")
            self.assertEqual(rows[0]["slot_id"], "0")
            self.assertEqual(rows[0]["gate_mean"], "0.4")
            self.assertEqual(rows[0]["gate_head_0"], "0.2")
            self.assertEqual(rows[0]["gate_head_1"], "0.6")
            self.assertEqual(rows[0]["slot_state"], "active")
            self.assertEqual(rows[0]["attention_mean"], "0.375")
            self.assertEqual(rows[1]["slot_id"], "2")
            self.assertEqual(rows[1]["gate_mean"], "0.2")
            self.assertEqual(rows[1]["attention_mean"], "0.625")

            adapter.policy.reset = Mock()
            adapter.observation_history = {}
            adapter.reset_model()
            adapter._record_memory_query()
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["episode_index"] for row in rows], ["2", "2", "3", "3"])
            self.assertTrue(all(None not in row and None not in row.values() for row in rows))

    def test_records_empty_first_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory_slot_gates.csv"
            adapter = self.make_adapter(
                path,
                {
                    "query_frame_index": 0,
                    "slot_ids": (),
                    "slot_frame_indices": (),
                    "slot_gates": (),
                },
            )

            adapter._record_memory_query()

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["episode_index"], "2")
            self.assertEqual(rows[0]["query_anchor_index"], "3")
            self.assertNotIn(None, rows[0])
            self.assertTrue(all(
                value == "" for key, value in rows[0].items()
                if key not in {"episode_index", "query_anchor_index"}
            ))

    def test_records_shadow_state_and_raw_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory_slot_gates.csv"
            adapter = self.make_adapter(path, {
                "query_frame_index": 40, "query_anchor_index": 4,
                "slot_ids": (0, 2), "slot_frame_indices": (0, 20),
                "slot_states": ("active", "shadow"), "slot_expire_at": (None, 5),
                "slot_gates": ((0.2, 0.6), (0.1, 0.3)),
                "slot_effective_gates": ((0.2, 0.6), (0.001, 0.003)),
            })
            adapter._record_memory_query()
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[1]["query_anchor_index"], "4")
            self.assertEqual(rows[1]["slot_state"], "shadow")
            self.assertEqual(rows[1]["gate_mean"], "0.2")
            self.assertEqual(rows[1]["gate_head_0"], "0.1")
            self.assertEqual(rows[1]["gate_head_1"], "0.3")
            self.assertEqual(rows[1]["attention_mean"], "0.625")


if __name__ == "__main__":
    unittest.main()
