from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

from imitation_learning.datasets.multi_traj_dataset import MultiTrajDataset
from imitation_learning.datasets.rmbench_dataset import RMBenchMultiTrajDataset
from test_rmbench_cached_features import FEATURE_DIM, MODEL_NAME, make_meta


class StreamingSamplingTest(unittest.TestCase):
    @contextmanager
    def dataset(self, lengths=(43,), history_offset=0, **overrides):
        """Frame-valued cache exercises the real sampler and loaded samples."""
        with tempfile.TemporaryDirectory(prefix="rmbench-streaming-sampling-") as directory:
            root = zarr.open_group(str(Path(directory) / "test.zarr"), mode="w")
            for episode_index, length in enumerate(lengths):
                episode = root.create_group(f"episode_{episode_index}")
                frames = np.arange(length, dtype=np.float32) + 1000 * episode_index
                features = episode.create_dataset(
                    "head_camera_feature",
                    data=np.broadcast_to(frames[:, None], (length, FEATURE_DIM)).copy(),
                )
                features.attrs["complete"] = True
                for key in ("agent_pos", "action"):
                    episode.create_dataset(key, data=np.zeros((length, 2), dtype=np.float32))
            root.attrs["episode_frame_nums"] = {str(i): length for i, length in enumerate(lengths)}
            root.attrs["dinov2_feature_cache"] = {
                "model_name": MODEL_NAME, "input_size": 224, "feature_kind": "cls",
                "feature_dim": FEATURE_DIM, "feature_dtype": "float32", "camera_names": ["head_camera"],
            }
            source_meta, output_meta = make_meta()
            source_meta["head_camera"]["include_indices"] = [-history_offset]
            kwargs = dict(
                root_dir=directory, name="test", compressed_dir=directory, robot_num=2,
                include_episode_num=-1, include_episode_indices=[], used_episode_ratio=1.0,
                random_split_dataset=False, index_pool_size_per_episode=2,
                history_padding_length=0, future_padding_length=0, seed=0,
                source_data_meta=source_meta, output_data_meta=output_meta,
                dataloader_cfg={"batch_size": 1}, starting_percentile_max=1.0,
                starting_percentile_min=0.0, apply_image_augmentation_in_cpu=True,
                use_relative_pose=False, use_relative_gripper_width=False,
                normalizer_sample_num=-1, normalizer_dir="", repeat_dataset_num=1,
                down_sample_steps=1, statistics_data_path="", image_keys=["head_camera"],
                use_cached_image_features=True, cached_image_feature_dim=FEATURE_DIM,
                cached_image_feature_model_name=MODEL_NAME, cached_image_feature_input_size=224,
                cached_image_feature_dtype="float32", cached_image_feature_kind="cls",
                traj_num=3, traj_interval_min=10, traj_interval_max=10,
                anchor_sampling_mode="streaming", split_dataloader_cfg=None,
                episode_starting_idx_max=None,
            )
            kwargs.update(overrides)
            yield RMBenchMultiTrajDataset(**kwargs)

    def test_five_selects_six_starts_and_keeps_full_streams_and_tail_padding(self):
        with self.dataset(sample_range=[0, 5]) as dataset:
            self.assertEqual(dataset.index_pool, [(0, i) for i in range(6)])
            self.assertEqual(dataset.traj_num, 5)
            for index, (_, start) in enumerate(dataset.index_pool):
                sample = dataset[index]
                valid = ~sample["entire_traj_is_padding"]
                actual = sample["head_camera_feature"][valid, 0, 0]
                expected = torch.tensor(list(range(start, 42, 10)), dtype=actual.dtype)
                torch.testing.assert_close(actual, expected)
                self.assertTrue(bool(sample["streaming_sequence"]))
            self.assertFalse(bool(dataset[0]["entire_traj_is_padding"].any()))
            self.assertTrue(bool(dataset[5]["entire_traj_is_padding"][-1]))
            self.assertEqual(float(dataset[0]["head_camera_feature"][-1, 0, 0]), 40.0)
            dataset.resample_index_pool()
            self.assertEqual(dataset.index_pool, [(0, i) for i in range(6)])

    def test_zero_always_starts_at_original_first_frame_and_reaches_episode_end(self):
        with self.dataset(sample_range=[0, 0]) as dataset:
            self.assertEqual(dataset.index_pool, [(0, 0)])
            sample = dataset[0]
            torch.testing.assert_close(
                sample["head_camera_feature"][:, 0, 0], torch.tensor([0., 10., 20., 30., 40.]),
            )

    def test_omitted_and_null_keep_original_phase_count(self):
        for kwargs in ({}, {"sample_range": None}):
            with self.subTest(kwargs=kwargs), self.dataset(**kwargs) as dataset:
                self.assertEqual(dataset.index_pool, [(0, i) for i in range(10)])
                self.assertEqual(dataset.traj_num, 5)

    def test_explicit_range_exceeds_interval_and_selects_all_requested_starts(self):
        with self.dataset(sample_range=[0, 40]) as dataset:
            self.assertEqual(dataset.index_pool, [(0, i) for i in range(41)])
            self.assertEqual(dataset.traj_num, 5)
            for index, (_, start) in enumerate(dataset.index_pool):
                sample = dataset[index]
                valid = ~sample["entire_traj_is_padding"]
                actual = sample["head_camera_feature"][valid, 0, 0]
                expected = torch.tensor(list(range(start, 42, 10)), dtype=actual.dtype)
                torch.testing.assert_close(actual, expected)

    def test_range_stops_at_last_valid_start_instead_of_creating_empty_streams(self):
        with self.dataset(sample_range=[0, 99]) as dataset:
            self.assertEqual(dataset.index_pool, [(0, i) for i in range(42)])
            last = dataset[len(dataset) - 1]
            self.assertFalse(bool(last["entire_traj_is_padding"][0]))
            self.assertTrue(bool(last["entire_traj_is_padding"][1:].all()))
            self.assertEqual(float(last["head_camera_feature"][0, 0, 0]), 41.0)

    def test_short_episodes_and_repetitions_use_only_valid_selected_starts(self):
        with self.dataset(lengths=(43, 4), sample_range=[0, 5], repeat_dataset_num=2) as dataset:
            counts = Counter(dataset.index_pool)
            self.assertEqual(set(counts), {(0, i) for i in range(6)} | {(1, i) for i in range(3)})
            self.assertEqual(set(counts.values()), {2})
            for index, (episode, start) in enumerate(dataset.index_pool):
                sample = dataset[index]
                self.assertFalse(bool(sample["entire_traj_is_padding"][0]))
                self.assertEqual(float(sample["head_camera_feature"][0, 0, 0]), 1000 * episode + start)
                if episode == 1:
                    self.assertTrue(bool(sample["entire_traj_is_padding"][1:].all()))

    def test_range_uses_absolute_frame_indices_and_respects_first_valid_frame(self):
        with self.dataset(history_offset=2, sample_range=[0, 5]) as dataset:
            self.assertEqual(dataset.index_pool, [(0, i) for i in range(2, 6)])
        with self.assertRaisesRegex(ValueError, "no valid anchors.*sample_range"):
            with self.dataset(history_offset=2, sample_range=[0, 1]):
                pass

    def test_late_range_loads_only_selected_start_frames_and_continues_to_end(self):
        with self.dataset(lengths=(83,), sample_range=[30, 40]) as dataset:
            self.assertEqual(dataset.index_pool, [(0, i) for i in range(30, 41)])
            self.assertEqual(len(dataset), 11)
            for index, (_, start) in enumerate(dataset.index_pool):
                sample = dataset[index]
                valid = ~sample["entire_traj_is_padding"]
                actual = sample["head_camera_feature"][valid, 0, 0]
                expected = torch.tensor(list(range(start, 82, 10)), dtype=actual.dtype)
                torch.testing.assert_close(actual, expected)
                self.assertTrue(bool((actual >= 30).all()))
            self.assertEqual(float(dataset[0]["head_camera_feature"][-1, 0, 0]), 80.0)

    def test_short_episodes_without_overlap_are_skipped_and_empty_dataset_is_rejected(self):
        with self.dataset(lengths=(83, 20), sample_range=[30, 40]) as dataset:
            self.assertEqual(dataset.index_pool, [(0, i) for i in range(30, 41)])
        with self.assertRaisesRegex(ValueError, "no valid anchors.*sample_range"):
            with self.dataset(lengths=(20,), sample_range=[30, 40]):
                pass

    def test_yaml_list_configuration_is_accepted(self):
        with self.dataset(lengths=(83,), sample_range=OmegaConf.create([30, 40])) as dataset:
            self.assertEqual(dataset.index_pool, [(0, i) for i in range(30, 41)])

    def test_legacy_sample_nums_maps_to_zero_based_range_for_saved_configs(self):
        for value in (0, 5, 40, None):
            with self.subTest(value=value), self.dataset(sample_nums=value) as dataset:
                expected = range(10 if value is None else value + 1)
                self.assertEqual(dataset.index_pool, [(0, i) for i in expected])
        with self.assertRaisesRegex(ValueError, "legacy setting"):
            with self.dataset(sample_nums=40, sample_range=[30, 40]):
                pass

    def test_invalid_ranges_and_non_streaming_usage_fail_before_loading_data(self):
        for value in (-1, True, 1.5, "30,40", [], [30], [30, 40, 50],
                      [-1, 40], [40, 30], [False, 40], [30, 40.5]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "sample_range"):
                MultiTrajDataset(traj_num=3, traj_interval_min=10, traj_interval_max=10,
                                 anchor_sampling_mode="streaming", sample_range=value)
        with self.assertRaisesRegex(ValueError, "only supported for streaming"):
            MultiTrajDataset(traj_num=3, traj_interval_min=10, traj_interval_max=10,
                             anchor_sampling_mode="contiguous", sample_range=[30, 40])
        for value in (-1, True, 1.5, "5"):
            with self.subTest(legacy=value), self.assertRaisesRegex(ValueError, "sample_nums"):
                MultiTrajDataset(traj_num=3, traj_interval_min=10, traj_interval_max=10,
                                 anchor_sampling_mode="streaming", sample_nums=value)


if __name__ == "__main__":
    unittest.main()
