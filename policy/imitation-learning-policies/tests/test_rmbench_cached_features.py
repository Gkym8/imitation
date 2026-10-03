from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import zarr

from imitation_learning.datasets.rmbench_dataset import (
    CAMERA_NAMES,
    RMBenchMultiTrajDataset,
    _prepare_kwargs,
)
from imitation_learning.models.encoders.multi_token_encoder import (
    MultiTokenEncoder,
)


FEATURE_DIM = 8
PATCH_TOKEN_NUM = 4
MODEL_NAME = "test/dinov2-with-registers"


class _CachedImageEncoder(nn.Module):
    def __init__(self, aggregation: str, token_num: int) -> None:
        super().__init__()
        self.feature_aggregation = aggregation
        self.token_num = token_num
        self.patch_token_num = PATCH_TOKEN_NUM


def make_cached_multi_token_encoder(
    aggregation: str, token_num: int
) -> MultiTokenEncoder:
    encoder = MultiTokenEncoder.__new__(MultiTokenEncoder)
    nn.Module.__init__(encoder)
    encoder.cond_meta = {
        "head_camera": SimpleNamespace(name="head_camera", length=1)
    }
    encoder.encoder_dict = nn.ModuleDict(
        {"head_camera": _CachedImageEncoder(aggregation, token_num)}
    )
    encoder.feature_dim = FEATURE_DIM
    encoder.token_num = token_num
    return encoder


def make_meta() -> tuple[dict, dict]:
    source = {
        camera: {"include_indices": [0], "shape": [240, 320, 3]}
        for camera in CAMERA_NAMES
    }
    source.update(
        {
            "agent_pos": {"include_indices": [0], "shape": [2]},
            "action": {"include_indices": [0, 1], "shape": [2]},
        }
    )
    output = {
        camera: {
            "data_type": "image",
            "length": 1,
            "normalizer": "identity",
            "augmentation": [
                {"name": "Resize", "size": [224, 224], "antialias": True}
            ],
            "shape": [3, 224, 224],
            "source_entry_names": [camera],
        }
        for camera in CAMERA_NAMES
    }
    output.update(
        {
            "agent_pos": {
                "data_type": "low_dim",
                "length": 1,
                "normalizer": "range",
                "augmentation": [],
                "shape": [2],
                "source_entry_names": ["agent_pos"],
            },
            "action": {
                "data_type": "low_dim",
                "length": 2,
                "normalizer": "range",
                "augmentation": [],
                "shape": [2],
                "source_entry_names": ["action"],
            },
        }
    )
    return source, output


class RMBenchCachedFeaturesTest(unittest.TestCase):
    def test_cls_plus_patch_cache_serves_both_encoder_aggregations(self) -> None:
        cached = torch.arange(
            2 * 3 * (1 + PATCH_TOKEN_NUM) * FEATURE_DIM,
            dtype=torch.float32,
        ).reshape(2, 3, 1, 1 + PATCH_TOKEN_NUM, FEATURE_DIM)

        patch_encoder = make_cached_multi_token_encoder(
            "patches", PATCH_TOKEN_NUM
        )
        patch_features = patch_encoder(
            {"head_camera_feature": cached}
        )
        self.assertEqual(
            tuple(patch_features.shape),
            (2, 3, PATCH_TOKEN_NUM, FEATURE_DIM),
        )
        torch.testing.assert_close(
            patch_features, cached[..., 1:, :].squeeze(2)
        )

        cls_encoder = make_cached_multi_token_encoder("map", 1)
        cls_features = cls_encoder({"head_camera_feature": cached})
        self.assertEqual(
            tuple(cls_features.shape), (2, 3, 1, FEATURE_DIM)
        )
        torch.testing.assert_close(cls_features, cached[..., 0, :])

    def test_metadata_replacement_is_dataset_local(self) -> None:
        source, output = make_meta()
        original_source = copy.deepcopy(source)
        original_output = copy.deepcopy(output)

        kwargs = _prepare_kwargs(
            list(CAMERA_NAMES),
            source,
            output,
            {},
            use_cached_image_features=True,
            cached_image_feature_dim=FEATURE_DIM,
        )

        self.assertEqual(source, original_source)
        self.assertEqual(output, original_output)
        for camera in CAMERA_NAMES:
            self.assertNotIn(camera, kwargs["source_data_meta"])
            self.assertNotIn(camera, kwargs["output_data_meta"])
            feature_name = f"{camera}_feature"
            self.assertEqual(
                kwargs["source_data_meta"][feature_name]["shape"],
                [FEATURE_DIM],
            )
            self.assertEqual(
                kwargs["output_data_meta"][feature_name]["data_type"],
                "low_dim",
            )

    def test_single_view_cache_uses_cls_plus_patch_storage(self) -> None:
        source, output = make_meta()
        kwargs = _prepare_kwargs(
            ["head_camera"],
            source,
            output,
            {},
            use_cached_image_features=True,
            cached_image_feature_dim=FEATURE_DIM,
            cached_image_feature_kind="auto",
            cached_image_feature_patch_token_num=PATCH_TOKEN_NUM,
        )

        self.assertNotIn("head_camera", kwargs["source_data_meta"])
        self.assertEqual(
            kwargs["source_data_meta"]["head_camera_patch_feature"]["shape"],
            [1 + PATCH_TOKEN_NUM, FEATURE_DIM],
        )
        self.assertEqual(
            kwargs["output_data_meta"]["head_camera_feature"]["shape"],
            [1 + PATCH_TOKEN_NUM, FEATURE_DIM],
        )
        self.assertEqual(
            kwargs["output_data_meta"]["head_camera_feature"]
            ["source_entry_names"],
            ["head_camera_patch_feature"],
        )

    def test_multitraj_dataset_returns_features_without_raw_images(self) -> None:
        source_meta, output_meta = make_meta()
        with tempfile.TemporaryDirectory(prefix="rmbench-cache-dataset-") as temp_dir:
            dataset_path = Path(temp_dir) / "test.zarr"
            root = zarr.open_group(str(dataset_path), mode="w")
            episode_length = 20
            episode = root.create_group("episode_0")
            for camera_index, camera in enumerate(CAMERA_NAMES):
                values = np.full(
                    (episode_length, FEATURE_DIM),
                    camera_index + 1,
                    dtype=np.float32,
                )
                features = episode.create_dataset(
                    f"{camera}_feature",
                    data=values,
                    chunks=(5, FEATURE_DIM),
                )
                features.attrs["complete"] = True
            episode.create_dataset(
                "agent_pos",
                data=np.zeros((episode_length, 2), dtype=np.float32),
            )
            episode.create_dataset(
                "action",
                data=np.zeros((episode_length, 2), dtype=np.float32),
            )
            root.attrs["episode_frame_nums"] = {"0": episode_length}
            root.attrs["dinov2_feature_cache"] = {
                "model_name": MODEL_NAME,
                "input_size": 224,
                "feature_kind": "cls",
                "feature_dim": FEATURE_DIM,
                "feature_dtype": "float32",
                "camera_names": list(CAMERA_NAMES),
            }

            dataset = RMBenchMultiTrajDataset(
                root_dir=temp_dir,
                name="test",
                compressed_dir=temp_dir,
                robot_num=2,
                include_episode_num=-1,
                include_episode_indices=[],
                used_episode_ratio=1.0,
                random_split_dataset=False,
                index_pool_size_per_episode=2,
                history_padding_length=0,
                future_padding_length=0,
                seed=0,
                source_data_meta=source_meta,
                output_data_meta=output_meta,
                dataloader_cfg={"batch_size": 1},
                starting_percentile_max=1.0,
                starting_percentile_min=0.0,
                apply_image_augmentation_in_cpu=True,
                use_relative_pose=False,
                use_relative_gripper_width=False,
                normalizer_sample_num=-1,
                normalizer_dir="",
                repeat_dataset_num=1,
                down_sample_steps=1,
                statistics_data_path="",
                image_keys=list(CAMERA_NAMES),
                use_cached_image_features=True,
                cached_image_feature_dim=FEATURE_DIM,
                cached_image_feature_model_name=MODEL_NAME,
                cached_image_feature_input_size=224,
                cached_image_feature_dtype="float32",
                traj_num=3,
                traj_interval_min=2,
                traj_interval_max=2,
                split_dataloader_cfg=None,
                episode_starting_idx_max=None,
            )
            sample = dataset[0]
            for camera_index, camera in enumerate(CAMERA_NAMES):
                self.assertNotIn(camera, sample)
                feature = sample[f"{camera}_feature"]
                self.assertEqual(tuple(feature.shape), (3, 1, FEATURE_DIM))
                np.testing.assert_allclose(
                    feature.numpy(), camera_index + 1
                )

    def test_single_view_dataset_returns_cls_plus_patch_cache(self) -> None:
        source_meta, output_meta = make_meta()
        with tempfile.TemporaryDirectory(
            prefix="rmbench-patch-cache-dataset-"
        ) as temp_dir:
            dataset_path = Path(temp_dir) / "test.zarr"
            root = zarr.open_group(str(dataset_path), mode="w")
            episode_length = 20
            episode = root.create_group("episode_0")
            values = np.arange(
                episode_length * (1 + PATCH_TOKEN_NUM) * FEATURE_DIM,
                dtype=np.float32,
            ).reshape(episode_length, 1 + PATCH_TOKEN_NUM, FEATURE_DIM)
            features = episode.create_dataset(
                "head_camera_patch_feature",
                data=values,
                chunks=(5, 1 + PATCH_TOKEN_NUM, FEATURE_DIM),
            )
            features.attrs["complete"] = True
            episode.create_dataset(
                "agent_pos",
                data=np.zeros((episode_length, 2), dtype=np.float32),
            )
            episode.create_dataset(
                "action",
                data=np.zeros((episode_length, 2), dtype=np.float32),
            )
            root.attrs["episode_frame_nums"] = {"0": episode_length}
            root.attrs["dinov2_patch_feature_cache"] = {
                "model_name": MODEL_NAME,
                "input_size": 224,
                "feature_kind": "patches",
                "feature_dim": FEATURE_DIM,
                "patch_token_num": PATCH_TOKEN_NUM,
                "feature_dtype": "float32",
                "camera_names": ["head_camera"],
            }

            dataset = RMBenchMultiTrajDataset(
                root_dir=temp_dir,
                name="test",
                compressed_dir=temp_dir,
                robot_num=2,
                include_episode_num=-1,
                include_episode_indices=[],
                used_episode_ratio=1.0,
                random_split_dataset=False,
                index_pool_size_per_episode=2,
                history_padding_length=0,
                future_padding_length=0,
                seed=0,
                source_data_meta=source_meta,
                output_data_meta=output_meta,
                dataloader_cfg={"batch_size": 1},
                starting_percentile_max=1.0,
                starting_percentile_min=0.0,
                apply_image_augmentation_in_cpu=True,
                use_relative_pose=False,
                use_relative_gripper_width=False,
                normalizer_sample_num=-1,
                normalizer_dir="",
                repeat_dataset_num=1,
                down_sample_steps=1,
                statistics_data_path="",
                image_keys=["head_camera"],
                use_cached_image_features=True,
                cached_image_feature_dim=FEATURE_DIM,
                cached_image_feature_model_name=MODEL_NAME,
                cached_image_feature_input_size=224,
                cached_image_feature_dtype="float32",
                cached_image_feature_kind="auto",
                cached_image_feature_patch_token_num=PATCH_TOKEN_NUM,
                traj_num=3,
                traj_interval_min=2,
                traj_interval_max=2,
                split_dataloader_cfg=None,
                episode_starting_idx_max=None,
            )
            sample = dataset[0]
            self.assertNotIn("head_camera", sample)
            self.assertEqual(
                tuple(sample["head_camera_feature"].shape),
                (3, 1, 1 + PATCH_TOKEN_NUM, FEATURE_DIM),
            )


if __name__ == "__main__":
    unittest.main()
