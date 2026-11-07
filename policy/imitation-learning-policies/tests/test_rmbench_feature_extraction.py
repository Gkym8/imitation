from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import zarr

from imitation_learning.datasets.rmbench_dataset import _validate_cached_feature_store
from scripts import extract_rmbench_dinov2_features as extraction


MODEL_NAME = "test/dinov2-with-registers"
FEATURE_DIM = 8
INPUT_SIZE = 4
PATCH_TOKEN_NUM = 4


class TinyDino(nn.Module):
    """Exercise real CPU autocast without loading a pretrained backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="dinov2_with_registers", hidden_size=FEATURE_DIM,
            patch_size=2, num_register_tokens=1,
        )
        self.projection = nn.Linear(3, FEATURE_DIM)
        self.output_dtype = None

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.input_dtype = pixel_values.dtype
        self.autocast_enabled = torch.is_autocast_enabled(pixel_values.device.type)
        features = self.projection(pixel_values.mean(dim=(-2, -1)))
        self.output_dtype = features.dtype
        return SimpleNamespace(
            last_hidden_state=features[:, None, :].expand(-1, 6, -1)
        )


class RMBenchFeatureExtractionTest(unittest.TestCase):
    def test_cli_reads_config_and_allows_explicit_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "task.yaml"
            config.write_text(
                "workspace:\n  train_dataset:\n    cached_image_feature_dtype: bf16\n"
            )
            args = extraction.parse_args(["unused.zarr", "--config", str(config)])
            self.assertEqual(args.cached_image_feature_dtype, "bfloat16")
            args = extraction.parse_args([
                "unused.zarr", "--config", str(config),
                "--cached-image-feature-dtype", "fp32",
            ])
            self.assertEqual(args.cached_image_feature_dtype, "float32")

    def test_precision_controls_backbone_computation(self) -> None:
        for dtype in ("float32", "bfloat16"):
            with self.subTest(dtype=dtype):
                model = TinyDino()
                with patch.object(extraction, "load_pretrained_dinov2", return_value=model):
                    extractor = extraction.DinoV2FeatureExtractor(
                        MODEL_NAME, INPUT_SIZE, torch.device("cpu"), "patches", dtype,
                    )
                features = extractor(np.full((2, 4, 4, 3), 128, dtype=np.uint8))
                self.assertEqual(model.input_dtype, torch.float32)
                self.assertEqual(model.projection.weight.dtype, torch.float32)
                self.assertEqual(model.autocast_enabled, dtype == "bfloat16")
                self.assertEqual(model.output_dtype, getattr(torch, dtype))
                self.assertEqual(features.shape, (2, 1 + PATCH_TOKEN_NUM, FEATURE_DIM))
                self.assertEqual(features.dtype, np.dtype("float32"))
                self.assertEqual(extractor.metadata["feature_compute_dtype"], dtype)

    def test_bf16_cache_round_trip_and_precision_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test.zarr"
            root = zarr.open_group(str(dataset_path), mode="w")
            images = np.full((2, 4, 4, 3), 128, dtype=np.uint8)
            root.create_group("episode_0").create_dataset("head_camera", data=images)
            kwargs = dict(
                dataset_path=dataset_path, cameras=["head_camera"],
                model_name=MODEL_NAME, input_size=INPUT_SIZE, batch_size=1,
                feature_chunk_length=1, device=torch.device("cpu"),
                overwrite=False, feature_kind="patches", feature_dtype="bf16",
            )
            with patch.object(extraction, "load_pretrained_dinov2", side_effect=lambda _: TinyDino()):
                extraction.extract_dataset(**kwargs)
                # Matching complete caches are reusable; a different compute
                # precision must never silently reuse or resume them.
                extraction.extract_dataset(**kwargs)
                with self.assertRaisesRegex(RuntimeError, "incompatible"):
                    extraction.extract_dataset(**{**kwargs, "feature_dtype": "float32"})
            root = zarr.open_group(str(dataset_path), mode="r+")
            metadata = root.attrs["dinov2_patch_feature_cache"]
            self.assertEqual(metadata["feature_compute_dtype"], "bfloat16")
            self.assertEqual(metadata["feature_dtype"], "float32")
            np.testing.assert_array_equal(root["episode_0/head_camera"][:], images)
            dataset = SimpleNamespace(
                zarr_store=root, zarr_path=dataset_path,
                used_episode_indices=[0], episode_frame_nums={0: 2},
            )
            validation = dict(
                image_keys=["head_camera"], feature_dim=FEATURE_DIM,
                model_name=MODEL_NAME, input_size=INPUT_SIZE, feature_kind="patches",
                patch_token_num=PATCH_TOKEN_NUM,
            )
            _validate_cached_feature_store(dataset, feature_dtype="bf16", **validation)
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                _validate_cached_feature_store(dataset, feature_dtype="float32", **validation)
            # An interrupted precision overwrite can leave a legacy root
            # header alongside newly written arrays. Validate each array too.
            metadata.pop("feature_compute_dtype")
            root.attrs["dinov2_patch_feature_cache"] = metadata
            with self.assertRaisesRegex(RuntimeError, "has compute dtype"):
                _validate_cached_feature_store(dataset, feature_dtype="float32", **validation)

    def test_legacy_fp32_incomplete_cache_can_resume_only_in_fp32(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = zarr.open_group(temp_dir, mode="w")
            episode = root.create_group("episode_0")
            metadata = extraction._model_metadata(
                TinyDino(), MODEL_NAME, INPUT_SIZE, "cls",
            )
            legacy_metadata = dict(metadata)
            legacy_metadata.pop("feature_compute_dtype")
            kwargs = dict(
                episode=episode, camera_name="head_camera", episode_length=2,
                feature_shape=(FEATURE_DIM,), feature_suffix="_feature",
                feature_chunk_length=1, compressor=None, overwrite=False,
            )
            array, _ = extraction._prepare_incomplete_array(
                metadata=legacy_metadata, **kwargs,
            )
            array.attrs["frames_written"] = 1
            resumed, start = extraction._prepare_incomplete_array(metadata=metadata, **kwargs)
            self.assertEqual(start, 1)
            self.assertEqual(resumed.path, array.path)
            bf16_metadata = {**metadata, "feature_compute_dtype": "bfloat16"}
            with self.assertRaisesRegex(RuntimeError, "Cannot resume incompatible"):
                extraction._prepare_incomplete_array(metadata=bf16_metadata, **kwargs)


if __name__ == "__main__":
    unittest.main()
