"""RMBench dataset adapters built on the existing episodic loaders."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import zarr
from omegaconf import OmegaConf

from imitation_learning.datasets.mujoco_dataset import (
    MujocoMultiTrajDataset,
    MujocoSingleTrajDataset,
)


CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
CACHED_FEATURE_SUFFIX = "_feature"
CACHED_PATCH_FEATURE_SUFFIX = "_patch_feature"


def _resolve_cached_feature_kind(
    feature_kind: str, image_keys: list[str]
) -> str:
    if feature_kind == "auto":
        return "cls" if tuple(image_keys) == CAMERA_NAMES else "patches"
    if feature_kind not in {"cls", "patches"}:
        raise ValueError(
            "cached_image_feature_kind must be 'auto', 'cls', or 'patches', "
            f"got {feature_kind!r}"
        )
    return feature_kind


def _select_camera_meta(
    meta: Mapping[str, Any], image_keys: list[str]
) -> dict[str, Any]:
    """Keep all non-camera entries and only the requested camera entries."""

    if OmegaConf.is_config(meta):
        resolved_meta = OmegaConf.to_container(meta, resolve=True)
        if not isinstance(resolved_meta, dict):
            raise TypeError("dataset metadata must resolve to a dictionary")
    else:
        resolved_meta = deepcopy(dict(meta))

    return {
        name: value
        for name, value in resolved_meta.items()
        if name not in CAMERA_NAMES or name in image_keys
    }


def _prepare_kwargs(
    image_keys: list[str],
    source_data_meta: Mapping[str, Any],
    output_data_meta: Mapping[str, Any],
    kwargs: dict[str, Any],
    use_cached_image_features: bool = False,
    cached_image_feature_dim: int = 768,
    cached_image_feature_kind: str = "auto",
    cached_image_feature_patch_token_num: int = 256,
) -> dict[str, Any]:
    image_keys = list(image_keys)
    if not image_keys:
        raise ValueError("image_keys must select at least one RMBench camera")
    if len(set(image_keys)) != len(image_keys):
        raise ValueError(f"image_keys contains duplicate cameras: {image_keys}")

    unsupported = sorted(set(image_keys) - set(CAMERA_NAMES))
    if unsupported:
        raise ValueError(
            f"Unsupported RMBench camera(s): {unsupported}. "
            f"Choose from {list(CAMERA_NAMES)}"
        )

    selected_source_meta = _select_camera_meta(source_data_meta, image_keys)
    selected_output_meta = _select_camera_meta(output_data_meta, image_keys)

    if use_cached_image_features:
        resolved_feature_kind = _resolve_cached_feature_kind(
            cached_image_feature_kind, image_keys
        )
        if cached_image_feature_dim <= 0:
            raise ValueError("cached_image_feature_dim must be positive")
        if cached_image_feature_patch_token_num <= 0:
            raise ValueError(
                "cached_image_feature_patch_token_num must be positive"
            )

        # This replacement is local to the dataset constructor. The original
        # Hydra output_data_meta remains image-based, so the model checkpoint
        # keeps enough metadata to accept raw camera images during evaluation.
        for camera_name in image_keys:
            feature_name = f"{camera_name}{CACHED_FEATURE_SUFFIX}"
            if resolved_feature_kind == "cls":
                stored_feature_name = feature_name
                feature_shape = [cached_image_feature_dim]
            else:
                stored_feature_name = (
                    f"{camera_name}{CACHED_PATCH_FEATURE_SUFFIX}"
                )
                # Token 0 is CLS; the remaining tokens are spatial patches.
                feature_shape = [
                    1 + cached_image_feature_patch_token_num,
                    cached_image_feature_dim,
                ]
            camera_source_meta = selected_source_meta.pop(camera_name)
            camera_output_meta = selected_output_meta.pop(camera_name)
            selected_source_meta[stored_feature_name] = {
                "name": stored_feature_name,
                "include_indices": list(
                    camera_source_meta["include_indices"]
                ),
                "shape": feature_shape,
            }
            selected_output_meta[feature_name] = {
                "name": feature_name,
                "data_type": "low_dim",
                "length": camera_output_meta["length"],
                "normalizer": "identity",
                "augmentation": [],
                "shape": feature_shape,
                "source_entry_names": [stored_feature_name],
            }

    kwargs.update(
        {
            "image_keys": image_keys,
            "source_data_meta": selected_source_meta,
            "output_data_meta": selected_output_meta,
        }
    )
    return kwargs


def _validate_cached_feature_store(
    dataset: Any,
    image_keys: list[str],
    feature_dim: int,
    model_name: str,
    input_size: int,
    feature_dtype: str,
    feature_kind: str,
    patch_token_num: int,
) -> None:
    expected_dtype = np.dtype(feature_dtype)
    if expected_dtype != np.dtype(np.float32):
        raise ValueError(
            "Only float32 RMBench feature caches are supported, got "
            f"{expected_dtype}"
        )

    resolved_feature_kind = _resolve_cached_feature_kind(
        feature_kind, image_keys
    )
    cache_attr_name = (
        "dinov2_feature_cache"
        if resolved_feature_kind == "cls"
        else "dinov2_patch_feature_cache"
    )
    cache_metadata = dataset.zarr_store.attrs.get(cache_attr_name)
    if not isinstance(cache_metadata, Mapping):
        raise RuntimeError(
            f"{dataset.zarr_path} has no complete DINOv2 feature-cache "
            "metadata. Run scripts/extract_rmbench_dinov2_features.py first."
        )

    expected_metadata = {
        "model_name": model_name,
        "input_size": input_size,
        "feature_kind": resolved_feature_kind,
        "feature_dim": feature_dim,
        "feature_dtype": expected_dtype.name,
    }
    if resolved_feature_kind == "patches":
        expected_metadata["patch_token_num"] = patch_token_num
    mismatches = {
        key: (cache_metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if cache_metadata.get(key) != expected
    }
    cached_cameras = cache_metadata.get("camera_names", [])
    if not isinstance(cached_cameras, (list, tuple)) or not set(
        image_keys
    ).issubset(set(cached_cameras)):
        mismatches["camera_names"] = (cached_cameras, image_keys)
    if mismatches:
        raise RuntimeError(
            f"DINOv2 feature cache metadata does not match the training "
            f"configuration for {dataset.zarr_path}: {mismatches}"
        )

    for episode_idx in dataset.used_episode_indices:
        episode_name = f"episode_{episode_idx}"
        episode = dataset.zarr_store[episode_name]
        expected_length = int(dataset.episode_frame_nums[episode_idx])
        for camera_name in image_keys:
            feature_suffix = (
                CACHED_FEATURE_SUFFIX
                if resolved_feature_kind == "cls"
                else CACHED_PATCH_FEATURE_SUFFIX
            )
            feature_name = f"{camera_name}{feature_suffix}"
            if feature_name not in episode:
                raise RuntimeError(
                    f"Missing {episode_name}/{feature_name}. Run "
                    "scripts/extract_rmbench_dinov2_features.py first."
                )
            features = episode[feature_name]
            if not isinstance(features, zarr.Array):
                raise RuntimeError(
                    f"{episode_name}/{feature_name} is not a Zarr array"
                )
            expected_shape = (
                (expected_length, feature_dim)
                if resolved_feature_kind == "cls"
                else (
                    expected_length,
                    1 + patch_token_num,
                    feature_dim,
                )
            )
            if features.shape != expected_shape:
                raise RuntimeError(
                    f"{episode_name}/{feature_name} has shape "
                    f"{features.shape}, expected {expected_shape}"
                )
            if np.dtype(features.dtype) != expected_dtype:
                raise RuntimeError(
                    f"{episode_name}/{feature_name} has dtype "
                    f"{features.dtype}, expected {expected_dtype}"
                )
            if not bool(features.attrs.get("complete", False)):
                raise RuntimeError(
                    f"{episode_name}/{feature_name} is not marked complete"
                )


class _RMBenchCachedFeatureMixin:
    use_cached_image_features: bool
    image_keys: list[str]
    cached_image_feature_dim: int
    cached_image_feature_model_name: str
    cached_image_feature_input_size: int
    cached_image_feature_dtype: str
    cached_image_feature_kind: str
    cached_image_feature_patch_token_num: int

    def _process_source_data(
        self, data_dict: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        processed = super()._process_source_data(data_dict)  # type: ignore[misc]
        if self.use_cached_image_features and _resolve_cached_feature_kind(
            self.cached_image_feature_kind, self.image_keys
        ) == "patches":
            for camera_name in self.image_keys:
                stored_name = (
                    f"{camera_name}{CACHED_PATCH_FEATURE_SUFFIX}"
                )
                output_name = f"{camera_name}{CACHED_FEATURE_SUFFIX}"
                if stored_name in processed:
                    processed[output_name] = processed[stored_name]
        return processed

    def _check_data_validity(self) -> None:
        super()._check_data_validity()  # type: ignore[misc]
        if self.use_cached_image_features:
            _validate_cached_feature_store(
                self,
                image_keys=self.image_keys,
                feature_dim=self.cached_image_feature_dim,
                model_name=self.cached_image_feature_model_name,
                input_size=self.cached_image_feature_input_size,
                feature_dtype=self.cached_image_feature_dtype,
                feature_kind=self.cached_image_feature_kind,
                patch_token_num=self.cached_image_feature_patch_token_num,
            )


class RMBenchSingleTrajDataset(
    _RMBenchCachedFeatureMixin, MujocoSingleTrajDataset
):
    """Single-trajectory loader that reads only the selected camera views."""

    def __init__(
        self,
        image_keys: list[str],
        source_data_meta: Mapping[str, Any],
        output_data_meta: Mapping[str, Any],
        use_cached_image_features: bool = False,
        cached_image_feature_dim: int = 768,
        cached_image_feature_model_name: str = "",
        cached_image_feature_input_size: int = 224,
        cached_image_feature_dtype: str = "float32",
        cached_image_feature_kind: str = "auto",
        cached_image_feature_patch_token_num: int = 256,
        **kwargs: Any,
    ) -> None:
        self.use_cached_image_features = use_cached_image_features
        self.image_keys = list(image_keys)
        self.cached_image_feature_dim = cached_image_feature_dim
        self.cached_image_feature_model_name = cached_image_feature_model_name
        self.cached_image_feature_input_size = cached_image_feature_input_size
        self.cached_image_feature_dtype = cached_image_feature_dtype
        self.cached_image_feature_kind = cached_image_feature_kind
        self.cached_image_feature_patch_token_num = (
            cached_image_feature_patch_token_num
        )
        super().__init__(
            **_prepare_kwargs(
                image_keys,
                source_data_meta,
                output_data_meta,
                kwargs,
                use_cached_image_features=use_cached_image_features,
                cached_image_feature_dim=cached_image_feature_dim,
                cached_image_feature_kind=cached_image_feature_kind,
                cached_image_feature_patch_token_num=(
                    cached_image_feature_patch_token_num
                ),
            )
        )


class RMBenchMultiTrajDataset(
    _RMBenchCachedFeatureMixin, MujocoMultiTrajDataset
):
    """Multi-trajectory loader that reads only the selected camera views."""

    def __init__(
        self,
        image_keys: list[str],
        source_data_meta: Mapping[str, Any],
        output_data_meta: Mapping[str, Any],
        use_cached_image_features: bool = False,
        cached_image_feature_dim: int = 768,
        cached_image_feature_model_name: str = "",
        cached_image_feature_input_size: int = 224,
        cached_image_feature_dtype: str = "float32",
        cached_image_feature_kind: str = "auto",
        cached_image_feature_patch_token_num: int = 256,
        **kwargs: Any,
    ) -> None:
        self.use_cached_image_features = use_cached_image_features
        self.image_keys = list(image_keys)
        self.cached_image_feature_dim = cached_image_feature_dim
        self.cached_image_feature_model_name = cached_image_feature_model_name
        self.cached_image_feature_input_size = cached_image_feature_input_size
        self.cached_image_feature_dtype = cached_image_feature_dtype
        self.cached_image_feature_kind = cached_image_feature_kind
        self.cached_image_feature_patch_token_num = (
            cached_image_feature_patch_token_num
        )
        super().__init__(
            **_prepare_kwargs(
                image_keys,
                source_data_meta,
                output_data_meta,
                kwargs,
                use_cached_image_features=use_cached_image_features,
                cached_image_feature_dim=cached_image_feature_dim,
                cached_image_feature_kind=cached_image_feature_kind,
                cached_image_feature_patch_token_num=(
                    cached_image_feature_patch_token_num
                ),
            )
        )
