"""RMBench dataset adapters; model and checkpoint import paths stay stable."""

from imitation_learning.datasets.cached_image_dataset import (
    CAMERA_NAMES,
    CACHED_FEATURE_SUFFIX,
    CACHED_PATCH_FEATURE_SUFFIX,
    CachedDinoFeatureMixin,
    _prepare_kwargs,
    _resolve_cached_feature_kind,
    _select_camera_meta,
    _validate_cached_feature_store,
)
from imitation_learning.datasets.mujoco_dataset import (
    MujocoMultiTrajDataset,
    MujocoSingleTrajDataset,
)


class RMBenchSingleTrajDataset(CachedDinoFeatureMixin, MujocoSingleTrajDataset):
    """Read only selected RMBench cameras, optionally from a DINOv2 cache."""


class RMBenchMultiTrajDataset(CachedDinoFeatureMixin, MujocoMultiTrajDataset):
    """Use the shared ordered sampler with selected RMBench cameras."""
