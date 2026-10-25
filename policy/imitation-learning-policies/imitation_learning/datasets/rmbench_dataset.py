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


class _RMBenchCameraMixin(CachedDinoFeatureMixin):
    def __init__(self, dit_three_view: bool = False, **kwargs):
        if not isinstance(dit_three_view, bool):
            raise ValueError("dit_three_view must be boolean")
        super().__init__(
            extra_image_keys=list(CAMERA_NAMES) if dit_three_view else None,
            **kwargs,
        )


class RMBenchSingleTrajDataset(_RMBenchCameraMixin, MujocoSingleTrajDataset):
    """Read only selected RMBench cameras, optionally from a DINOv2 cache."""


class RMBenchMultiTrajDataset(_RMBenchCameraMixin, MujocoMultiTrajDataset):
    """Use the shared ordered sampler with selected RMBench cameras."""
