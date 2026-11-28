"""Mikasa adapters using the same feature cache and streaming memory training."""

from imitation_learning.datasets.cached_image_dataset import CachedDinoFeatureMixin
from imitation_learning.datasets.mujoco_dataset import (
    MujocoMultiTrajDataset,
    MujocoSingleTrajDataset,
)


CAMERA_NAMES = ("third_person_camera", "robot0_wrist_camera")


class MikasaSingleTrajDataset(CachedDinoFeatureMixin, MujocoSingleTrajDataset):
    camera_names = CAMERA_NAMES


class MikasaMultiTrajDataset(CachedDinoFeatureMixin, MujocoMultiTrajDataset):
    camera_names = CAMERA_NAMES
