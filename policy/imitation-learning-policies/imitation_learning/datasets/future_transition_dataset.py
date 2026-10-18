"""Exact, unpadded expert transitions backed by frozen RMBench DINO caches."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr


class FutureTransitionDataset(Dataset):
    """Return (v[t], p[t], A[t:t+L], v[t+L]) from a single episode.

    Sources optionally specify disjoint ``train_episodes``/``validation_episodes``
    as integer IDs, taking precedence over automatic subsampling. Otherwise
    optionally select ``episodes_per_task`` complete episodes before the seeded
    train/validation split. With no limit the split matches the policy loader.
    Only complete windows are indexed.
    """

    def __init__(
        self, sources: list[dict[str, Any]], split: str,
        execution_chunk_length: int = 10,
        camera_names: list[str] | None = None,
        train_ratio: float = 0.98, seed: int = 42,
        encoder_model_name: str = "facebook/dinov2-with-registers-base",
        encoder_input_size: int = 224,
        control_interface: str | None = None,
        episodes_per_task: int | None = None,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if not sources or execution_chunk_length <= 0 or not 0 < train_ratio < 1:
            raise ValueError("Nonempty sources, a positive horizon and 0 < train_ratio < 1 are required")
        if episodes_per_task is not None and (type(episodes_per_task) is not int or episodes_per_task < 2):
            raise ValueError("episodes_per_task must be null or an integer >= 2")
        self.camera_names = list(["head_camera"] if camera_names is None else camera_names)
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names must be nonempty and unique")
        self.execution_chunk_length = execution_chunk_length
        self.split = split
        self.sources: list[dict[str, Any]] = []
        self.episodes: list[tuple[int, str, int]] = []
        self.ends: list[int] = []
        self.source_counts: list[int] = []
        self._stores: dict[int, Any] = {}
        # Zarr groups reconstruct Array objects (and reread metadata files) on
        # every lookup. Retain validated handles, never whole feature arrays.
        self._arrays: dict[tuple[int, str, str], Any] = {}
        self.feature_contract: dict[str, Any] = {}
        self.action_dim = self.proprio_dim = 0
        interfaces = set()
        alignments = set()
        seen_paths = set()
        seen_names = set()
        total = 0
        for source_index, source in enumerate(sources):
            path = str(Path(source["path"]).expanduser().resolve())
            name = str(source.get("name", Path(path).stem))
            if path in seen_paths or name in seen_names:
                raise ValueError("Source paths and names must be unique")
            seen_paths.add(path)
            seen_names.add(name)
            interface = source.get("control_interface", control_interface)
            if len(sources) > 1 and not interface:
                raise ValueError("Multi-task training requires an explicit shared control_interface")
            interfaces.add(interface)
            root = zarr.open_group(path, mode="r")
            self._stores[source_index] = root
            cache = dict(root.attrs.get("dinov2_patch_feature_cache", {}))
            required = ("model_name", "input_size", "feature_kind", "feature_dim",
                        "feature_dtype", "patch_token_num", "camera_names", "resize",
                        "resize_antialias", "image_scale", "image_mean", "image_std")
            if any(key not in cache for key in required):
                raise ValueError(f"{path}: missing complete patch cache; run scripts/extract_rmbench_dinov2_features.py --feature-kind patches")
            if (cache["model_name"] != encoder_model_name or cache["input_size"] != encoder_input_size
                    or cache["feature_kind"] != "patches" or cache["feature_dtype"] != "float32"
                    or not set(self.camera_names).issubset(cache["camera_names"])):
                raise ValueError(f"{path}: incompatible DINO patch cache")
            # All representation/preprocessing fields must agree across tasks.
            contract = {key: value for key, value in cache.items()
                        if key not in {"camera_names", "array_suffix", "extractor_version"}}
            contract["camera_names"] = self.camera_names
            if self.feature_contract and contract != self.feature_contract:
                raise ValueError("DINO feature contracts differ between sources")
            self.feature_contract = contract
            episode_ids = sorted(int(match.group(1)) for key in root.group_keys()
                                 if (match := re.fullmatch(r"episode_(\d+)", key)))
            explicit = "train_episodes" in source or "validation_episodes" in source
            if explicit:
                if "train_episodes" not in source or "validation_episodes" not in source:
                    raise ValueError("Specify both train_episodes and validation_episodes")
                train = list(source["train_episodes"])
                validation = list(source["validation_episodes"])
                for selected in (train, validation):
                    if (not selected or len(set(selected)) != len(selected)
                            or any(not isinstance(item, int) for item in selected)
                            or not set(selected).issubset(episode_ids)):
                        raise ValueError(f"{name}: invalid or empty explicit episode split")
                if set(train) & set(validation):
                    raise ValueError(f"{name}: train/validation episodes overlap")
            else:
                rng = np.random.default_rng(seed)
                if episodes_per_task is not None and len(episode_ids) > episodes_per_task:
                    episode_ids = sorted(rng.choice(episode_ids, size=episodes_per_task,
                                                    replace=False).tolist())
                count = int(len(episode_ids) * train_ratio)
                if not 0 < count < len(episode_ids):
                    raise ValueError(f"{name}: not enough episodes for a nonempty split")
                train = rng.choice(episode_ids, size=count, replace=False).tolist()
                validation = sorted(set(episode_ids) - set(train))
            selected = sorted(train if split == "train" else validation)
            source_record = dict(path=path, name=name, control_interface=interface,
                                 train_episodes=sorted(train), validation_episodes=sorted(validation),
                                 state_action_alignment=root.attrs.get("state_action_alignment"))
            alignments.add(source_record["state_action_alignment"])
            self.sources.append(source_record)
            source_start = total
            for episode_id in selected:
                episode_name = f"episode_{episode_id}"
                episode = root[episode_name]
                actions, proprio = episode["action"], episode["agent_pos"]
                self._arrays[source_index, episode_name, "action"] = actions
                self._arrays[source_index, episode_name, "agent_pos"] = proprio
                if actions.ndim != 2 or proprio.ndim != 2:
                    raise ValueError(f"{name}/{episode_name}: action and agent_pos must be [T,D]")
                dims = (actions.shape[1], proprio.shape[1])
                if self.action_dim and dims != (self.action_dim, self.proprio_dim):
                    raise ValueError("Action/proprio dimensions differ between episodes or tasks")
                self.action_dim, self.proprio_dim = dims
                length = proprio.shape[0]
                if actions.shape[0] not in (length, length - 1):
                    raise ValueError(f"{name}/{episode_name}: incompatible observation/action lengths")
                shape = (length, 1 + cache["patch_token_num"], cache["feature_dim"])
                for camera in self.camera_names:
                    key = f"{camera}_patch_feature"
                    if key not in episode:
                        raise ValueError(f"{name}/{episode_name}/{key}: missing or invalid patch cache")
                    array = episode[key]
                    if array.shape != shape or array.dtype != np.dtype("float32"):
                        raise ValueError(f"{name}/{episode_name}/{key}: missing or invalid patch cache")
                    if not array.attrs.get("complete", False):
                        raise ValueError(f"{name}/{episode_name}/{key}: patch cache is incomplete")
                    self._arrays[source_index, episode_name, key] = array
                count = max(0, min(length - execution_chunk_length,
                                   actions.shape[0] - execution_chunk_length + 1))
                if count:
                    self.episodes.append((source_index, episode_name, count))
                    total += count
                    self.ends.append(total)
            self.source_counts.append(total - source_start)
            if total == source_start:
                raise ValueError(f"{name}: {split} has no complete future transition")
        if len(interfaces) != 1:
            raise ValueError("Sources must use the same control_interface")
        if len(alignments) != 1:
            raise ValueError("Sources must use the same state_action_alignment")
        self.control_interface = interfaces.pop()

    def __len__(self) -> int:
        return self.ends[-1]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_stores"] = {}
        state["_arrays"] = {}
        return state

    def _episode(self, source_index: int, episode_name: str) -> Any:
        if source_index not in self._stores:
            self._stores[source_index] = zarr.open_group(self.sources[source_index]["path"], mode="r")
        return self._stores[source_index][episode_name]

    def _array(self, source_index: int, episode_name: str, name: str) -> Any:
        key = (source_index, episode_name, name)
        if key not in self._arrays:
            self._arrays[key] = self._episode(source_index, episode_name)[name]
        return self._arrays[key]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = bisect_right(self.ends, index)
        start = 0 if episode_index == 0 else self.ends[episode_index - 1]
        source_index, episode_name, _ = self.episodes[episode_index]
        frame = index - start
        future = frame + self.execution_chunk_length

        def patches(at: int) -> torch.Tensor:
            return torch.from_numpy(np.concatenate([
                np.asarray(self._array(source_index, episode_name, f"{camera}_patch_feature")[at, 1:],
                           dtype=np.float32)
                for camera in self.camera_names], axis=0))

        return dict(
            current_features=patches(frame), target_features=patches(future),
            proprio=torch.from_numpy(np.asarray(self._array(source_index, episode_name, "agent_pos")[frame],
                                               dtype=np.float32)),
            actions=torch.from_numpy(np.asarray(self._array(source_index, episode_name, "action")[frame:future],
                                               dtype=np.float32)),
            source_index=source_index, episode=episode_name,
            frame_index=frame, target_frame_index=future,
        )

    def fit_normalizer(self, block_size: int = 4096) -> dict[str, list[float]]:
        """Fit shared ranges on TRAIN episodes only, streaming low-dimensional data."""
        if self.split != "train":
            raise ValueError("Only training episodes may fit the normalizer")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        result = {}
        for field, key in (("action", "action"), ("proprio", "agent_pos")):
            minimum = maximum = None
            for source_index, episode_name, _ in self.episodes:
                array = self._array(source_index, episode_name, key)
                for start in range(0, array.shape[0], block_size):
                    values = np.asarray(array[start:start + block_size], dtype=np.float64)
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite {key} in {episode_name}")
                    low, high = values.min(axis=0), values.max(axis=0)
                    minimum = low if minimum is None else np.minimum(minimum, low)
                    maximum = high if maximum is None else np.maximum(maximum, high)
            scale = (maximum - minimum) / 2
            scale = np.where(scale < 1e-6, 1.0, scale)
            result[f"{field}_offset"] = ((minimum + maximum) / 2).astype(np.float32).tolist()
            result[f"{field}_scale"] = scale.astype(np.float32).tolist()
        return result

    def sampling_weights(self) -> torch.Tensor:
        """Equal source probability, uniform complete transitions within a source."""
        return torch.cat([torch.full((count,), 1.0 / count, dtype=torch.double)
                          for count in self.source_counts])

    def metadata(self) -> dict[str, Any]:
        return dict(feature_contract=self.feature_contract,
                    control_interface=self.control_interface, sources=self.sources,
                    execution_chunk_length=self.execution_chunk_length,
                    action_dim=self.action_dim, proprio_dim=self.proprio_dim)
