"""Checkpoint-backed inference adapter for RMBench."""

from __future__ import annotations

import csv
import sys
from collections import deque
from pathlib import Path
from typing import Any

import dill
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


POLICY_ROOT = Path(__file__).resolve().parent
if str(POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICY_ROOT))

from imitation_learning.common.dataclasses import construct_data_meta_dict
from imitation_learning.datasets.rmbench_dataset import _select_camera_meta
from imitation_learning.datasets.normalizer import FixedNormalizer
from imitation_learning.datasets.transforms import BaseTransforms
from imitation_learning.policies.base_policy import BasePolicy
from imitation_learning.utils.config_utils import remove_keys_from_config
from robot_utils.config_utils import register_resolvers
from robot_utils.torch_utils import torch_load


if not OmegaConf.has_resolver("eval"):
    register_resolvers()


class RMBenchMemoryPolicy:
    """Load an imitation-learning checkpoint and expose RMBench methods.

    Prediction horizon and executed chunk length are recovered from the model
    configuration stored in the checkpoint. Evaluation changes only the target
    device; all model and memory settings remain exactly as they were saved by
    training.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        eval_output_dir: str | None = None,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.device = torch.device(device)

        checkpoint = torch_load(
            self.checkpoint_path,
            map_location=self.device,
            pickle_module=dill,
            weights_only=False,
        )
        if "cfg_str_unresolved" not in checkpoint:
            raise KeyError("Checkpoint does not contain cfg_str_unresolved")
        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint does not contain model_state_dict")
        if "normalizer_state_dict" not in checkpoint:
            raise KeyError("Checkpoint does not contain normalizer_state_dict")

        cfg = OmegaConf.create(checkpoint["cfg_str_unresolved"])
        if not isinstance(cfg, DictConfig):
            raise TypeError("Checkpoint configuration must be a DictConfig")
        OmegaConf.set_struct(cfg, False)
        cfg.workspace.model.device = str(self.device)
        self.cfg = remove_keys_from_config(cfg)

        policy = hydra.utils.instantiate(self.cfg.workspace.model)
        if not isinstance(policy, BasePolicy):
            raise TypeError(f"Expected BasePolicy, got {type(policy).__name__}")
        policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        policy.to(self.device)
        policy.eval()
        self.policy = policy
        self.attention_bias = bool(getattr(self.policy, "attention_bias", True))
        prediction_error_config = getattr(
            self.policy, "prediction_error_queue_config", None
        )
        self.weight_min = (
            float(prediction_error_config.weight_min)
            if prediction_error_config is not None
            else None
        )
        self.weight_max = (
            float(prediction_error_config.weight_max)
            if prediction_error_config is not None
            else None
        )
        self.first_slot_weight = (
            float(prediction_error_config.first_slot_weight)
            if prediction_error_config is not None
            else None
        )
        self.recent_slot_attention_floor = (
            float(prediction_error_config.recent_slot_attention_floor)
            if prediction_error_config is not None
            and prediction_error_config.recent_slot_attention_floor is not None
            else None
        )
        self._memory_attention_record_key = (
            "history_cross_attention_slot_weights"
        )
        if eval_output_dir is not None:
            record_entries = list(self.policy.denoising_network.record_data_entries)
            if self._memory_attention_record_key not in record_entries:
                record_entries.append(self._memory_attention_record_key)
            self.policy.denoising_network.record_data_entries = record_entries
        dynamic_memory_config = getattr(
            self.policy, "dynamic_memory_weight_update_config", None
        )
        self.recent_slot_protection_num = (
            int(dynamic_memory_config.recent_slot_protection_num)
            if dynamic_memory_config is not None
            else 0
        )

        self.image_keys = list(self.cfg.workspace.train_dataset.image_keys)
        selected_output_meta = _select_camera_meta(
            self.cfg.workspace.train_dataset.output_data_meta,
            self.image_keys,
        )
        data_meta = construct_data_meta_dict(selected_output_meta)
        self.normalizer = FixedNormalizer(data_meta)
        self.normalizer.load_state_dict(
            checkpoint["normalizer_state_dict"], strict=False
        )
        self.normalizer.to(self.device)

        self.transforms = BaseTransforms(
            data_meta=data_meta,
            apply_image_augmentation_in_cpu=True,
            seed=int(self.cfg.seed),
        )
        self.transforms.to(self.device)

        self.input_lengths = self._collect_input_lengths()
        self.observation_history = {
            name: deque(maxlen=length)
            for name, length in self.input_lengths.items()
        }

        action_names = list(self.policy.action_decoder.data_entry_names)
        if action_names != ["action"]:
            raise ValueError(
                "RMBench expects exactly one action entry named 'action', "
                f"got {action_names}"
            )
        self.action_name = action_names[0]
        self.prediction_horizon = int(
            self.policy.action_decoder.action_meta[self.action_name].length
        )

        configured_chunk_length = getattr(
            self.policy, "history_action_num_per_chunk", None
        )
        if configured_chunk_length is None:
            configured_chunk_length = self.prediction_horizon
        self.execution_chunk_length = int(configured_chunk_length)
        if not 1 <= self.execution_chunk_length <= self.prediction_horizon:
            raise ValueError(
                "history_action_num_per_chunk must be in [1, action_length], "
                f"got {self.execution_chunk_length} and "
                f"{self.prediction_horizon}"
            )

        self.eval_output_dir = (
            Path(eval_output_dir).expanduser().resolve()
            if eval_output_dir is not None
            else None
        )
        self.slot_weight_record_path = (
            self.eval_output_dir / "memory_slot_weights.csv"
            if self.eval_output_dir is not None
            else None
        )
        self._eval_episode_index = -1
        self._eval_decision_index = 0
        self._initialize_slot_weight_recording()

        print(
            f"Loaded checkpoint {self.checkpoint_path}: "
            f"prediction_horizon={self.prediction_horizon}, "
            f"execution_chunk_length={self.execution_chunk_length}, "
            f"recent_slot_protection_num={self.recent_slot_protection_num}, "
            f"attention_bias={self.attention_bias}, "
            f"weight_range=({self.weight_min}, {self.weight_max}), "
            f"first_slot_weight={self.first_slot_weight}, "
            f"recent_slot_attention_floor="
            f"{self.recent_slot_attention_floor}, "
            "config_source=checkpoint, "
            f"image_keys={self.image_keys}, "
            f"input_keys={list(self.input_lengths)}"
        )
        self._reset_runtime_state()

    def _initialize_slot_weight_recording(self) -> None:
        if self.slot_weight_record_path is None:
            return
        self.slot_weight_record_path.parent.mkdir(parents=True, exist_ok=True)
        with self.slot_weight_record_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "episode_index",
                    "decision_index",
                    "query_frame_index",
                    "active_slot_count",
                    "buffer_position",
                    "slot_id",
                    "slot_frame_index",
                    "slot_age_frames",
                    "slot_weight",
                    "attention_mean",
                    "attention_std",
                    "attention_max",
                ]
            )
        print(
            "Memory slot weights and attention will be recorded to "
            f"{self.slot_weight_record_path}"
        )

    def _latest_slot_attention_stats(
        self, active_slot_count: int
    ) -> tuple[list[float], list[float], list[float]]:
        """Summarize the actual post-softmax attention for retained slots."""

        if active_slot_count == 0:
            return [], [], []
        episode_records = getattr(self.policy, "recorded_data_dicts", {}).get(0)
        if not episode_records:
            raise RuntimeError("Missing recorded history attention for episode 0")
        attention = episode_records[-1].get(self._memory_attention_record_key)
        if attention is None:
            raise RuntimeError(
                "The latest memory query did not record post-softmax slot attention"
            )
        attention = attention.float()
        if attention.ndim < 2 or attention.shape[-1] < active_slot_count:
            raise RuntimeError(
                "Recorded slot attention has incompatible shape "
                f"{tuple(attention.shape)} for {active_slot_count} active slots"
            )

        # Online history is right-aligned in the fixed-length history tensor.
        # Flatten retrieval/diffusion/head/query axes while keeping the slot
        # axis, then retain exactly the slots described by the CSV metadata.
        attention = attention.reshape(-1, attention.shape[-1])
        attention = attention[:, -active_slot_count:]
        if not torch.isfinite(attention).all():
            raise RuntimeError("Recorded slot attention contains non-finite values")
        return (
            attention.mean(dim=0).tolist(),
            attention.std(dim=0, unbiased=False).tolist(),
            attention.amax(dim=0).tolist(),
        )

    def _record_memory_query(self) -> None:
        if self.slot_weight_record_path is None:
            return
        snapshots = getattr(
            self.policy, "latest_online_memory_query_dict", None
        )
        if snapshots is None:
            raise RuntimeError(
                "The loaded policy does not expose online memory query records"
            )
        snapshot = snapshots.get(0)
        if snapshot is None:
            raise RuntimeError("Missing online memory query record for episode 0")

        query_frame_index = int(snapshot["query_frame_index"])
        slot_ids = tuple(snapshot["slot_ids"])
        slot_frame_indices = tuple(snapshot["slot_frame_indices"])
        slot_weights = tuple(snapshot["slot_weights"])
        if not (
            len(slot_ids) == len(slot_frame_indices) == len(slot_weights)
        ):
            raise RuntimeError("Recorded slot metadata is not aligned")

        if self._eval_episode_index < 0:
            # Keep standalone adapter use sensible even if its caller omits an
            # explicit reset before the first action.
            self._eval_episode_index = 0
        active_slot_count = len(slot_ids)
        attention_means, attention_stds, attention_maxes = (
            self._latest_slot_attention_stats(active_slot_count)
        )
        if active_slot_count == 0:
            rows = [
                [
                    self._eval_episode_index,
                    self._eval_decision_index,
                    query_frame_index,
                    0,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            ]
        else:
            rows = []
            for buffer_position, (
                slot_id,
                slot_frame_index,
                slot_weight,
                attention_mean,
                attention_std,
                attention_max,
            ) in enumerate(
                zip(
                    slot_ids,
                    slot_frame_indices,
                    slot_weights,
                    attention_means,
                    attention_stds,
                    attention_maxes,
                )
            ):
                rows.append(
                    [
                        self._eval_episode_index,
                        self._eval_decision_index,
                        query_frame_index,
                        active_slot_count,
                        buffer_position,
                        int(slot_id),
                        int(slot_frame_index),
                        query_frame_index - int(slot_frame_index),
                        "" if slot_weight is None else format(slot_weight, ".9g"),
                        format(attention_mean, ".9g"),
                        format(attention_std, ".9g"),
                        format(attention_max, ".9g"),
                    ]
                )
        with self.slot_weight_record_path.open(
            "a", newline="", encoding="utf-8"
        ) as file:
            csv.writer(file).writerows(rows)

    def _collect_input_lengths(self) -> dict[str, int]:
        encoders = [self.policy.global_cond_encoder]
        if self.policy.local_cond_encoder is not None:
            encoders.append(self.policy.local_cond_encoder)
        history_encoder = getattr(
            self.policy, "history_img_feature_encoder", None
        )
        if history_encoder is not None:
            encoders.append(history_encoder)

        lengths: dict[str, int] = {}
        for encoder in encoders:
            for meta in encoder.cond_meta.values():
                if meta.name in lengths and lengths[meta.name] != meta.length:
                    raise ValueError(
                        f"Inconsistent history lengths for {meta.name}: "
                        f"{lengths[meta.name]} != {meta.length}"
                    )
                lengths[meta.name] = int(meta.length)
        if not lengths:
            raise ValueError("Checkpoint policy has no observation inputs")
        return lengths

    @staticmethod
    def _prepare_observation(name: str, value: Any) -> np.ndarray:
        array = np.asarray(value)
        if "camera" in name or "image" in name:
            if array.ndim != 3:
                raise ValueError(
                    f"{name} must have three image dimensions, got {array.shape}"
                )
            if array.shape[-1] in (1, 3, 4):
                array = np.moveaxis(array, -1, 0)
            if array.shape[0] not in (1, 3, 4):
                raise ValueError(
                    f"{name} must be CHW or HWC, got {array.shape}"
                )
            if array.dtype == np.uint8:
                array = array.astype(np.float32) / 255.0
            else:
                array = array.astype(np.float32, copy=False)
        else:
            array = array.astype(np.float32, copy=False)
        return np.ascontiguousarray(array)

    def _make_batch(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        missing = sorted(set(self.input_lengths) - set(obs))
        if missing:
            raise KeyError(f"RMBench observation is missing keys: {missing}")

        batch: dict[str, torch.Tensor] = {}
        for name, required_length in self.input_lengths.items():
            current = self._prepare_observation(name, obs[name])
            history = self.observation_history[name]
            history.append(current)
            padded_history = [history[0]] * (required_length - len(history))
            padded_history.extend(history)
            stacked = np.stack(padded_history, axis=0)
            batch[name] = (
                torch.from_numpy(stacked)
                .unsqueeze(0)
                .to(self.device, non_blocking=True)
            )

        batch = self.transforms.apply(batch)
        batch["episode_idx"] = torch.zeros(
            1, dtype=torch.long, device=self.device
        )
        return self.normalizer.normalize(batch)

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        batch = self._make_batch(obs)
        # Do not use torch.inference_mode(): prediction-error memory performs
        # an explicit local gradient update inside predict_action().
        with torch.no_grad():
            normalized_action = self.policy.predict_action(batch)
        self._record_memory_query()
        self._eval_decision_index += 1
        action = self.normalizer.unnormalize(normalized_action)[self.action_name]
        if (
            action.ndim != 3
            or action.shape[0] != 1
            or action.shape[1] != self.prediction_horizon
        ):
            raise ValueError(f"Unexpected action shape: {tuple(action.shape)}")
        return (
            action[0, : self.execution_chunk_length]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    def _reset_runtime_state(self) -> None:
        self.policy.reset()
        for history in self.observation_history.values():
            history.clear()

    def reset_model(self) -> None:
        self._eval_episode_index += 1
        self._eval_decision_index = 0
        self._reset_runtime_state()
