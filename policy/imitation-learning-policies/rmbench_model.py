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


def _restore_dit_encoder_value_types(cfg: DictConfig) -> None:
    """Recover types lost by old whole-dictionary checkpoint interpolation."""

    dit_encoder = OmegaConf.select(
        cfg, "workspace.model.dit_img_feature_encoder_partial.image_encoder_partial"
    )
    current_encoder = OmegaConf.select(
        cfg, "workspace.model.global_cond_encoder.image_encoder_partial"
    )
    if not isinstance(dit_encoder, DictConfig) or not isinstance(current_encoder, DictConfig):
        return
    saved_values = OmegaConf.to_container(dit_encoder, resolve=True)
    current_values = OmegaConf.to_container(current_encoder, resolve=True)
    restored = []
    for name, value in saved_values.items():
        reference = current_values.get(name)
        # The DiT encoder aliases the current-image encoder. Only restore a
        # scalar when its spelling matches that checkpoint's typed value;
        # retain explicit differences and ordinary string identifiers.
        if (
            isinstance(value, str)
            and isinstance(reference, (bool, int, float))
            and value == str(reference)
        ):
            dit_encoder[name] = reference
            restored.append(name)
    if restored:
        print(f"Restored checkpoint DiT encoder value types: {', '.join(restored)}")


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
        _restore_dit_encoder_value_types(self.cfg)

        policy = hydra.utils.instantiate(self.cfg.workspace.model)
        if not isinstance(policy, BasePolicy):
            raise TypeError(f"Expected BasePolicy, got {type(policy).__name__}")
        policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        policy.to(self.device)
        policy.eval()
        self.policy = policy
        self._gate_head_num = int(self.policy.denoising_network.head_num)
        self._memory_attention_record_key = (
            "history_cross_attention_slot_weights"
        )
        if eval_output_dir is not None:
            record_entries = list(self.policy.denoising_network.record_data_entries)
            if self._memory_attention_record_key not in record_entries:
                record_entries.append(self._memory_attention_record_key)
            self.policy.denoising_network.record_data_entries = record_entries
        self.image_keys = list(self.cfg.workspace.train_dataset.image_keys)
        dit_encoder = getattr(self.policy, "dit_img_feature_encoder", None)
        input_image_keys = list(dict.fromkeys([
            *self.image_keys,
            *(dit_encoder.data_entry_names if dit_encoder is not None else []),
        ]))
        selected_output_meta = _select_camera_meta(
            self.cfg.workspace.train_dataset.output_data_meta,
            input_image_keys,
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
            # Keep deterministic preprocessing such as Resize, but never use
            # stochastic training augmentation during evaluation.
            apply_stochastic_image_augmentation=False,
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
            self.policy, "execution_chunk_length", None
        )
        if configured_chunk_length is None:
            configured_chunk_length = self.prediction_horizon
        self.execution_chunk_length = int(configured_chunk_length)
        if not 1 <= self.execution_chunk_length <= self.prediction_horizon:
            raise ValueError(
                "execution_chunk_length must be in [1, action_length], "
                f"got {self.execution_chunk_length} and "
                f"{self.prediction_horizon}"
            )

        self.eval_output_dir = (
            Path(eval_output_dir).expanduser().resolve()
            if eval_output_dir is not None
            else None
        )
        self.slot_gate_record_path = (
            self.eval_output_dir / "memory_slot_gates.csv"
            if self.eval_output_dir is not None
            else None
        )
        self._eval_episode_index = -1
        self._initialize_slot_gate_recording()

        print(
            f"Loaded checkpoint {self.checkpoint_path}: "
            f"prediction_horizon={self.prediction_horizon}, "
            f"execution_chunk_length={self.execution_chunk_length}, "
            f"keep_first={self.policy.keep_first_memory}, "
            f"keep_recent={self.policy.keep_recent_memory}, "
            f"shadow_window={self.policy.denoising_network.shadow_window}, "
            f"shadow_horizon={self.policy.denoising_network.effective_shadow_horizon}, "
            f"fusion_gate={self.policy.denoising_network.fusion_gate}, "
            f"shadow_gate_scale={self.policy.denoising_network.memory_retriever.shadow_gate_scale}, "
            "config_source=checkpoint, "
            f"image_keys={self.image_keys}, "
            f"input_keys={list(self.input_lengths)}"
        )
        self._reset_runtime_state()

    def _initialize_slot_gate_recording(self) -> None:
        if self.slot_gate_record_path is None:
            return
        self._slot_gate_record_fieldnames = [
            "episode_index", "query_anchor_index", "slot_id", "slot_state",
            "gate_mean",
            *[f"gate_head_{i}" for i in range(self._gate_head_num)],
            "attention_mean",
        ]
        self.slot_gate_record_path.parent.mkdir(parents=True, exist_ok=True)
        with self.slot_gate_record_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            csv.DictWriter(file, fieldnames=self._slot_gate_record_fieldnames).writeheader()
        print(
            "Memory slot gates and attention will be recorded to "
            f"{self.slot_gate_record_path}"
        )

    def _latest_slot_attention(
        self, retained_slot_count: int
    ) -> list[float]:
        """Average actual post-softmax attention for each retained slot."""

        if retained_slot_count == 0:
            return []
        episode_records = getattr(self.policy, "recorded_data_dicts", {}).get(0)
        if not episode_records:
            raise RuntimeError("Missing recorded history attention for episode 0")
        attention = episode_records[-1].get(self._memory_attention_record_key)
        if attention is None:
            raise RuntimeError(
                "The latest memory query did not record post-softmax slot attention"
            )
        attention = attention.float()
        if attention.ndim < 2 or attention.shape[-1] < retained_slot_count:
            raise RuntimeError(
                "Recorded slot attention has incompatible shape "
                f"{tuple(attention.shape)} for {retained_slot_count} retained slots"
            )

        # Online history is right-aligned in the fixed-length history tensor.
        # Flatten retrieval/diffusion/head/query axes while keeping the slot
        # axis, then retain exactly the slots described by the CSV metadata.
        attention = attention.reshape(-1, attention.shape[-1])
        attention = attention[:, -retained_slot_count:]
        if not torch.isfinite(attention).all():
            raise RuntimeError("Recorded slot attention contains non-finite values")
        return attention.mean(dim=0).tolist()

    def _record_memory_query(self) -> None:
        if self.slot_gate_record_path is None:
            return
        snapshot = self.policy.latest_online_memory_query_dict.get(0)
        if snapshot is None:
            raise RuntimeError("Missing online memory query record for episode 0")
        ids = tuple(snapshot["slot_ids"])
        gates = tuple(snapshot["slot_gates"])
        states = tuple(snapshot["slot_states"])
        anchor = int(snapshot["query_anchor_index"])
        if not (len(ids) == len(gates) == len(states)):
            raise RuntimeError("Recorded slot metadata is not aligned")
        if any(len(heads) != self._gate_head_num for heads in gates):
            raise RuntimeError("Recorded gate head count does not match checkpoint")
        if gates and not np.isfinite(np.asarray(gates)).all():
            raise RuntimeError("Recorded gates contain non-finite values")
        if any(state not in {"active", "shadow"} for state in states):
            raise RuntimeError("Recorded slot state is invalid")
        if self._eval_episode_index < 0:
            self._eval_episode_index = 0
        attention = self._latest_slot_attention(len(ids))
        prefix = {
            "episode_index": self._eval_episode_index,
            "query_anchor_index": anchor,
        }
        rows = []
        if not ids:
            rows.append(prefix)
        for position, (slot_id, heads, state) in enumerate(
            zip(ids, gates, states)
        ):
            row = dict(prefix)
            row.update({
                "slot_id": int(slot_id),
                "slot_state": state,
                "gate_mean": format(sum(heads) / len(heads), ".9g"),
                "attention_mean": format(attention[position], ".9g"),
            })
            row.update({f"gate_head_{i}": format(value, ".9g") for i, value in enumerate(heads)})
            rows.append(row)
        with self.slot_gate_record_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=self._slot_gate_record_fieldnames).writerows(rows)

    def _collect_input_lengths(self) -> dict[str, int]:
        encoders = [self.policy.global_cond_encoder]
        dit_encoder = getattr(self.policy, "dit_img_feature_encoder", None)
        if dit_encoder is not None:
            encoders.append(dit_encoder)
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
        with torch.inference_mode():
            normalized_action = self.policy.predict_action(batch)
        self._record_memory_query()
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
        self._reset_runtime_state()
