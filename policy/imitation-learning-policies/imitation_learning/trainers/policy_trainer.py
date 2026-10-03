import copy
import contextlib
import csv
import os
from functools import partial
from typing import Any, cast

import numpy as np
from imitation_learning.datasets.base_dataset import BaseDataset
import torch
import torch.nn.functional as F
import tqdm
from accelerate import Accelerator

from imitation_learning.common.datatypes import batch_type
from imitation_learning.envs.base_env import BaseEnv
from imitation_learning.models.memory.slot_weight_table import (
    write_active_slot_weight_csv)
from imitation_learning.policies.base_policy import BasePolicy
from imitation_learning.policies.history_denoising_policy import HistoryDenoisingPolicy
from imitation_learning.trainers.base_trainer import BaseTrainer
from robot_utils.torch_utils import aggregate_batch, torch_save
from imitation_learning.utils.data_utils import get_shakiness_score_torch



class PolicyTrainer(BaseTrainer):

    def __init__(
        self,
        rollout_every: int,
        rollout_env: BaseEnv | None,
        critical_action_loss_weights: list[float],
        memory_gate_loss_weight: float,
        future_prediction_loss_weight: float = 1.0,
        streaming_chunk_length: int = 4,
        streaming_chunk_accumulate: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rollout_every: int = rollout_every
        assert rollout_every >= 0, "rollout_every must be non-negative"
        assert self.rollout_every == 0 or (
            self.rollout_every % self.checkpoint_every == 0
        ), "rollout_every must be a multiple of checkpoint_every"

        if not isinstance(rollout_env, BaseEnv):
            rollout_env = None
        self.rollout_env: BaseEnv | None = rollout_env
        self.memory_gate_loss_weight: float = memory_gate_loss_weight
        if future_prediction_loss_weight < 0:
            raise ValueError("future_prediction_loss_weight must be non-negative")
        self.future_prediction_loss_weight = future_prediction_loss_weight
        if (
            isinstance(streaming_chunk_length, bool)
            or not isinstance(streaming_chunk_length, int)
            or streaming_chunk_length <= 0
        ):
            raise ValueError(
                "streaming_chunk_length must be a positive integer"
            )
        self.streaming_chunk_length = streaming_chunk_length
        if (
            isinstance(streaming_chunk_accumulate, bool)
            or not isinstance(streaming_chunk_accumulate, int)
            or streaming_chunk_accumulate <= 0
        ):
            raise ValueError(
                "streaming_chunk_accumulate must be a positive integer"
            )
        self.streaming_chunk_accumulate = streaming_chunk_accumulate
        self.critical_action_loss_weights: list[float] = critical_action_loss_weights
        assert (
            len(self.critical_action_loss_weights) > 0
        ), "critical_action_loss_weights must be non-empty"
        self._prediction_error_queue_epoch_records: list[
            dict[str, Any]
        ] = []
        self._slot_weight_epoch_records: list[dict[str, Any]] = []

    @torch.no_grad()
    def _record_prediction_error_queue_stats(
        self, queue_stats: dict[str, torch.Tensor]
    ) -> None:
        """Collect queue diagnostics for local epoch files, never trackers."""

        if not queue_stats:
            return
        stat_names = sorted(queue_stats)
        scalar_stats = []
        for name in stat_names:
            value = queue_stats[name]
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise ValueError(
                    "prediction-error queue statistics must be scalar tensors; "
                    f"{name} has value {value!r}"
                )
            scalar_stats.append(value.detach().float().reshape(()))
        local_stats = torch.stack(scalar_stats).unsqueeze(0)

        if self.accelerator is not None:
            gathered_stats = self.accelerator.gather(local_stats)
            is_main_process = self.accelerator.is_main_process
        else:
            gathered_stats = local_stats
            is_main_process = True

        if is_main_process:
            self._prediction_error_queue_epoch_records.append(
                {
                    "global_step": self.global_step,
                    "stats": {
                        name: gathered_stats[:, idx].detach().cpu()
                        for idx, name in enumerate(stat_names)
                    },
                }
            )

    @torch.no_grad()
    def _record_slot_weight_table(
        self, slot_table: dict[str, torch.Tensor]
    ) -> None:
        """Collect active memory-slot weights locally for an epoch CSV."""

        required_keys = {
            "weight",
            "valid_mask",
            "source_slot_index",
            "anchor_frame_index",
        }
        missing_keys = required_keys.difference(slot_table)
        if missing_keys:
            raise ValueError(
                "slot weight table is missing fields: "
                f"{sorted(missing_keys)}"
            )
        unexpected_keys = set(slot_table).difference(required_keys)
        if unexpected_keys:
            raise ValueError(
                "slot weight table has unexpected fields: "
                f"{sorted(unexpected_keys)}"
            )

        weight = slot_table["weight"].detach().to(dtype=torch.float32)
        if weight.ndim != 2:
            raise ValueError(
                "slot weights must have shape (batch, history_len), got "
                f"{tuple(weight.shape)}"
            )
        local_batch_size = weight.shape[0]
        tensors = {
            "weight": weight,
            "valid_mask": slot_table["valid_mask"].detach().to(
                device=weight.device, dtype=torch.uint8
            ),
            "source_slot_index": slot_table["source_slot_index"].detach().to(
                device=weight.device, dtype=torch.long
            ),
            "anchor_frame_index": slot_table["anchor_frame_index"].detach().to(
                device=weight.device, dtype=torch.long
            ),
        }
        for name, tensor in tensors.items():
            if tensor.shape != weight.shape:
                raise ValueError(
                    f"{name} must have shape {tuple(weight.shape)}, got "
                    f"{tuple(tensor.shape)}"
                )

        process_index = (
            0 if self.accelerator is None else self.accelerator.process_index
        )
        tensors["rank"] = torch.full(
            (local_batch_size,),
            process_index,
            dtype=torch.long,
            device=weight.device,
        )
        tensors["sample_index"] = torch.arange(
            local_batch_size, dtype=torch.long, device=weight.device
        )

        if self.accelerator is not None:
            gathered = {
                name: self.accelerator.gather(tensor.contiguous())
                for name, tensor in tensors.items()
            }
            is_main_process = self.accelerator.is_main_process
        else:
            gathered = tensors
            is_main_process = True

        if is_main_process:
            self._slot_weight_epoch_records.append(
                {
                    "global_step": self.global_step,
                    **{
                        name: tensor.detach().cpu()
                        for name, tensor in gathered.items()
                    },
                }
            )

    def save_epoch_offline_records(self) -> None:
        """Write queue diagnostics and active-slot tables only to disk."""

        if self.accelerator is not None and not self.accelerator.is_main_process:
            self._prediction_error_queue_epoch_records.clear()
            self._slot_weight_epoch_records.clear()
            return

        if self._prediction_error_queue_epoch_records:
            records = self._prediction_error_queue_epoch_records
            stat_names = sorted(
                {
                    name
                    for record in records
                    for name in record["stats"]
                }
            )
            stat_templates = {
                name: next(
                    record["stats"][name]
                    for record in records
                    if name in record["stats"]
                )
                for name in stat_names
            }
            payload = {
                "format_version": 1,
                "epoch": self.epoch,
                "global_steps": torch.tensor(
                    [record["global_step"] for record in records],
                    dtype=torch.long,
                ),
                # Each tensor is [optimizer_steps, distributed_processes].
                "stats": {
                    name: torch.stack(
                        [
                            record["stats"].get(
                                name,
                                torch.full_like(
                                    stat_templates[name], float("nan")
                                ),
                            )
                            for record in records
                        ],
                        dim=0,
                    )
                    for name in stat_names
                },
            }
            record_dir = os.path.join(
                self.output_dir,
                "offline_training_records",
                "prediction_error_queue",
            )
            os.makedirs(record_dir, exist_ok=True)
            record_path = os.path.join(
                record_dir, f"epoch_{self.epoch:04d}.pt"
            )
            torch_save(payload, record_path)
            self._prediction_error_queue_epoch_records.clear()
            print(
                "Saved offline prediction-error queue records to "
                f"{record_path}"
            )

        if self._slot_weight_epoch_records:
            record_dir = os.path.join(
                self.output_dir,
                "offline_training_records",
                "slot_weights",
            )
            os.makedirs(record_dir, exist_ok=True)
            record_path = os.path.join(
                record_dir, f"epoch_{self.epoch:04d}.csv"
            )
            write_active_slot_weight_csv(
                record_path,
                epoch=self.epoch,
                records=self._slot_weight_epoch_records,
            )
            self._slot_weight_epoch_records.clear()
            print(f"Saved offline active slot-weight table to {record_path}")

    @staticmethod
    def _combine_attention_layers(
        metric: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Combine exact moments over the transformer-layer dimension."""
        mean = metric["mean"].astype(np.float64)
        std = metric["std"].astype(np.float64)
        count = metric["count"].astype(np.float64)
        total_count = count.sum(axis=0)
        safe_count = np.maximum(total_count, 1.0)
        total_sum = np.nansum(mean * count, axis=0)
        total_sum_sq = np.nansum((std ** 2 + mean ** 2) * count, axis=0)
        combined_mean = total_sum / safe_count
        combined_var = np.maximum(total_sum_sq / safe_count - combined_mean ** 2, 0)
        combined_mean = np.where(total_count > 0, combined_mean, np.nan)
        combined_std = np.where(total_count > 0, np.sqrt(combined_var), np.nan)
        return {
            "mean": combined_mean,
            "std": combined_std,
            "min": np.nanmin(metric["min"], axis=0),
            "max": np.nanmax(metric["max"], axis=0),
            "count": total_count,
        }

    def _merge_attention_stats_across_processes(
        self, stats: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Merge exact moments from every DDP rank before rank-zero export."""
        if (
            self.accelerator is None
            or self.accelerator.num_processes <= 1
            or not torch.distributed.is_initialized()
        ):
            return stats

        device = self.accelerator.device
        merged: dict[str, dict[str, torch.Tensor]] = {}
        for metric_name, metric in stats.items():
            count = metric["count"].to(device=device, dtype=torch.float64)
            mean = metric["mean"].to(device=device, dtype=torch.float64)
            std = metric["std"].to(device=device, dtype=torch.float64)
            has_data = count > 0

            value_sum = torch.where(
                has_data, mean * count, torch.zeros_like(mean)
            )
            value_sum_sq = torch.where(
                has_data,
                (std.square() + mean.square()) * count,
                torch.zeros_like(mean),
            )
            minimum = torch.where(
                has_data,
                metric["min"].to(device=device, dtype=torch.float64),
                torch.full_like(mean, float("inf")),
            )
            maximum = torch.where(
                has_data,
                metric["max"].to(device=device, dtype=torch.float64),
                torch.full_like(mean, -float("inf")),
            )

            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                value_sum, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                value_sum_sq, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                minimum, op=torch.distributed.ReduceOp.MIN
            )
            torch.distributed.all_reduce(
                maximum, op=torch.distributed.ReduceOp.MAX
            )

            safe_count = count.clamp_min(1)
            merged_mean = value_sum / safe_count
            merged_variance = (
                value_sum_sq / safe_count - merged_mean.square()
            ).clamp_min(0)
            has_merged_data = count > 0
            nan = torch.full_like(merged_mean, float("nan"))
            merged[metric_name] = {
                "mean": torch.where(has_merged_data, merged_mean, nan).float().cpu(),
                "std": torch.where(
                    has_merged_data, merged_variance.sqrt(), nan
                ).float().cpu(),
                "min": torch.where(has_merged_data, minimum, nan).float().cpu(),
                "max": torch.where(has_merged_data, maximum, nan).float().cpu(),
                "count": count.to(dtype=torch.int64).cpu(),
            }
        return merged

    def _save_history_attention_epoch_stats(self) -> None:
        """Write local NPZ/CSV/PNG diagnostics once at the end of an epoch."""
        if self.accelerator is None or self.model is None:
            return

        policy = self.accelerator.unwrap_model(self.model)
        if not isinstance(policy, HistoryDenoisingPolicy):
            return

        denoising_network = policy.denoising_network
        # Defensive support for a model that was compiled before recording was
        # enabled.  New recording runs intentionally skip torch.compile.
        if hasattr(denoising_network, "_orig_mod"):
            denoising_network = denoising_network._orig_mod
        if not hasattr(denoising_network, "pop_history_attention_epoch_stats"):
            return

        # Pop on every rank so a future epoch never mixes with this one.  The
        # moments are then merged across ranks; only rank zero writes files.
        torch_stats = denoising_network.pop_history_attention_epoch_stats()
        torch_stats = self._merge_attention_stats_across_processes(torch_stats)
        if not torch_stats or not self.accelerator.is_main_process:
            return

        stats: dict[str, dict[str, np.ndarray]] = {
            metric_name: {
                stat_name: tensor.detach().cpu().numpy()
                for stat_name, tensor in metric.items()
            }
            for metric_name, metric in torch_stats.items()
        }

        epoch_dir = os.path.join(
            self.output_dir, "attention_stats", f"epoch_{self.epoch:03d}"
        )
        os.makedirs(epoch_dir, exist_ok=True)

        flat_arrays = {
            f"{metric_name}_{stat_name}": array
            for metric_name, metric in stats.items()
            for stat_name, array in metric.items()
        }
        np.savez_compressed(
            os.path.join(epoch_dir, "history_attention_stats.npz"),
            **flat_arrays,
        )

        slot_metric_names = [
            "slot_logit",
            "slot_log_probability",
            "slot_attention_weight",
        ]
        combined = {
            name: self._combine_attention_layers(stats[name])
            for name in slot_metric_names
        }
        history_len = int(combined["slot_logit"]["mean"].shape[0])
        relative_ages = np.arange(-history_len, 0)

        frame_combined = self._combine_attention_layers(stats["frame_logit"])
        frame_token_num = int(frame_combined["mean"].shape[-1])
        history_token_labels: list[str] = []
        if getattr(denoising_network, "include_action_history", False):
            history_token_labels.extend(
                [
                    f"history_action_{idx}"
                    for idx in range(
                        int(denoising_network.history_action_num_per_chunk)
                    )
                ]
            )
        history_encoder = getattr(policy, "history_img_feature_encoder", None)
        image_names = (
            list(history_encoder.data_entry_names)
            if history_encoder is not None
            else []
        )
        remaining_token_num = frame_token_num - len(history_token_labels)
        if remaining_token_num == len(image_names):
            history_token_labels.extend(image_names)
        else:
            history_token_labels.extend(
                [
                    f"history_image_token_{idx}"
                    for idx in range(remaining_token_num)
                ]
            )
        if len(history_token_labels) != frame_token_num:
            history_token_labels = [
                f"history_token_{idx}" for idx in range(frame_token_num)
            ]

        summary_path = os.path.join(epoch_dir, "slot_summary.csv")
        with open(summary_path, "w", newline="") as csv_file:
            fieldnames = [
                "slot_index",
                "relative_age",
                "valid_count",
                "slot_logit_mean",
                "slot_logit_std",
                "slot_logit_min",
                "slot_logit_max",
                "slot_log_probability_mean",
                "slot_attention_weight_mean",
            ] + [f"{label}_logit_mean" for label in history_token_labels]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for slot_idx, relative_age in enumerate(relative_ages):
                row: dict[str, float | int] = {
                    "slot_index": slot_idx,
                    "relative_age": int(relative_age),
                    "valid_count": int(combined["slot_logit"]["count"][slot_idx]),
                    "slot_logit_mean": float(combined["slot_logit"]["mean"][slot_idx]),
                    "slot_logit_std": float(combined["slot_logit"]["std"][slot_idx]),
                    "slot_logit_min": float(combined["slot_logit"]["min"][slot_idx]),
                    "slot_logit_max": float(combined["slot_logit"]["max"][slot_idx]),
                    "slot_log_probability_mean": float(
                        combined["slot_log_probability"]["mean"][slot_idx]
                    ),
                    "slot_attention_weight_mean": float(
                        combined["slot_attention_weight"]["mean"][slot_idx]
                    ),
                }
                for token_idx, label in enumerate(history_token_labels):
                    row[f"{label}_logit_mean"] = float(
                        frame_combined["mean"][slot_idx, token_idx]
                    )
                writer.writerow(row)

        layer_summary_path = os.path.join(epoch_dir, "layer_slot_summary.csv")
        with open(layer_summary_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "layer",
                    "slot_index",
                    "relative_age",
                    "valid_count",
                    "slot_logit_mean",
                    "slot_logit_std",
                    "slot_log_probability_mean",
                    "slot_attention_weight_mean",
                ]
            )
            layer_num = stats["slot_logit"]["mean"].shape[0]
            for layer_idx in range(layer_num):
                for slot_idx, relative_age in enumerate(relative_ages):
                    writer.writerow(
                        [
                            layer_idx,
                            slot_idx,
                            int(relative_age),
                            int(stats["slot_logit"]["count"][layer_idx, slot_idx]),
                            float(stats["slot_logit"]["mean"][layer_idx, slot_idx]),
                            float(stats["slot_logit"]["std"][layer_idx, slot_idx]),
                            float(
                                stats["slot_log_probability"]["mean"][
                                    layer_idx, slot_idx
                                ]
                            ),
                            float(
                                stats["slot_attention_weight"]["mean"][
                                    layer_idx, slot_idx
                                ]
                            ),
                        ]
                    )

        # Matplotlib is imported only once per epoch on rank zero and never
        # participates in training or SwanLab logging.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 13), constrained_layout=True)
        slot_logit = combined["slot_logit"]
        axes[0].plot(relative_ages, slot_logit["mean"], marker="o", label="mean")
        axes[0].fill_between(
            relative_ages,
            slot_logit["mean"] - slot_logit["std"],
            slot_logit["mean"] + slot_logit["std"],
            alpha=0.2,
            label="mean +/- std",
        )
        axes[0].set_title("Raw slot attention logits (all layers)")
        axes[0].set_xlabel("Relative history slot (t + age)")
        axes[0].set_ylabel("Logit")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        image = axes[1].imshow(
            stats["slot_logit"]["mean"], aspect="auto", origin="lower"
        )
        axes[1].set_title("Mean raw slot logit by transformer layer")
        axes[1].set_ylabel("Transformer layer")
        axes[1].set_xlabel("Relative history slot")
        axes[1].set_xticks(np.arange(history_len))
        axes[1].set_xticklabels(relative_ages, rotation=90)
        fig.colorbar(image, ax=axes[1], label="logit")

        image = axes[2].imshow(
            stats["slot_attention_weight"]["mean"],
            aspect="auto",
            origin="lower",
            vmin=0,
        )
        axes[2].set_title("Mean slot attention probability by transformer layer")
        axes[2].set_ylabel("Transformer layer")
        axes[2].set_xlabel("Relative history slot")
        axes[2].set_xticks(np.arange(history_len))
        axes[2].set_xticklabels(relative_ages, rotation=90)
        fig.colorbar(image, ax=axes[2], label="probability")
        fig.savefig(os.path.join(epoch_dir, "slot_attention_overview.png"), dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(
            frame_token_num,
            1,
            figsize=(14, max(4, 4 * frame_token_num)),
            constrained_layout=True,
            squeeze=False,
        )
        for token_idx, label in enumerate(history_token_labels):
            axis = axes[token_idx, 0]
            image = axis.imshow(
                stats["frame_logit"]["mean"][:, :, token_idx],
                aspect="auto",
                origin="lower",
            )
            axis.set_title(f"Mean raw history-token logit: {label}")
            axis.set_ylabel("Transformer layer")
            axis.set_xlabel("Relative history slot")
            axis.set_xticks(np.arange(history_len))
            axis.set_xticklabels(relative_ages, rotation=90)
            fig.colorbar(image, ax=axis, label="logit")
        fig.savefig(os.path.join(epoch_dir, "frame_token_logits.png"), dpi=180)
        plt.close(fig)

        print(f"History attention statistics saved to {epoch_dir}")

    def compute_loss(self, batch: batch_type) -> tuple[torch.Tensor, dict[str, float]]:
        assert self.model is not None

        # import cv2
        # import numpy as np
        # img = batch["third_person_camera"][0, 0].detach().cpu().numpy()
        # img = img.transpose(1, 2, 0)
        # img = img * 255.0
        # img = img.astype(np.uint8)
        # cv2.imwrite("debug_img.png", img)
        # exit()

        # print(f"{batch['traj_idx']=}")

        loss_dict = self.model(batch)
        detached_loss_dict: dict[str, float] = {}
        detached_loss_dict["train/action_loss"] = float(loss_dict["action"].detach().cpu())
        loss = loss_dict["action"]  # Will be replaced by other weighted sum

        if "prediction_error_queue_stats" in loss_dict:
            queue_stats = loss_dict["prediction_error_queue_stats"]
            self._record_prediction_error_queue_stats(queue_stats)

        if "prediction_error_slot_weight_table" in loss_dict:
            self._record_slot_weight_table(
                loss_dict["prediction_error_slot_weight_table"]
            )

        if "memory_gate_val" in loss_dict:  
            loss += loss_dict["memory_gate_val"] * self.memory_gate_loss_weight
            detached_loss_dict["train/memory_gate_val"] = float(
                loss_dict["memory_gate_val"].detach().cpu()
            )
            detached_loss_dict["train/binary_memory_gate_val"] = float(
                loss_dict["binary_memory_gate_val"].detach().cpu()
            )

        if "future_prediction" in loss_dict:
            loss += (
                loss_dict["future_prediction"]
                * self.future_prediction_loss_weight
            )
            detached_loss_dict["train/future_prediction_loss"] = float(
                loss_dict["future_prediction"].detach().cpu()
            )

        if "critical_action" in loss_dict:
            idx = min(self.epoch, len(self.critical_action_loss_weights) - 1)
            try:
                loss += (
                    loss_dict["critical_action"]
                    * self.critical_action_loss_weights[idx]
                )
                detached_loss_dict["train/critical_action_loss"] = float(
                    loss_dict["critical_action"].detach().cpu()
                )
            except:
                print(f"{loss=}")
                print(f"{loss_dict['critical_action']=}")
                raise

        if "critical_memory_gate_val" in loss_dict:
            # Will not add to loss, but will log it
            detached_loss_dict["train/critical_memory_gate_val"] = float(
                loss_dict["critical_memory_gate_val"].detach().cpu()
            )
            detached_loss_dict["train/critical_binary_memory_gate_val"] = float(
                loss_dict["critical_binary_memory_gate_val"].detach().cpu()
            )

        detached_loss_dict["train/loss"] = float(loss.detach().cpu())

        return loss, detached_loss_dict

    def backward_streaming_batch(
        self, batch: batch_type
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run one phase stream with a persistent bank and bounded graphs."""

        assert self.model is not None
        assert self.accelerator is not None
        padding = batch.get("entire_traj_is_padding")
        if not isinstance(padding, torch.Tensor) or padding.ndim != 2:
            raise ValueError(
                "streaming batches require entire_traj_is_padding with "
                "shape [batch, sequence]"
            )
        batch_size, sequence_length = padding.shape
        valid_anchor_mask = ~padding.bool()
        total_valid_anchor_num = int(valid_anchor_mask.sum().item())
        if total_valid_anchor_num <= 0:
            raise ValueError("streaming batch contains no valid anchors")
        valid_columns = torch.nonzero(
            valid_anchor_mask.any(dim=0), as_tuple=False
        ).flatten()
        local_effective_sequence_length = int(valid_columns[-1].item()) + 1
        # Every DDP rank must execute exactly the same sequence of forward/
        # backward calls.  Streaming samples have different valid lengths even
        # though their tensors are padded to one fixed sequence dimension.  A
        # rank-local effective length therefore lets short ranks skip trailing
        # chunks while long ranks enter another DDP collective, eventually
        # deadlocking NCCL.  Gather the scalar lengths and use the maximum on
        # every rank; locally empty chunks still run with a differentiable zero
        # loss below.
        effective_sequence_length = int(
            self.accelerator.gather(
                torch.tensor(
                    [local_effective_sequence_length],
                    dtype=torch.long,
                    device=padding.device,
                )
            ).max().item()
        )
        if effective_sequence_length > sequence_length:
            raise RuntimeError(
                "global streaming length exceeds the padded sequence length: "
                f"{effective_sequence_length} > {sequence_length}"
            )

        assert self.optimizer is not None
        assert self.lr_scheduler is not None
        assert self.ema_model is not None

        total_loss = torch.zeros((), device=padding.device)
        detached_totals: dict[str, float] = {}
        grad_norm_total = 0.0
        clipped_grad_norm_total = 0.0
        grad_norm_record_num = 0
        streaming_step_count = 0

        chunk_specs: list[tuple[int, int, int]] = []
        for start in range(
            0, effective_sequence_length, self.streaming_chunk_length
        ):
            end = min(
                start + self.streaming_chunk_length,
                effective_sequence_length,
            )
            chunk_valid_num = int(
                valid_anchor_mask[:, start:end].sum().item()
            )
            # Keep zero-valid local chunks. Other ranks may still contain valid
            # anchors here, and all ranks must make the same DDP calls.
            chunk_specs.append((start, end, chunk_valid_num))

        for group_start in range(
            0, len(chunk_specs), self.streaming_chunk_accumulate
        ):
            chunk_group = chunk_specs[
                group_start : group_start + self.streaming_chunk_accumulate
            ]
            local_group_valid_num = sum(spec[2] for spec in chunk_group)
            global_group_valid_num = self.accelerator.reduce(
                torch.tensor(
                    float(local_group_valid_num),
                    dtype=torch.float32,
                    device=padding.device,
                ),
                reduction="sum",
            ).clamp_min(1.0)

            # The memory state remains detached at every chunk boundary. Only
            # parameter gradients are accumulated, so peak activation memory is
            # still bounded by ``streaming_chunk_length`` rather than by the
            # number of chunks in this optimizer group.
            with self.accelerator.accumulate(self.model):
                for chunk_idx, (start, end, chunk_valid_num) in enumerate(
                    chunk_group
                ):
                    # The final encoded anchor is a target-only look-ahead
                    # whenever it exists. It is re-encoded as the first
                    # supervised anchor of the following chunk, while the bank
                    # itself persists across calls.
                    input_stop = min(end + 1, sequence_length)
                    chunk = self._slice_streaming_batch(
                        batch,
                        start,
                        input_stop,
                        batch_size,
                        sequence_length,
                    )
                    chunk["_streaming_supervised_length"] = end - start
                    chunk["_streaming_reset"] = start == 0
                    chunk["_streaming_final"] = (
                        end == effective_sequence_length
                    )

                    is_last_chunk_in_group = (
                        chunk_idx == len(chunk_group) - 1
                    )
                    sync_context = (
                        contextlib.nullcontext()
                        if is_last_chunk_in_group
                        else self.accelerator.no_sync(self.model)
                    )
                    # DDP averages gradients across ranks. Scale each local
                    # mean by world_size * local_count / global_count to obtain
                    # the true global mean over valid anchors. For an empty
                    # local chunk this is zero, but backward still runs and
                    # participates in the same collectives as every other rank.
                    backward_scale = (
                        self.accelerator.num_processes
                        * chunk_valid_num
                        / float(global_group_valid_num.item())
                    )
                    with sync_context:
                        with self.accelerator.autocast():
                            chunk_loss, chunk_detached = self.compute_loss(
                                chunk
                            )
                            scaled_chunk_loss = (
                                chunk_loss * backward_scale
                            )
                        self.accelerator.backward(scaled_chunk_loss)

                    loss_scale = (
                        chunk_valid_num / total_valid_anchor_num
                    )
                    total_loss = (
                        total_loss + chunk_loss.detach() * loss_scale
                    )
                    for key, value in chunk_detached.items():
                        detached_totals[key] = (
                            detached_totals.get(key, 0.0)
                            + value * loss_scale
                        )

                if (
                    self.accelerator.sync_gradients
                    and len(self.clip_grad_norms) > 0
                ):
                    if self.epoch >= len(self.clip_grad_norms):
                        max_norm = self.clip_grad_norms[-1]
                    else:
                        max_norm = self.clip_grad_norms[self.epoch]
                    if max_norm == 0:
                        max_norm = float("inf")
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), max_norm=max_norm
                    )
                    grad_norm_clipped = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), max_norm=max_norm
                    )
                    assert isinstance(grad_norm, torch.Tensor)
                    assert isinstance(grad_norm_clipped, torch.Tensor)
                    grad_norm_total += float(grad_norm.item())
                    clipped_grad_norm_total += float(
                        grad_norm_clipped.item()
                    )
                    grad_norm_record_num += 1

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

                if self.accelerator.sync_gradients:
                    if self.accelerator.is_main_process:
                        self.ema_model.step(
                            self.accelerator.unwrap_model(self.model)
                        )

            streaming_step_count += 1

        if not detached_totals:
            raise RuntimeError("streaming training produced no chunk losses")
        self._last_streaming_step_count = streaming_step_count
        detached_totals["train/loss"] = float(total_loss.cpu())
        if grad_norm_record_num > 0:
            detached_totals["info/grad_norm"] = (
                grad_norm_total / grad_norm_record_num
            )
            detached_totals["info/clipped_grad_norm"] = (
                clipped_grad_norm_total / grad_norm_record_num
            )
        return total_loss, detached_totals

    def backward_prefix_batch(
        self, batch: batch_type
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Backpropagate each anchor, then update once for the whole prefix."""

        assert self.model is not None
        assert self.accelerator is not None
        assert self.optimizer is not None
        assert self.lr_scheduler is not None
        assert self.ema_model is not None

        supervised_mask = batch.get("prefix_supervised_mask")
        feature_valid = batch.get("prefix_feature_valid")
        if (
            not isinstance(supervised_mask, torch.Tensor)
            or supervised_mask.ndim != 2
            or not isinstance(feature_valid, torch.Tensor)
            or feature_valid.shape != supervised_mask.shape
        ):
            raise ValueError(
                "prefix batches require prefix_supervised_mask and "
                "prefix_feature_valid with shape [batch, sequence]"
            )
        batch_size, local_sequence_length = supervised_mask.shape
        valid_columns = torch.nonzero(
            supervised_mask.bool().any(dim=0), as_tuple=False
        ).flatten()
        if valid_columns.numel() == 0:
            raise ValueError("prefix batch contains no supervised anchor")
        local_anchor_num = int(valid_columns[-1].item()) + 1

        local_anchor_tensor = torch.tensor(
            local_anchor_num,
            dtype=torch.long,
            device=supervised_mask.device,
        )
        # Accelerate 0.x silently treats unknown reductions as SUM, so do not
        # pass ``reduction='max'`` here. Gather the one scalar per rank and
        # compute the maximum explicitly.
        global_anchor_num = int(
            self.accelerator.gather(
                local_anchor_tensor.reshape(1)
            ).max().item()
        )
        required_sequence_length = global_anchor_num + 1

        def pad_value(value: Any, key: str) -> Any:
            if isinstance(value, dict):
                return {
                    nested_key: pad_value(nested_value, nested_key)
                    for nested_key, nested_value in value.items()
                }
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim < 2
                or value.shape[0] != batch_size
                or value.shape[1] != local_sequence_length
                or local_sequence_length >= required_sequence_length
            ):
                return value
            pad_length = required_sequence_length - local_sequence_length
            pad_shape = (
                batch_size,
                pad_length,
                *value.shape[2:],
            )
            if key == "entire_traj_is_padding":
                padding = torch.ones(
                    pad_shape, dtype=torch.bool, device=value.device
                )
            elif key == "traj_idx":
                # Keep DDP-only padding compatible with absolute frame-time
                # encoding.  These repeated indices are placeholders only:
                # the corresponding supervised/feature masks remain false.
                padding = value[:, -1:].expand(pad_shape).clone()
            else:
                padding = torch.zeros(
                    pad_shape, dtype=value.dtype, device=value.device
                )
            return torch.cat([value, padding], dim=1)

        if local_sequence_length < required_sequence_length:
            batch = cast(
                batch_type,
                {
                    key: pad_value(value, key)
                    for key, value in batch.items()
                },
            )
            supervised_mask = cast(
                torch.Tensor, batch["prefix_supervised_mask"]
            )

        local_valid_num = supervised_mask[:, :global_anchor_num].sum()
        global_valid_num = self.accelerator.reduce(
            local_valid_num.to(dtype=torch.float32), reduction="sum"
        ).clamp_min(1.0)

        total_loss = torch.zeros((), device=supervised_mask.device)
        detached_totals: dict[str, float] = {}

        with self.accelerator.accumulate(self.model):
            for anchor_idx in range(global_anchor_num):
                anchor_batch = self._slice_streaming_batch(
                    batch,
                    anchor_idx,
                    anchor_idx + 2,
                    batch_size,
                    required_sequence_length,
                )
                anchor_batch["_prefix_reset"] = anchor_idx == 0
                anchor_batch["_prefix_final"] = (
                    anchor_idx == global_anchor_num - 1
                )

                local_current_valid = int(
                    supervised_mask[:, anchor_idx].sum().item()
                )
                # DDP averages gradients across ranks. Multiplying by world
                # size yields a true global mean over all valid anchor rows.
                backward_scale = (
                    self.accelerator.num_processes
                    * local_current_valid
                    / float(global_valid_num.item())
                )
                local_log_scale = local_current_valid / max(
                    1, int(local_valid_num.item())
                )

                is_last_anchor = anchor_idx == global_anchor_num - 1
                sync_context = (
                    contextlib.nullcontext()
                    if is_last_anchor
                    else self.accelerator.no_sync(self.model)
                )
                with sync_context:
                    with self.accelerator.autocast():
                        anchor_loss, anchor_detached = self.compute_loss(
                            anchor_batch
                        )
                        scaled_loss = anchor_loss * backward_scale
                    self.accelerator.backward(scaled_loss)

                total_loss = (
                    total_loss
                    + anchor_loss.detach() * local_log_scale
                )
                for key, value in anchor_detached.items():
                    detached_totals[key] = (
                        detached_totals.get(key, 0.0)
                        + value * local_log_scale
                    )

            if self.accelerator.sync_gradients and self.clip_grad_norms:
                max_norm = (
                    self.clip_grad_norms[-1]
                    if self.epoch >= len(self.clip_grad_norms)
                    else self.clip_grad_norms[self.epoch]
                )
                if max_norm == 0:
                    max_norm = float("inf")
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), max_norm=max_norm
                )
                grad_norm_clipped = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), max_norm=max_norm
                )
                detached_totals["info/grad_norm"] = float(
                    grad_norm.item()
                )
                detached_totals["info/clipped_grad_norm"] = float(
                    grad_norm_clipped.item()
                )

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()
            if (
                self.accelerator.sync_gradients
                and self.accelerator.is_main_process
            ):
                self.ema_model.step(
                    self.accelerator.unwrap_model(self.model)
                )

        if not detached_totals:
            raise RuntimeError("prefix training produced no anchor losses")
        detached_totals["train/loss"] = float(total_loss.cpu())
        return total_loss, detached_totals

    def eval_model_step(
        self, step_log: dict[str, Any], train_sampling_batch: batch_type
    ):

        self._save_history_attention_epoch_stats()

        if self.epoch == self.num_epochs - 1:
            # There are some unknown bugs in the last epoch of validation
            return

        assert self.ema_model is not None
        assert self.val_dataloader is not None
        assert self.accelerator is not None

        eval_policy = self.ema_model.averaged_model
        assert isinstance(eval_policy, BasePolicy)

        def log_action_mse(
            step_log: "dict[str, Any]",
            category: str,
            pred_action: "dict[str, torch.Tensor]",
            gt_action: "dict[str, torch.Tensor]",
            accelerator: Accelerator,
            entire_traj_is_padding: torch.Tensor | None,  # (batch_size, traj_num)
            traj_is_error: torch.Tensor | None,  # (batch_size, traj_num)
            traj_is_critical: torch.Tensor | None,  # (batch_size, traj_num)
        ):
            single_gpu_stats: dict[str, Any] = {}
            # print(f"{entire_traj_is_padding=}, {traj_is_error=}, {traj_is_critical=}")
            for key in pred_action.keys():
                assert (
                    pred_action[key].shape == gt_action[key].shape
                ), f"{pred_action[key].shape=}, {gt_action[key].shape=}"

                mse = F.mse_loss(pred_action[key], gt_action[key], reduction="none")
                mse = mse.mean(
                    dim=(-2, -1)
                )  # Single-traj: (batch_size, ), Multi-traj: (batch_size, traj_num)

                if (
                    entire_traj_is_padding is not None
                ):  # For multi-trajectory evaluation
                    assert entire_traj_is_padding.shape == mse.shape
                    mse = mse * (~entire_traj_is_padding)

                if traj_is_error is not None:
                    assert traj_is_error.shape == mse.shape, f"{traj_is_error.shape=}, {mse.shape=}"
                    mse = mse * (~traj_is_error)

                if traj_is_critical is not None:
                    assert traj_is_critical.shape == mse.shape
                    critical_mse = mse * traj_is_critical
                    assert (mse != 0).sum() > 0, f"All {key} are 0"
                    if (critical_mse != 0).sum() == 0:
                        critical_mse = torch.zeros(1, device=critical_mse.device)
                    else:
                        critical_mse = critical_mse.sum() / (critical_mse != 0).sum()
                    single_gpu_stats[f"{category}/{key}_critical_mse"] = critical_mse

                mse = mse.sum() / (mse != 0).sum()
                single_gpu_stats[f"{category}/{key}_mse"] = mse
                single_gpu_stats[f"{category}/{key}_shakiness"] = torch.mean(
                    get_shakiness_score_torch(pred_action[key])
                )

            if accelerator.num_processes > 1:
                gathered_stats: dict[str, torch.Tensor] = {}
                for key in single_gpu_stats.keys():
                    gathered_stat = accelerator.gather(single_gpu_stats[key])
                    assert isinstance(gathered_stat, torch.Tensor)
                    gathered_stats[key] = torch.mean(gathered_stat, dim=0)

                if accelerator.is_main_process:
                    step_log.update(gathered_stats)
            else:
                step_log.update(single_gpu_stats)

        def get_traj_flags(
            batch: batch_type,
        ) -> "dict[str, torch.Tensor | None]":
            if "action_is_critical" in batch:
                traj_is_critical = batch["action_is_critical"].squeeze(-1).any(dim=-1) # (batch_size, traj_num) or (batch_size, )
            else:
                traj_is_critical = None

            if "action_is_error" in batch and isinstance(eval_policy, HistoryDenoisingPolicy):
                action_key_names = eval_policy.action_decoder.data_entry_names
                action_traj_length = batch[action_key_names[0]].shape[-2]
                traj_error_mask = torch.zeros(
                    action_traj_length,
                    device=batch["action_is_error"].device,
                )
                traj_error_mask[
                    cast(HistoryDenoisingPolicy, eval_policy)
                    .action_no_error_range[0] : cast(HistoryDenoisingPolicy, eval_policy)
                    .action_no_error_range[1]
                ] = 1
                traj_is_error = batch["action_is_error"].squeeze(-1) * traj_error_mask
                traj_is_error = traj_is_error.any(dim=-1) # (batch_size, traj_num) or (batch_size, )
            else:
                traj_is_error = None

            if "entire_traj_is_padding" in batch:
                entire_traj_is_padding = batch["entire_traj_is_padding"]
            else:
                entire_traj_is_padding = None

            if self.debug:
                print(
                    f"entire_traj_is_padding: {entire_traj_is_padding is not None}, traj_is_error: {traj_is_error is not None}, traj_is_critical: {traj_is_critical is not None}"
                )

            return {
                "entire_traj_is_padding": entire_traj_is_padding, # (batch_size, traj_num) or (batch_size, ) or None
                "traj_is_error": traj_is_error, # (batch_size, traj_num) or (batch_size, ) or None
                "traj_is_critical": traj_is_critical, # (batch_size, traj_num) or (batch_size, ) or None
            }

        if self.sample_every != 0 and (self.epoch % self.sample_every) == 0 and not self.debug:
            with torch.no_grad():
                flags = get_traj_flags(train_sampling_batch)

                gt_action = {}
                action_key_names = eval_policy.action_decoder.data_entry_names
                for key in action_key_names:
                    gt_action[key] = train_sampling_batch.pop(key)
                
                pred_action = eval_policy.predict_action(train_sampling_batch)

                log_action_mse(
                    step_log, "train_sample", pred_action, gt_action, self.accelerator, **flags
                )

        if self.val_every != 0 and (self.epoch % self.val_every) == 0:
            # The last epoch of validation is buggy. Will skip it

            with torch.no_grad():
                all_gt_actions: list[dict[str, torch.Tensor]] = []
                all_pred_actions: list[dict[str, torch.Tensor]] = []
                all_traj_flags: list[dict[str, torch.Tensor | None]] = (
                    []
                )  # (val_batch_num): (batch_size, traj_num)
                if hasattr(self.val_dataloader.base_dataloader.dataset, "resample_index_pool"):
                    self.val_dataloader.base_dataloader.dataset.resample_index_pool()
                for batch_idx, val_batch in enumerate(self.val_dataloader):
                    traj_flags = get_traj_flags(val_batch)
                    gt_action = {}
                    action_key_names = eval_policy.action_decoder.data_entry_names
                    for key in action_key_names:
                        gt_action[key] = val_batch.pop(key)

                    pred_action = eval_policy.predict_action(val_batch)

                    # Prefix batches are padded only to the longest sample in
                    # that batch, so their trajectory dimension may differ
                    # across validation batches.  Flatten (batch, trajectory)
                    # into one sample dimension before collecting results.
                    # The action/time dimensions then stay fixed and can be
                    # concatenated without introducing any new fake anchors.
                    reference_action = gt_action[action_key_names[0]]
                    if reference_action.ndim >= 4:
                        val_batch_size, val_traj_num = (
                            reference_action.shape[:2]
                        )

                        def flatten_traj_dim(
                            value: torch.Tensor,
                        ) -> torch.Tensor:
                            if (
                                value.ndim >= 2
                                and tuple(value.shape[:2])
                                == (val_batch_size, val_traj_num)
                            ):
                                return value.reshape(
                                    val_batch_size * val_traj_num,
                                    *value.shape[2:],
                                )
                            return value

                        gt_action = {
                            key: flatten_traj_dim(value)
                            for key, value in gt_action.items()
                        }
                        pred_action = {
                            key: flatten_traj_dim(value)
                            for key, value in pred_action.items()
                        }
                        traj_flags = {
                            key: (
                                flatten_traj_dim(value)
                                if isinstance(value, torch.Tensor)
                                else value
                            )
                            for key, value in traj_flags.items()
                        }

                    all_traj_flags.append(traj_flags)
                    all_gt_actions.append(gt_action)
                    all_pred_actions.append(pred_action)

                    if self.debug:
                        break

                gt_actions = aggregate_batch(all_gt_actions, partial(torch.cat, dim=0))
                pred_actions = aggregate_batch(
                    all_pred_actions, partial(torch.cat, dim=0)
                )

                flags: dict[str, torch.Tensor | None] = {}
                for key in all_traj_flags[0]:
                    values = [item[key] for item in all_traj_flags]
                    tensor_values = [
                        value
                        for value in values
                        if isinstance(value, torch.Tensor)
                    ]
                    if not tensor_values:
                        flags[key] = None
                    elif len(tensor_values) != len(values):
                        raise ValueError(
                            f"validation flag {key!r} is present in only "
                            "some batches"
                        )
                    else:
                        flags[key] = torch.cat(tensor_values, dim=0)

                log_action_mse(
                    step_log, "val_sample", pred_actions, gt_actions, self.accelerator, **flags
                )

    def rollout_model_step(self, step_log: dict[str, Any]):
        assert self.accelerator is not None
        if (
            self.rollout_every != 0
            and self.epoch % self.rollout_every == 0
            and self.accelerator.is_main_process
        ):
            if self.rollout_env is not None:
                self.rollout_env.start_rollout(
                    self.epoch, self.checkpoint_manager.get_last_ckpt_path()
                )
                results = self.rollout_env.fetch_results()
                print(f"Rollout results: {results}")
                if len(results) > 0:
                    # If everything works well, there should be only one result,
                    # but in case there are multiple, use the latest one
                    result = results[-1]
                    step_log["rollout/success_rate"] = result["success_rate"]
                    step_log["rollout/epoch"] = result["epoch"]

    def eval_model(
        self, rounds: int, dataloader_names: list[str], ema_only: bool = True, use_episode_num: int = -1
    ):

        for dataloader_name in dataloader_names:
            assert dataloader_name in ["train", "val"]

        assert (
            self.cfg_str_unresolved is not None
            and self.model is not None
            and self.train_dataloader is not None
            and self.val_dataloader is not None
            and self.ema_model is not None
            and self.accelerator is not None
        )

        # To make it compatible with accelerate multi-gpu evaluation
        self.model.forward = self.model.predict_action
        self.ema_model.averaged_model.forward = (
            self.ema_model.averaged_model.predict_action
        )

        (
            self.train_dataloader,
            self.val_dataloader,
            self.model,
        ) = self.accelerator.prepare(
            self.train_dataloader,
            self.val_dataloader,
            self.model,
        )

        assert (
            self.train_dataloader is not None
            and self.val_dataloader is not None
            and self.model is not None
            and self.ema_model is not None
        )

        device: torch.device = self.model.device
        self.ema_model.to(device)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
        self.model.requires_grad_(False)
        self.model.eval()

        dataloader_dict = {
            "val": self.val_dataloader,
            "train": self.train_dataloader,
        }

        all_results = []

        for dataloader_name in dataloader_names:
            dataloader = dataloader_dict[dataloader_name]
            if use_episode_num != -1:
                dataset: BaseDataset = dataloader.base_dataloader.dataset
                dataset.trim_dataset_episodes(remaining_episode_num=use_episode_num)
                
            batch_results: list[dict[str, torch.Tensor]] = []
            with torch.no_grad():
                for round in range(rounds):
                    with tqdm.tqdm(
                        dataloader,
                        desc=f"Eval {dataloader_name}, round {round+1}/{rounds}",
                        leave=False,
                        mininterval=self.tqdm_interval_sec,
                        ncols=70,
                    ) as tepoch:
                        for i, batch in enumerate(tepoch):

                            gt_action = {}
                            for key, value in batch.items():
                                if key.startswith("action"):
                                    gt_action[key] = value

                            ema_pred_action = self.ema_model.averaged_model(batch)

                            batch_result: dict[str, torch.Tensor] = {}
                            # Calculate MSE errors for each key

                            if not ema_only:
                                policy_pred_action = self.model(batch)
                                for key in policy_pred_action.keys():
                                    policy_result_key = f"policy_{key}"
                                    assert (
                                        policy_pred_action[key].shape
                                        == gt_action[key].shape
                                    )
                                    batch_result[policy_result_key] = (
                                        policy_pred_action[key]
                                    )

                                    if (
                                        len(policy_pred_action[key].shape) == 3
                                    ):  # Single trajectory (batch_size, traj_length, action_dim)

                                        batch_result[
                                            f"{policy_result_key}_mse"
                                        ] = torch.nn.functional.mse_loss(
                                            policy_pred_action[key],
                                            gt_action[key],
                                            reduction="none",
                                        ).mean(
                                            dim=(1)
                                        )  # (batch_size, action_dim)

                                    elif (
                                        len(policy_pred_action[key].shape) == 4
                                    ):  # Multi-trajectory (batch_size, traj_num, traj_length, action_dim)
                                        entire_traj_is_padding: torch.Tensor = batch["entire_traj_is_padding"]

                                        policy_mse = torch.nn.functional.mse_loss(
                                            policy_pred_action[key],
                                            gt_action[key],
                                            reduction="none",
                                        ).mean(
                                            dim=(2)
                                        )  # (batch_size, traj_num, action_dim)
                                        policy_mse = policy_mse * (
                                            ~entire_traj_is_padding[:, :, None]
                                        )
                                        batch_result[f"{policy_result_key}_mse"] = (
                                            policy_mse
                                        )

                            for key in ema_pred_action.keys():
                                batch_result[f"gt_{key}"] = gt_action[key]
                                ema_result_key = f"ema_{key}"

                                assert (
                                    ema_pred_action[key].shape == gt_action[key].shape
                                )
                                batch_result[ema_result_key] = ema_pred_action[key]

                                if (
                                    len(ema_pred_action[key].shape) == 3
                                ):  # Single trajectory (batch_size, traj_length, action_dim)
                                    batch_result[
                                        f"{ema_result_key}_mse"
                                    ] = torch.nn.functional.mse_loss(
                                        ema_pred_action[key],
                                        gt_action[key],
                                        reduction="none",
                                    ).mean(
                                        dim=(1)
                                    )  # (batch_size, action_dim)

                                elif (
                                    len(ema_pred_action[key].shape) == 4
                                ):  # Multi-trajectory (batch_size, traj_num, traj_length, action_dim)
                                    entire_traj_is_padding: torch.Tensor = batch["entire_traj_is_padding"]
                                    batch_result["entire_traj_is_padding"] = entire_traj_is_padding
                                    ema_mse = torch.nn.functional.mse_loss(
                                        ema_pred_action[key],
                                        gt_action[key],
                                        reduction="none",
                                    ).mean(
                                        dim=(2)
                                    )  # (batch_size, traj_num, action_dim)
                                    ema_mse = ema_mse * (
                                        ~entire_traj_is_padding[:, :, None]
                                    )
                                    batch_result[f"{ema_result_key}_mse"] = ema_mse

                            batch_result["traj_idx"] = batch["traj_idx"]
                            batch_result["episode_idx"] = batch["episode_idx"]
                            batch_results.append(copy.deepcopy(batch_result))

            # Aggregate results accross gpus
            gathered_results = []

            for batch in batch_results:
                gathered_batch = {
                    key: self.accelerator.gather(val) for key, val in batch.items()
                }
                gathered_results.append(gathered_batch)

            if self.accelerator.is_main_process:
                results: dict[str, torch.Tensor] = aggregate_batch(
                    gathered_results, partial(torch.cat, dim=0)
                )
                results = {key: val.cpu() for key, val in results.items()}
                file_name = f"{dataloader_name}_results.pt"
                torch_save(
                    results,
                    os.path.join(
                        self.output_dir,
                        f"epoch_{self.epoch-1}_eval",
                        file_name,
                    ),
                )
                print(
                    f"Evaluation results saved to {os.path.join(self.output_dir, f'epoch_{self.epoch-1}_eval', file_name)}"
                )
                all_results.append(copy.deepcopy(results))
                
        if self.accelerator.is_main_process and len(dataloader_names) > 1:
            all_results = aggregate_batch(all_results, partial(torch.cat, dim=0))
            file_name = f"all_results.pt"
            torch_save(
                all_results,
                os.path.join(
                    self.output_dir, f"epoch_{self.epoch-1}_eval", file_name
                ),
            )
            print(
                f"All evaluation results saved to {os.path.join(self.output_dir, f'epoch_{self.epoch-1}_eval', file_name)}"
            )

        if self.accelerator.num_processes > 1:
            self.accelerator.wait_for_everyone()
