import copy
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Mapping, cast

import einops
import torch
import torch.nn.functional as F

from imitation_learning.common.datatypes import batch_type
from imitation_learning.models.common.memory_gate import \
    MemoryGate
from imitation_learning.models.denoising_networks.memory_transformer import \
    MemoryTransformer
from imitation_learning.models.encoders.multi_token_encoder import \
    MultiTokenEncoder
from imitation_learning.models.future_prediction.future_feature_predictor import (
    FutureFeaturePredictor)
from imitation_learning.models.memory.dynamic_memory_weight import (
    DynamicMemoryWeightRecord, DynamicMemoryWeightUpdateConfig,
    apply_dynamic_memory_weight_update, select_memory_slot_eviction_index,
    select_random_training_slot_eviction_index)
from imitation_learning.models.memory.prediction_error_queue import (
    PredictionErrorQueue, PredictionErrorQueueConfig, PredictionErrorRecord,
    apply_recent_slot_attention_floor)
from imitation_learning.models.memory.slot_weight_table import (
    build_active_slot_weight_table)
from imitation_learning.policies.base_denoising_policy import BaseDenoisingPolicy
from torch._dynamo.eval_frame import OptimizedModule
from robot_utils.data_utils import dict_apply
from robot_utils.torch_utils import aggregate_batch, split_batch

import cv2
import numpy as np
import time


@dataclass
class PendingDynamicMemoryUpdate:
    """Detached inference inputs needed to rebuild d(error)/d(memory weight)."""

    retrieval_inputs: dict[str, torch.Tensor]
    stored_slot_weights: torch.Tensor
    history_slot_ids: tuple[int, ...]
    history_frame_indices: tuple[int, ...]
    action_condition: torch.Tensor | None


@dataclass
class StreamingTrainingState:
    """Detached causal bank persisted between bounded training chunks."""

    history_latents: list[list[torch.Tensor]]
    history_frame_indices: list[list[int]]
    history_weights: list[list[torch.Tensor]] | None
    next_slot_weights: list[torch.Tensor] | None
    queues: list[PredictionErrorQueue] | None


@dataclass
class PrefixTrainingState:
    """Raw detached slot inputs retained across per-anchor backwards."""

    history_img_features: list[list[torch.Tensor]]
    history_actions: list[list[torch.Tensor]]


@dataclass
class StreamingRawHistoryState:
    """Detached encoder outputs used to rebuild streaming history slots.

    The frozen image encoder outputs are the persistent state. Projected
    latents are rebuilt at the start of every chunk so later-chunk losses can
    still train the shared history/action projectors without retaining an
    autograd graph across optimizer steps.
    """

    history_img_features: list[list[torch.Tensor]]
    history_actions: list[list[torch.Tensor]]


class HistoryDenoisingPolicy(BaseDenoisingPolicy):
    def __init__(
        self,
        skip_memory: bool,
        history_mask_max_prob: float,
        history_img_feature_encoder: MultiTokenEncoder | None,
        action_no_error_range: tuple[int, int],
        train_history_action_noise_level: str,
        eval_history_action_noise_level: str,
        history_action_num_per_chunk: int,
        future_img_feature_encoder: MultiTokenEncoder | None = None,
        future_prediction_context_pooling: str = "none",
        future_feature_predictor: (
            FutureFeaturePredictor | Callable[..., FutureFeaturePredictor] | None
        ) = None,
        prediction_error_queue: Mapping[str, Any] | None = None,
        dynamic_memory_weight_update: Mapping[str, Any] | None = None,
        attention_bias: bool = True,
        training_slot_weight: bool = True,
        compact_three_view_slots: bool = False,
        state_conditioned_single_view_slots: bool = False,
        # add_noise_to_history_img_features: bool,
        memory_gate: MemoryGate | None = None,
        max_training_traj_num: int = -1,
        **kwargs,
    ):

        global_cond_encoder = kwargs.get("global_cond_encoder")
        history_view_num = (
            len(history_img_feature_encoder.data_entry_names)
            if history_img_feature_encoder is not None
            else 0
        )
        compact_multiview_slot = (
            compact_three_view_slots and history_view_num == 3
        )
        state_conditioned_history_slot = (
            state_conditioned_single_view_slots and history_view_num == 1
        )
        if compact_multiview_slot:
            if history_img_feature_encoder is None:
                raise ValueError(
                    "compact three-view slots require a history image encoder"
                )
            if history_img_feature_encoder.token_num != history_view_num:
                raise ValueError(
                    "each compact history view must produce exactly one token, "
                    f"got {history_img_feature_encoder.token_num} tokens for "
                    f"{history_view_num} views"
                )
            if not isinstance(global_cond_encoder, MultiTokenEncoder):
                raise TypeError(
                    "compact three-view slots require MultiTokenEncoder global condition"
                )
            if (
                len(global_cond_encoder.data_entry_names) != history_view_num
                or global_cond_encoder.token_num != history_view_num
            ):
                raise ValueError(
                    "compact current/history views must be aligned one-to-one"
                )
        if state_conditioned_history_slot:
            if history_img_feature_encoder is None:
                raise ValueError(
                    "state-conditioned single-view slots require a history "
                    "image encoder"
                )
            if not isinstance(global_cond_encoder, MultiTokenEncoder):
                raise TypeError(
                    "state-conditioned single-view slots require a "
                    "MultiTokenEncoder global condition"
                )
            if global_cond_encoder.token_num != history_img_feature_encoder.token_num:
                raise ValueError(
                    "state-conditioned current/history patch counts must match"
                )

        if history_img_feature_encoder is not None:
            kwargs["denoising_network_partial"] = partial(
                kwargs["denoising_network_partial"],
                history_img_features_dim=history_img_feature_encoder.feature_dim,
                history_img_features_token_num=history_img_feature_encoder.token_num,
                history_action_num_per_chunk=history_action_num_per_chunk,
                compact_multiview_slot=compact_multiview_slot,
                state_conditioned_history_slot=(
                    state_conditioned_history_slot
                ),
                history_view_num=(history_view_num if compact_multiview_slot else 0),
            )
        if memory_gate is not None and isinstance(memory_gate, MemoryGate):
            kwargs["denoising_network_partial"] = partial(
                kwargs["denoising_network_partial"],
            )

        super().__init__(**kwargs)

        if (
            not isinstance(self.denoising_network, MemoryTransformer)
            and not skip_memory
        ):
            raise TypeError(
                "HistoryDenoisingPolicy requires MemoryTransformer"
            )
        self.history_attention_bias_mode = (
            self.denoising_network.memory_retriever.history_cross_attn
            .attention_bias_mode
            if isinstance(self.denoising_network, MemoryTransformer)
            else "additive"
        )
        self.compact_multiview_slot = compact_multiview_slot
        self.state_conditioned_history_slot = (
            state_conditioned_history_slot
        )
        self.stores_state_conditioned_slots = (
            compact_multiview_slot or state_conditioned_history_slot
        )

        if future_prediction_context_pooling not in {"none", "mean"}:
            raise ValueError(
                "future_prediction_context_pooling must be 'none' or 'mean', "
                f"got {future_prediction_context_pooling!r}"
            )
        self.future_prediction_context_pooling = (
            future_prediction_context_pooling
        )
        if future_img_feature_encoder is None:
            # Keep legacy checkpoints unchanged: the fallback is the already
            # registered history encoder, not a second nn.Module alias with a
            # duplicate state-dict prefix.
            self.__dict__["future_img_feature_encoder"] = (
                history_img_feature_encoder
            )
        else:
            self.future_img_feature_encoder = future_img_feature_encoder

        if future_feature_predictor is not None and self.future_img_feature_encoder is None:
            raise ValueError(
                "an image encoder is required for future feature targets"
            )
        if future_feature_predictor is not None and not isinstance(
            future_feature_predictor, FutureFeaturePredictor
        ):
            if not callable(future_feature_predictor):
                raise TypeError(
                    "future_feature_predictor must be a predictor or callable"
                )
            if not isinstance(self.denoising_network, MemoryTransformer):
                raise TypeError(
                    "Future prediction requires MemoryTransformer decision memory"
                )
            assert self.future_img_feature_encoder is not None
            future_feature_predictor = future_feature_predictor(
                hidden_dim=self.denoising_network.hidden_dim,
                output_feature_dim=(
                    self.denoising_network.hidden_dim
                    if self.compact_multiview_slot
                    else self.future_img_feature_encoder.feature_dim
                ),
                future_token_num=(
                    history_view_num
                    if self.compact_multiview_slot
                    else self.future_img_feature_encoder.token_num
                ),
                action_dim=self.action_decoder.latent_dim,
                max_action_length=self.action_decoder.token_num,
            )

        self.future_feature_predictor = future_feature_predictor
        if self.future_feature_predictor is not None:
            denoising_network = self.denoising_network
            if not isinstance(denoising_network, MemoryTransformer):
                raise TypeError(
                    "Future prediction requires MemoryTransformer decision memory"
                )
            if (
                self.future_feature_predictor.input_feature_dim
                != denoising_network.hidden_dim
            ):
                raise ValueError(
                    "Future predictor input_feature_dim must match "
                    "MemoryTransformer hidden_dim "
                    f"({self.future_feature_predictor.input_feature_dim} != "
                    f"{denoising_network.hidden_dim})"
                )
            assert self.future_img_feature_encoder is not None
            if self.compact_multiview_slot:
                if (
                    self.future_feature_predictor.output_feature_dim
                    != denoising_network.hidden_dim
                    or self.future_feature_predictor.future_token_num
                    != history_view_num
                ):
                    raise ValueError(
                        "compact future predictor must output one hidden-dim token "
                        "per camera view"
                    )
            else:
                if (
                    self.future_feature_predictor.output_feature_dim
                    != self.future_img_feature_encoder.feature_dim
                ):
                    raise ValueError(
                        "Future predictor output_feature_dim must match the future "
                        "target image encoder feature_dim"
                    )
                if (
                    self.future_feature_predictor.future_token_num
                    != self.future_img_feature_encoder.token_num
                ):
                    raise ValueError(
                        "Future predictor future_token_num must match the number of "
                        "future target image tokens"
                    )
            if (
                self.future_feature_predictor.use_action_condition
                and self.future_feature_predictor.action_dim
                != self.action_decoder.latent_dim
            ):
                raise ValueError(
                    "Future predictor action_dim must match action latent_dim"
                )

        if not isinstance(training_slot_weight, bool):
            raise TypeError(
                "training_slot_weight must be a bool, got "
                f"{type(training_slot_weight).__name__}"
            )
        self.training_slot_weight = training_slot_weight

        self.prediction_error_queue_config: PredictionErrorQueueConfig | None = None
        if prediction_error_queue is not None:
            self.prediction_error_queue_config = (
                PredictionErrorQueueConfig.from_mapping(prediction_error_queue)
            )
        if (
            self.prediction_error_queue_config is not None
            and self.prediction_error_queue_config.enabled
            and self.future_feature_predictor is None
        ):
            raise ValueError(
                "prediction_error_queue requires future_feature_predictor"
            )
        if (
            self.prediction_error_queue_config is not None
            and self.prediction_error_queue_config.enabled
            and max_training_traj_num > 0
        ):
            raise ValueError(
                "prediction_error_queue requires temporally ordered anchors; "
                "max_training_traj_num must be -1 because random anchor "
                "subsampling breaks the causal queue update"
            )
        if (
            self.prediction_error_queue_config is not None
            and self.prediction_error_queue_config.enabled
            and skip_memory
        ):
            raise ValueError(
                "prediction_error_queue cannot be enabled when skip_memory is True"
            )

        self.dynamic_memory_weight_update_config: (
            DynamicMemoryWeightUpdateConfig | None
        ) = None
        if not isinstance(attention_bias, bool):
            raise TypeError(
                f"attention_bias must be a bool, got {type(attention_bias).__name__}"
            )
        self.attention_bias = attention_bias
        if dynamic_memory_weight_update is not None:
            self.dynamic_memory_weight_update_config = (
                DynamicMemoryWeightUpdateConfig.from_mapping(
                    dynamic_memory_weight_update
                )
            )
        if (
            self.dynamic_memory_weight_update_config is not None
            and (
                self.dynamic_memory_weight_update_config.enabled
                or self.dynamic_memory_weight_update_config
                .evict_lowest_weight_when_full
            )
            and not self.prediction_error_queue_enabled
        ):
            raise ValueError(
                "dynamic weight update and minimum-weight eviction require an "
                "enabled prediction_error_queue"
            )
        if (
            self.dynamic_memory_weight_update_enabled
            and isinstance(self.denoising_network, MemoryTransformer)
            and self.denoising_network.skip_history_attn
        ):
            raise ValueError(
                "dynamic memory weight update requires history attention"
            )
        if (
            self.dynamic_memory_weight_update_enabled
            and isinstance(self.denoising_network, MemoryTransformer)
            and self.denoising_network.max_history_len <= 0
        ):
            raise ValueError(
                "dynamic memory weight update requires max_history_len > 0"
            )
        if (
            self.dynamic_memory_weight_update_config is not None
            and isinstance(self.denoising_network, MemoryTransformer)
            and self.dynamic_memory_weight_update_config
            .recent_slot_protection_num
            > self.denoising_network.max_history_len
        ):
            raise ValueError(
                "recent_slot_protection_num cannot exceed max_history_len, "
                f"got {self.dynamic_memory_weight_update_config.recent_slot_protection_num} "
                f"and {self.denoising_network.max_history_len}"
            )
        self.action_no_error_range: tuple[int, ...] = tuple(action_no_error_range)
        """
        If any training action between the two values is error, will not backprop the loss
        """
        assert self.action_no_error_range[1] > self.action_no_error_range[0] >= 0


        self.history_noisy_actions_dict: dict[int, list[list[torch.Tensor]]] = {}
        """
        episode_idx -> list [num_history] of lists [num_inference_steps] of tensors [noisy_history_action, shape: (action_length, action_dim)]
        history_mask_max_prob: [0, 1], higher means more history latents will be masked
        Since the action diffusion is not in latent space, we directly store the noisy action as latents
        In the future, we can also store image features/latents in the buffer
        """
        self.history_img_features_dict: dict[int, list[torch.Tensor]] = {}
        """
        episode_idx -> list [num_history] of tensors [history_img_features, shape: (history_img_features_length, img_length*history_img_features_token_num, history_img_features_dim)]
        """
        assert (
            0 <= history_mask_max_prob <= 1
        ), f"history_mask_max_prob must be in [0, 1], but got {history_mask_max_prob}"
        self.history_mask_max_prob: float = history_mask_max_prob

        if not isinstance(memory_gate, MemoryGate):
            self.memory_gate: MemoryGate | None = None
            print(f"No memory gate provided. Got {memory_gate}")
        else:
            self.memory_gate: MemoryGate | None = memory_gate

        self.skip_memory: bool = skip_memory
        """
        If True, will not pass history latents to the denoising network
        """
        if skip_memory:
            self.enable_skip_memory()

        self.recorded_data_dicts: dict[int, list[dict[str, torch.Tensor]]] = {}
        """
        episode_idx -> list of dicts
        each item in the list:
            "history_cross_attention": (diffusion_step_num, 1, head_num,
                decision_memory_token_num, history_len*history_token_num)
            "history_attention_logits": same layout before softmax
            "decision_memory": (diffusion_step_num, 1,
                decision_memory_token_num, hidden_dim)
        """
        self.history_memory_weights_dict: dict[int, list[torch.Tensor]] = {}
        """episode_idx -> retained additive memory attention biases."""
        self.history_memory_slot_ids_dict: dict[int, list[int]] = {}
        """Stable ids aligned with every retained feature/action/weight slot."""
        self.next_memory_slot_id_dict: dict[int, int] = {}
        """Next monotonically increasing online slot id for each episode."""
        self.history_frame_indices_dict: dict[int, list[int]] = {}
        """Episode-absolute frame indices aligned with every retained slot."""
        self.next_frame_index_dict: dict[int, int] = {}
        """Absolute frame index assigned to the next online observation."""
        self.latest_online_memory_query_dict: dict[int, dict[str, Any]] = {}
        """Exact retained slots and weights queried by the latest decision."""
        self.prediction_error_queues: dict[int, PredictionErrorQueue] = {}
        """Independent online prediction-error queue for every episode."""
        self.pending_future_predictions_dict: dict[int, torch.Tensor] = {}
        """Prediction made at the previous decision, resolved on observation arrival."""
        self.pending_dynamic_memory_updates_dict: dict[
            int, PendingDynamicMemoryUpdate
        ] = {}
        """Per-episode snapshots used to rebuild online weight gradients."""
        self._streaming_training_state: StreamingTrainingState | None = None
        self._streaming_raw_history_state: (
            StreamingRawHistoryState | None
        ) = None
        self._prefix_training_state: PrefixTrainingState | None = None

        self.history_img_feature_encoder: MultiTokenEncoder | None = (
            history_img_feature_encoder
        )
        self.history_storage_token_num = (
            history_view_num
            if self.compact_multiview_slot
            else (
                history_img_feature_encoder.token_num
                if history_img_feature_encoder is not None
                else 0
            )
        )
        self.history_storage_feature_dim = (
            self.denoising_network.hidden_dim
            if self.stores_state_conditioned_slots
            else (
                history_img_feature_encoder.feature_dim
                if history_img_feature_encoder is not None
                else 0
            )
        )

        if self.compact_multiview_slot:
            # Preserve all three enhanced tokens as future-prediction context.
            # The single-view path keeps the configured pooling unchanged.
            self.future_prediction_context_pooling = "none"

        self.history_action_num_per_chunk: int = history_action_num_per_chunk
        """
        Number of history actions to be stored in the buffer. Should be the number of executed actions in one chunk.
        """

        assert train_history_action_noise_level in ["last_step", "none", "random"]
        assert eval_history_action_noise_level in ["last_step", "none", "random"]
        self.train_history_action_noise_level: str = train_history_action_noise_level
        self.eval_history_action_noise_level: str = eval_history_action_noise_level
        """
        last_step: use one-step less noisy history action as condition
        none: use clean history action as condition
        random: use random noise levey history action as condition
        """


        self.max_training_traj_num: int = max_training_traj_num
        """
        Maximum number of trajectories to be used for training. If -1, will use all the trajectories. Otherwise, will sample a random subset of trajectories.
        This is used when there are too many trajectories in the dataloader (say 150+) to save memory.
        """

    def enable_skip_memory(self):
        self.skip_memory = True
        print(f"Setting skip_memory to {self.skip_memory}")
        for key, params in self.denoising_network.named_parameters():
            if "history" in key:
                if self.skip_memory:
                    params.requires_grad = False

        if self.memory_gate is not None:
            for key, params in self.memory_gate.named_parameters():
                if self.skip_memory:
                    params.requires_grad = False

    # ================================ Inference =================================

    def _encode_current_image_features(
        self,
        normalized_batch: batch_type,
        batch_size: int,
        encoder: MultiTokenEncoder | None,
        encoder_name: str,
    ) -> torch.Tensor:
        if encoder is None:
            raise RuntimeError(f"{encoder_name} is not configured")
        img_dict = {}
        for key in encoder.data_entry_names:
            if key in normalized_batch:
                img_dict[key] = normalized_batch[key]
            elif f"{key}_feature" in normalized_batch:
                img_dict[f"{key}_feature"] = normalized_batch[f"{key}_feature"]
            else:
                raise ValueError(f"Key {key} not found in normalized_batch")
        features = encoder(img_dict).reshape(
            batch_size, -1, encoder.feature_dim
        )
        expected_shape = (
            batch_size,
            encoder.token_num,
            encoder.feature_dim,
        )
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                f"{encoder_name} must produce exactly one "
                f"slot of features with shape {expected_shape}, got "
                f"{tuple(features.shape)}"
            )
        return features

    def _encode_current_history_image_features(
        self, normalized_batch: batch_type, batch_size: int
    ) -> torch.Tensor:
        return self._encode_current_image_features(
            normalized_batch,
            batch_size,
            self.history_img_feature_encoder,
            "history_img_feature_encoder",
        )

    def _encode_current_future_image_features(
        self, normalized_batch: batch_type, batch_size: int
    ) -> torch.Tensor:
        return self._encode_current_image_features(
            normalized_batch,
            batch_size,
            self.future_img_feature_encoder,
            "future_img_feature_encoder",
        )

    def _pool_future_prediction_context(
        self, context_tokens: torch.Tensor
    ) -> torch.Tensor:
        if self.future_prediction_context_pooling == "mean":
            return context_tokens.mean(dim=-2, keepdim=True)
        return context_tokens

    def _set_compact_future_targets(
        self,
        target: batch_type,
        projected_input: dict[str, torch.Tensor],
        entire_traj_is_padding: torch.Tensor,
        future_transition_valid: torch.Tensor | None = None,
    ) -> None:
        """Use the next real S_t slot as the three-token prediction target."""

        slot_features = projected_input.get("all_current_slot_features")
        if slot_features is None:
            raise KeyError(
                "compact multi-view projection is missing all_current_slot_features"
            )
        if slot_features.ndim != 4:
            raise ValueError(
                "all_current_slot_features must have shape "
                "[batch, trajectory, view, hidden]"
            )
        if slot_features.shape[2:] != (
            self.history_storage_token_num,
            self.history_storage_feature_dim,
        ):
            raise ValueError(
                "compact slot feature shape does not match configured storage: "
                f"{tuple(slot_features.shape)}"
            )

        future_features = torch.zeros_like(slot_features)
        future_features[:, :-1] = slot_features[:, 1:]
        future_valid = torch.zeros(
            slot_features.shape[:2],
            dtype=torch.bool,
            device=slot_features.device,
        )
        future_valid[:, :-1] = (
            ~entire_traj_is_padding[:, :-1]
            & ~entire_traj_is_padding[:, 1:]
        )
        future_valid = self._mask_invalid_future_transitions(
            future_valid,
            future_transition_valid,
        )
        target["future_features"] = future_features
        target["future_feature_mask"] = future_valid[:, :, None].expand(
            -1, -1, self.history_storage_token_num
        )

    @staticmethod
    def _mask_invalid_future_transitions(
        future_valid: torch.Tensor,
        future_transition_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        """Disable supervision across non-local random-pair boundaries."""

        if future_transition_valid is None:
            return future_valid
        if future_transition_valid.shape != future_valid.shape:
            raise ValueError(
                "future_transition_valid must have shape "
                f"{tuple(future_valid.shape)}, got "
                f"{tuple(future_transition_valid.shape)}"
            )
        return future_valid & future_transition_valid.to(
            device=future_valid.device,
            dtype=torch.bool,
        )

    def predict_action(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """
        Single-trajectory input:
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
                "robot0_10d": (batch_size, traj_length, 8)
                "episode_idx": (batch_size,)
            return:
                "action0_10d": (batch_size, traj_length, 8)

        Multi-trajectory input: (only available when skip_memory is False)
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size)
                "robot0_10d": (batch_size, traj_num, traj_length, 8)
            return:
                "action0_10d": (batch_size, traj_num, traj_length, 8)
        """

        meta = next(iter(self.global_cond_encoder.cond_meta.values()))
        if meta.name not in normalized_batch:
            feature_key = f"{meta.name}_feature"
            input_shape = normalized_batch[feature_key].shape
            encoder = self.global_cond_encoder.encoder_dict[meta.name]
            if getattr(encoder, "feature_aggregation", None) == "patches":
                # A full-patch cache stores [CLS, patch_0, ..., patch_N].
                # The trajectory-length dimension precedes this tail and must
                # remain part of the single-/multi-trajectory rank test below.
                expected_shape = torch.Size(
                    [encoder.token_num + 1, self.global_cond_encoder.feature_dim]
                )
            else:
                expected_shape = torch.Size(
                    [self.global_cond_encoder.feature_dim]
                )
        else:
            input_shape = normalized_batch[meta.name].shape
            expected_shape = meta.shape

        if self.skip_memory:
            assert (
                len(input_shape) - len(expected_shape) == 2
            ), "Please make sure you are using single-trajectory dataset when skip_memory is True"
            return super().predict_action(normalized_batch)
        
        if len(input_shape) - len(expected_shape) == 2:
            return self.predict_single_traj(normalized_batch)

        elif len(input_shape) - len(expected_shape) == 3:
            return self.predict_multi_traj(normalized_batch)

        else:
            raise ValueError(
                f"Unexpected input shape: {input_shape}, expected shape: {expected_shape}"
            )

    def predict_single_traj(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """
        normalized_batch:
            "robot0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
            "robot0_10d": (batch_size, traj_length, 8)
            "third_person_camera": (batch_size, traj_length, 3, image_size, image_size) # For table-bin scenario
            "episode_idx": (batch_size,)
        return:
            "action0_10d": (batch_size, traj_length, 8)
        """

        assert (
            "action" not in normalized_batch
        ), "Please exclude the batch `action` for evaluation"

        data_dict, _ = self._encode_input_add_noise(normalized_batch, mode="eval")

        if self.memory_gate is not None:
            memory_gate_val = (
                self.memory_gate.get_gate_value(normalized_batch)
            ) # (batch_size, )
            bs = memory_gate_val.shape[0]
            binarized_memory_gate_val = (memory_gate_val > 0.5).bool()

            assert isinstance(self.denoising_network, MemoryTransformer)

            if not torch.torch.torch.is_grad_enabled() \
                and self.denoising_network.binary_gating \
                and bs == 1 \
                and sum(binarized_memory_gate_val) == 0 \
                and "history_cross_attention" not in self.denoising_network.record_data_entries:
                self.denoising_network.set_skip_history_attn(True)
            else:
                self.denoising_network.set_skip_history_attn(False)

        else:
            memory_gate_val = None

        new_history_action_dict: dict[int, list[torch.Tensor]] = {}
        batch_size, traj_length, action_dim = data_dict["noisy_action"].shape
        assert batch_size == len(
            normalized_batch["episode_idx"]
        ), f"Please make sure the batch size {batch_size} in data_dict['trajectory'].shape: {data_dict['trajectory'].shape}, is the same as the number of episodes {len(normalized_batch['episode_idx'])}"


        if isinstance(self.denoising_network, OptimizedModule):
            # After torch compile: Just fix the type of the denoising network for type checking.
            self.denoising_network = cast(MemoryTransformer, cast(Any, self.denoising_network))
        else:
            assert isinstance(self.denoising_network, MemoryTransformer)
        max_history_len: int = self.denoising_network.max_history_len

        current_frame_indices = torch.empty(
            batch_size, dtype=torch.long, device=self.device
        )
        for idx, episode_idx_tensor in enumerate(
            normalized_batch["episode_idx"]
        ):
            episode_idx = int(episode_idx_tensor)
            current_frame_indices[idx] = self.next_frame_index_dict.get(
                episode_idx, 0
            )

        current_history_img_features: torch.Tensor | None = None
        current_future_img_features: torch.Tensor | None = None
        if self.stores_state_conditioned_slots and max_history_len > 0:
            current_history_img_features = (
                self.denoising_network.project_state_conditioned_slot_features(
                    global_cond=data_dict["global_cond"],
                    local_cond=data_dict.get("local_cond"),
                    global_cond_mask=data_dict.get("global_cond_mask"),
                    local_cond_mask=data_dict.get("local_cond_mask"),
                )
            )
            if self.compact_multiview_slot:
                # Compact three-view prediction targets the exact future S_t.
                current_future_img_features = current_history_img_features
            elif self.future_img_feature_encoder is not None:
                # Full-patch single-view slots still predict the compact DINO
                # CLS target used by the original prediction-error objective.
                current_future_img_features = (
                    self._encode_current_future_image_features(
                        normalized_batch, batch_size
                    )
                )
        else:
            if self.history_img_feature_encoder is not None and max_history_len > 0:
                current_history_img_features = (
                    self._encode_current_history_image_features(
                        normalized_batch, batch_size
                    )
                )
            if self.future_img_feature_encoder is not None and max_history_len > 0:
                current_future_img_features = (
                    self._encode_current_future_image_features(
                        normalized_batch, batch_size
                    )
                )

        current_slot_weights: torch.Tensor | None = None
        if self.prediction_error_queue_enabled and max_history_len > 0:
            if current_future_img_features is None:
                raise RuntimeError(
                    "prediction_error_queue requires future target features"
                )
            if self.future_feature_predictor is None:
                raise RuntimeError(
                    "prediction_error_queue requires future_feature_predictor"
                )
            current_slot_weights = torch.zeros(
                batch_size, dtype=torch.float32, device=self.device
            )
            for idx, episode_idx_tensor in enumerate(
                normalized_batch["episode_idx"]
            ):
                episode_idx = int(episode_idx_tensor)
                queue = self._get_online_prediction_error_queue(episode_idx)
                history_size = len(
                    self.history_img_features_dict.get(episode_idx, [])
                )
                if history_size == 0:
                    current_slot_weights[idx] = queue.first_slot_weight(
                        current_slot_weights[idx],
                        current_frame_indices[idx],
                    )
                    continue
                if self.dynamic_memory_weight_update_enabled:
                    self._resolve_pending_dynamic_memory_update(
                        episode_idx,
                        current_future_img_features[idx : idx + 1],
                    )
                if episode_idx not in self.pending_future_predictions_dict:
                    raise RuntimeError(
                        "Missing pending future prediction for non-empty episode "
                        f"{episode_idx}. Call reset() before reusing episode ids."
                    )
                predicted_features = self.pending_future_predictions_dict.pop(
                    episode_idx
                ).unsqueeze(0)
                current_features = current_future_img_features[idx].unsqueeze(0)
                prediction_error = (
                    self.future_feature_predictor.cosine_prediction_error(
                        predicted_features=predicted_features,
                        target_features=current_features,
                        detach_target=True,
                    ).mean()
                )
                record = queue.score_and_push(prediction_error)
                current_slot_weights[idx] = self._freeze_first_slot_weight(
                    record.initial_weight,
                    int(current_frame_indices[idx].item()),
                )

        queried_history_slot_ids: list[tuple[int, ...]] = []
        for episode_idx_tensor in normalized_batch["episode_idx"]:
            episode_idx = int(episode_idx_tensor)
            slot_ids = tuple(
                self.history_memory_slot_ids_dict.get(episode_idx, [])
            )
            if self.prediction_error_queue_enabled:
                history_size = len(
                    self.history_img_features_dict.get(episode_idx, [])
                )
                if len(slot_ids) != history_size:
                    raise RuntimeError(
                        "History slot IDs and image features are misaligned for "
                        f"episode {episode_idx}"
                    )
            queried_history_slot_ids.append(slot_ids)

        # Capture the exact memory state used by this decision after dynamic
        # weight correction and before the current observation is appended as
        # a new slot. This remains correct when the post-decision buffer evicts
        # a low-weight slot instead of using FIFO.
        self.latest_online_memory_query_dict = {}
        for batch_idx, episode_idx_tensor in enumerate(
            normalized_batch["episode_idx"]
        ):
            episode_idx = int(episode_idx_tensor)
            frame_indices = tuple(
                self.history_frame_indices_dict.get(episode_idx, [])
            )
            slot_ids = queried_history_slot_ids[batch_idx]
            if self.prediction_error_queue_enabled:
                weights = self.history_memory_weights_dict.get(episode_idx, [])
                if not (
                    len(frame_indices) == len(slot_ids) == len(weights)
                ):
                    raise RuntimeError(
                        "online slot ID/frame-index/weight buffers are "
                        f"misaligned for episode {episode_idx}"
                    )
                if weights:
                    weight_values = (
                        torch.stack(weights).detach().float().cpu().tolist()
                    )
                    slot_weights: tuple[float | None, ...] = tuple(
                        float(weight) for weight in weight_values
                    )
                else:
                    slot_weights = ()
            else:
                # Policies without prediction-error weighting still expose
                # frame positions; their weight column is intentionally empty.
                if slot_ids and len(slot_ids) != len(frame_indices):
                    raise RuntimeError(
                        "online slot ID/frame-index buffers are misaligned for "
                        f"episode {episode_idx}"
                    )
                if not slot_ids:
                    slot_ids = tuple(range(len(frame_indices)))
                slot_weights = tuple(None for _ in frame_indices)
            self.latest_online_memory_query_dict[episode_idx] = {
                "query_frame_index": int(current_frame_indices[batch_idx]),
                "slot_ids": slot_ids,
                "slot_frame_indices": frame_indices,
                "slot_weights": slot_weights,
            }

        # history_img_features is invariant to the diffusion step
        if self.history_img_feature_encoder is not None and max_history_len > 0:
            history_img_features = torch.zeros(
                (
                    batch_size,
                    max_history_len,
                    self.history_storage_token_num,
                    self.history_storage_feature_dim,
                ),
                device=self.device,
            )
            for idx, episode_idx in enumerate(normalized_batch["episode_idx"]):
                if int(episode_idx) in self.history_img_features_dict.keys():
                    history_len = len(self.history_img_features_dict[int(episode_idx)])
                    history_img_features[idx, -history_len:] = torch.stack(
                        self.history_img_features_dict[int(episode_idx)], dim=0
                    )

        history_frame_indices = torch.zeros(
            (batch_size, max_history_len),
            dtype=torch.long,
            device=self.device,
        )
        for idx, episode_idx_tensor in enumerate(
            normalized_batch["episode_idx"]
        ):
            episode_idx = int(episode_idx_tensor)
            retained_frame_indices = self.history_frame_indices_dict.get(
                episode_idx, []
            )
            retained_action_count = len(
                self.history_noisy_actions_dict.get(episode_idx, [])
            )
            if len(retained_frame_indices) != retained_action_count:
                raise RuntimeError(
                    "online action/frame-index buffers are misaligned for "
                    f"episode {episode_idx}"
                )
            if retained_frame_indices:
                history_frame_indices[idx, -len(retained_frame_indices):] = (
                    torch.as_tensor(
                        retained_frame_indices,
                        dtype=torch.long,
                        device=self.device,
                    )
                )

        recorded_data_dicts: list[dict[str, torch.Tensor]] = []
        cached_decision_memory: torch.Tensor | None = None
        pending_retrieval_inputs: dict[str, torch.Tensor] | None = None

        for k, t in enumerate(self.noise_scheduler.get_inference_timesteps()):
            decision_memory_was_cached = cached_decision_memory is not None
            history_noisy_actions = torch.zeros(
                (
                    batch_size,
                    max_history_len,
                    self.history_action_num_per_chunk,
                    action_dim
                ),
                device=self.device,
            ) # (batch_size, max_history_len, history_action_num_per_chunk, action_dim)

            history_mask = torch.zeros(
                (batch_size, max_history_len),
                device=self.device,
                dtype=torch.bool,
            )
            history_attention_bias = torch.zeros(
                (batch_size, max_history_len),
                device=self.device,
                dtype=torch.float32,
            )
            # Keep the persistent weights separate from the query-only recent
            # slot floor. Dynamic updates and eviction must use this tensor,
            # never the temporarily raised attention bias.
            history_stored_weights = torch.zeros_like(history_attention_bias)

            # print(f"{self.history_noisy_actions_dict.keys()=}")
            # print(f"{normalized_batch['episode_idx']=}")

            for l, episode_idx in enumerate(normalized_batch["episode_idx"]):
                if int(episode_idx) in self.history_noisy_actions_dict.keys():

                    if self.eval_history_action_noise_level == "none":
                        diffusion_step_idx = -1
                    elif self.eval_history_action_noise_level == "random":
                        rand_idx = int(torch.randint(0, len(self.noise_scheduler.get_inference_timesteps()), (1,)).item())
                        diffusion_step_idx = rand_idx
                    elif self.eval_history_action_noise_level == "last_step":
                        diffusion_step_idx = k
                    else:
                        raise ValueError(f"Invalid history action noise level: {self.eval_history_action_noise_level}")

                    history_len = len(self.history_noisy_actions_dict[int(episode_idx)])
                    stacked_history_noisy_action = torch.stack(
                        [
                            self.history_noisy_actions_dict[int(episode_idx)][i][
                                diffusion_step_idx
                            ]
                            for i in range(history_len)
                        ],
                        dim=0,
                    )  # (history_len, token_num, hidden_dim)

                    history_noisy_actions[l, -history_len:] = (
                        stacked_history_noisy_action
                    )

                    history_mask[l, -history_len:] = 1
                    if self.prediction_error_queue_enabled:
                        episode_idx_int = int(episode_idx)
                        if episode_idx_int not in self.history_memory_weights_dict:
                            raise RuntimeError(
                                "Missing memory weights for episode "
                                f"{episode_idx_int}"
                            )
                        weights = self.history_memory_weights_dict[
                            episode_idx_int
                        ]
                        if len(weights) != history_len:
                            raise RuntimeError(
                                "History feature/action/weight buffers are not "
                                f"aligned for episode {episode_idx_int}"
                            )
                        assert self.prediction_error_queue_config is not None
                        stored_weights = torch.stack(weights).to(
                            history_attention_bias
                        )
                        history_stored_weights[l, -history_len:] = stored_weights
                        retained_frame_indices = (
                            self.history_frame_indices_dict[episode_idx_int]
                        )
                        fixed_slot_mask = self._frozen_first_slot_mask(
                            torch.as_tensor(
                                retained_frame_indices,
                                dtype=torch.long,
                                device=stored_weights.device,
                            )
                        )
                        recent_slot_protection_num = (
                            self.dynamic_memory_weight_update_config
                            .recent_slot_protection_num
                            if self.dynamic_memory_weight_update_config
                            is not None
                            else 0
                        )
                        history_attention_bias[l, -history_len:] = (
                            apply_recent_slot_attention_floor(
                                stored_weights,
                                attention_bias_mode=(
                                    self.history_attention_bias_mode
                                ),
                                recent_slot_protection_num=(
                                    recent_slot_protection_num
                                ),
                                recent_slot_attention_floor=(
                                    self.prediction_error_queue_config
                                    .recent_slot_attention_floor
                                ),
                                fixed_slot_mask=fixed_slot_mask,
                            )
                        )

            # These keys need to be overridden every time before the denoising network is called
            # Since the denoising network will pop the keys after the forward pass
            data_dict["history_noisy_actions"] = history_noisy_actions
            data_dict["history_mask"] = history_mask
            data_dict["history_frame_indices"] = history_frame_indices
            if self.prediction_error_queue_enabled and self.attention_bias:
                data_dict["history_attention_bias"] = history_attention_bias
            else:
                data_dict.pop("history_attention_bias", None)
            if memory_gate_val is not None: 
                data_dict["memory_gate_val"] = memory_gate_val

                
            data_dict["step"] = (
                torch.ones((batch_size,), device=self.device) * t
            )

            if self.history_img_feature_encoder is not None and max_history_len > 0:
                noise_ratio = t / self.noise_scheduler.train_step_num
                data_dict["history_img_features"] = history_img_features

            if (
                cached_decision_memory is None
                and self.dynamic_memory_weight_update_enabled
            ):
                pending_retrieval_inputs = (
                    self.denoising_network.prepare_single_memory_retrieval(
                        data_dict
                    )
                )
                if len(self.denoising_network.record_data_entries) > 0:
                    retrieval_record_data: (
                        dict[str, list[torch.Tensor]] | None
                    ) = {
                        entry: []
                        for entry in self.denoising_network.record_data_entries
                    }
                    if (
                        "memory_gate_val" in pending_retrieval_inputs
                        and "memory_gate_val" in retrieval_record_data
                    ):
                        retrieval_record_data["memory_gate_val"].append(
                            pending_retrieval_inputs["memory_gate_val"].clone()
                        )
                else:
                    retrieval_record_data = None
                cached_decision_memory = (
                    self.denoising_network.retrieve_decision_memory(
                        pending_retrieval_inputs,
                        record_data_dict=retrieval_record_data,
                    ).detach()
                )
                if (
                    self.prediction_error_queue_enabled
                    and not self.attention_bias
                ):
                    # Keep a shadow copy solely for the next prediction-error
                    # gradient update. The decision memory above, and therefore
                    # the policy action, was retrieved without this bias.
                    pending_retrieval_inputs["history_attention_bias"] = (
                        history_attention_bias.detach().clone()
                    )
                if retrieval_record_data is not None:
                    self.denoising_network.recorded_data_dict = (
                        retrieval_record_data
                    )
                    # The observation-driven retrieval has already run and its
                    # decision memory is reused by the DiT below. Carry the
                    # corresponding attention record through that forward;
                    # otherwise _run_blocks() would replace it with an empty
                    # record because it does not retrieve history a second time.
                    data_dict["_precomputed_memory_record_data"] = (
                        retrieval_record_data
                    )
                data_dict["decision_memory"] = cached_decision_memory
            elif cached_decision_memory is not None:
                data_dict["decision_memory"] = cached_decision_memory

            # if self.mask_in_eval:
            #     self._add_random_masks(data_dict)

            model_output = self.denoising_network.forward(data_dict)
            # forward() deep-copies its input, so remove the one-shot record
            # handoff from the policy-side dictionary after the first use.
            data_dict.pop("_precomputed_memory_record_data", None)
            if cached_decision_memory is None:
                cached_decision_memory = model_output["decision_memory"].detach()
            if (
                len(self.denoising_network.record_data_entries) > 0
                and not decision_memory_was_cached
            ):
                # print(f"{self.denoising_network.record_data_entries=}, {self.denoising_network.recorded_data_dict=}")
                merged_data_dict = dict_apply(
                    self.denoising_network.recorded_data_dict,
                    lambda x: torch.stack(x, dim=1).detach(),
                )
                recorded_data_dicts.append(copy.deepcopy(merged_data_dict))

            data_dict["noisy_action"] = self.noise_scheduler.step(
                model_output["action"],
                int(t),
                data_dict["noisy_action"],
            )

            for l, episode_idx in enumerate(normalized_batch["episode_idx"]):
                if int(episode_idx) not in new_history_action_dict.keys():
                    new_history_action_dict[int(episode_idx)] = []
                new_history_action_dict[int(episode_idx)].append(
                    data_dict["noisy_action"][l, :self.history_action_num_per_chunk].detach().clone()
                )

        if max_history_len == 0: # For ablation study
            output = self.action_decoder.forward(data_dict["noisy_action"])
            return output  # (batch_size, traj_length, action_dim)

        # ================================ Update history buffer ================================

        for episode_idx, history_action in new_history_action_dict.items():
            # History: list [num_inference_steps] of tensors [noisy_history_action, shape: (action_length, action_dim)]
            if episode_idx not in self.history_noisy_actions_dict.keys():
                self.history_noisy_actions_dict[episode_idx] = []
            self.history_noisy_actions_dict[episode_idx].append(history_action)

        for episode_idx_tensor, frame_index_tensor in zip(
            normalized_batch["episode_idx"], current_frame_indices
        ):
            episode_idx = int(episode_idx_tensor)
            frame_index = int(frame_index_tensor)
            if episode_idx not in self.history_frame_indices_dict:
                self.history_frame_indices_dict[episode_idx] = []
            self.history_frame_indices_dict[episode_idx].append(frame_index)
            self.next_frame_index_dict[episode_idx] = (
                frame_index + self.history_action_num_per_chunk
            )

        
        if current_history_img_features is not None:
            for episode_idx, history_img_features in zip(
                normalized_batch["episode_idx"], current_history_img_features
            ):
                episode_idx = int(episode_idx)
                if episode_idx not in self.history_img_features_dict.keys():
                    self.history_img_features_dict[episode_idx] = []
                self.history_img_features_dict[episode_idx].append(
                    history_img_features.detach().clone()
                )

        if current_slot_weights is not None:
            for episode_idx_tensor, slot_weight in zip(
                normalized_batch["episode_idx"], current_slot_weights
            ):
                episode_idx = int(episode_idx_tensor)
                if episode_idx not in self.history_memory_weights_dict:
                    self.history_memory_weights_dict[episode_idx] = []
                    self.history_memory_slot_ids_dict[episode_idx] = []
                    self.next_memory_slot_id_dict[episode_idx] = 0
                self.history_memory_weights_dict[episode_idx].append(
                    slot_weight.detach().clone()
                )
                slot_id = self.next_memory_slot_id_dict[episode_idx]
                self.next_memory_slot_id_dict[episode_idx] = slot_id + 1
                self.history_memory_slot_ids_dict[episode_idx].append(slot_id)

        for episode_idx in new_history_action_dict:
            self._trim_online_history_buffer(episode_idx, max_history_len)

        if current_slot_weights is not None:
            if cached_decision_memory is None or self.future_feature_predictor is None:
                raise RuntimeError("Cannot create the next future prediction")
            if self.future_feature_predictor.use_action_condition:
                # The next memory slot is observed after executing exactly one
                # history-action chunk. Do not condition the future predictor
                # on the unexecuted tail of the longer diffusion horizon.
                action_condition = data_dict["noisy_action"][
                    :, : self.history_action_num_per_chunk
                ]
            else:
                action_condition = None
            pending_predictions = self.future_feature_predictor(
                current_observation_tokens=(
                    self._pool_future_prediction_context(
                        cached_decision_memory
                    )
                ),
                action_chunk=action_condition,
            ).detach()
            for episode_idx_tensor, pending_prediction in zip(
                normalized_batch["episode_idx"], pending_predictions
            ):
                self.pending_future_predictions_dict[int(episode_idx_tensor)] = (
                    pending_prediction.clone()
                )
            if self.dynamic_memory_weight_update_enabled:
                if pending_retrieval_inputs is None:
                    raise RuntimeError(
                        "Cannot create a dynamic-memory recomputation snapshot"
                    )
                for batch_idx, episode_idx_tensor in enumerate(
                    normalized_batch["episode_idx"]
                ):
                    episode_idx = int(episode_idx_tensor)
                    snapshot_inputs = {
                        key: value[batch_idx : batch_idx + 1].detach().clone()
                        for key, value in pending_retrieval_inputs.items()
                    }
                    if action_condition is None:
                        snapshot_action_condition = None
                    else:
                        snapshot_action_condition = (
                            action_condition[batch_idx : batch_idx + 1]
                            .detach()
                            .clone()
                        )
                    self.pending_dynamic_memory_updates_dict[episode_idx] = (
                        PendingDynamicMemoryUpdate(
                            retrieval_inputs=snapshot_inputs,
                            stored_slot_weights=(
                                history_stored_weights[
                                    batch_idx : batch_idx + 1
                                ]
                                .detach()
                                .clone()
                            ),
                            history_slot_ids=queried_history_slot_ids[batch_idx],
                            history_frame_indices=tuple(
                                self.latest_online_memory_query_dict[episode_idx][
                                    "slot_frame_indices"
                                ]
                            ),
                            action_condition=snapshot_action_condition,
                        )
                    )


        if len(recorded_data_dicts) > 0:
            merged_data_dict: dict[str, torch.Tensor] = {}
            merged_data_dict = aggregate_batch(
                recorded_data_dicts, partial(torch.stack, dim=1)
            )
            # Raw history is retrieved once to construct h_t, so dimension 2 is 1
            # rather than transformer_layer_num.
            splitted_data_dicts = split_batch(
                merged_data_dict, partial(torch.unbind, dim=0)
            )
            for k, splitted_data_dict in enumerate(splitted_data_dicts):
                episode_idx = normalized_batch["episode_idx"][k]
                splitted_data_dict = dict_apply(
                    splitted_data_dict, lambda x: x.detach().clone().cpu()
                )
                if int(episode_idx) not in self.recorded_data_dicts.keys():
                    self.recorded_data_dicts[int(episode_idx)] = []
                self.recorded_data_dicts[int(episode_idx)].append(splitted_data_dict)

        output = self.action_decoder.forward(data_dict["noisy_action"])

        return output  # (batch_size, traj_length, action_dim)

    def predict_multi_traj(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """
        Used when the batch contains multiple trajectories in the same episode.
        This function is only used when running validation with multiple ground-truth trajectories.

        normalized_batch:
            "robot0_wrist_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size)
            "robot0_10d": (batch_size, traj_num, traj_length, 8)
            "third_person_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size) # For table-bin scenario
            "episode_idx": (batch_size) # Need to be overridden
        return:
            "action0_10d": (batch_size, traj_num, traj_length, 8)
        """
        traj_num_dim_idx = 1  # batch_size is 0

        # Use single trajectory prediction to iteratively predict all trajectories
        self.reset()
        actions: list[dict[str, torch.Tensor]] = []
        # ``streaming_sequence`` is a per-sample trainer marker.  After
        # collation it has shape [batch] and therefore has no trajectory
        # dimension to unbind below.  It is not model input, so discard it
        # before splitting a multi-trajectory batch for validation/sampling.
        normalized_batch.pop("streaming_sequence", None)
        normalized_batch.pop("prefix_sequence", None)
        if "variance_temperature" in normalized_batch:
            normalized_batch.pop(
                "variance_temperature"
            )  # Remove variance_temperature from meta

        batch_size = normalized_batch["episode_idx"].shape[0]
        traj_num = normalized_batch["episode_idx"].shape[1]


        for batch in split_batch(
            normalized_batch,
            partial(torch.unbind, dim=traj_num_dim_idx),
        ):  # Along the traj_num dimension
            """
            batch:
                "robot0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
                "robot0_10d": (batch_size, traj_length, 8)
                "third_person_camera": (batch_size, traj_length, 3, image_size, image_size) # For table-bin scenario
                "episode_idx": (batch_size, )
            """
            batch["episode_idx"] = torch.arange(batch_size, device=self.device) # Override episode idx so the history can be correctly recorded
            actions.append(
                self.predict_single_traj(batch)
            )  # (batch_size, traj_length, action_dim)

        return aggregate_batch(
            actions, partial(torch.stack, dim=traj_num_dim_idx)
        )  # (batch_size, traj_num, traj_length, action_dim)


    # ================================ Training =================================

    @property
    def prediction_error_queue_enabled(self) -> bool:
        return (
            self.prediction_error_queue_config is not None
            and self.prediction_error_queue_config.enabled
        )

    @property
    def dynamic_memory_weight_update_enabled(self) -> bool:
        return (
            self.dynamic_memory_weight_update_config is not None
            and self.dynamic_memory_weight_update_config.enabled
        )

    def _new_prediction_error_queue(self) -> PredictionErrorQueue:
        if self.prediction_error_queue_config is None:
            raise RuntimeError("prediction_error_queue is not configured")
        return PredictionErrorQueue(self.prediction_error_queue_config)

    def _is_frozen_first_slot_frame(self, frame_index: int) -> bool:
        """Whether an absolute episode frame keeps the first-slot prior."""

        config = self.prediction_error_queue_config
        return bool(
            config is not None
            and config.first_slot_frozen
            and 0 <= frame_index <= config.first_slot_max_frame_distance
        )

    def _frozen_first_slot_mask(
        self,
        frame_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized counterpart of ``_is_frozen_first_slot_frame``."""

        config = self.prediction_error_queue_config
        if config is None or not config.first_slot_frozen:
            return torch.zeros_like(frame_indices, dtype=torch.bool)
        return (
            (frame_indices >= 0)
            & (frame_indices <= config.first_slot_max_frame_distance)
        )

    def _freeze_first_slot_weight(
        self,
        weight: torch.Tensor,
        frame_index: int,
    ) -> torch.Tensor:
        """Replace an eligible slot weight with the immutable prior."""

        if not self._is_frozen_first_slot_frame(frame_index):
            return weight
        assert self.prediction_error_queue_config is not None
        return weight.new_tensor(
            self.prediction_error_queue_config.first_slot_weight
        )

    def _get_online_prediction_error_queue(
        self, episode_idx: int
    ) -> PredictionErrorQueue:
        if episode_idx not in self.prediction_error_queues:
            self.prediction_error_queues[episode_idx] = (
                self._new_prediction_error_queue()
            )
        return self.prediction_error_queues[episode_idx]

    def _resolve_pending_dynamic_memory_update(
        self,
        episode_idx: int,
        current_image_features: torch.Tensor,
    ) -> DynamicMemoryWeightRecord | None:
        """Recompute the previous prediction and correct retained old slots."""

        if not self.dynamic_memory_weight_update_enabled:
            return None
        if episode_idx not in self.pending_dynamic_memory_updates_dict:
            raise RuntimeError(
                "Missing dynamic-memory snapshot for non-empty episode "
                f"{episode_idx}. Call reset() before reusing episode ids."
            )
        snapshot = self.pending_dynamic_memory_updates_dict.pop(episode_idx)
        if not snapshot.history_slot_ids:
            return None
        if self.future_feature_predictor is None:
            raise RuntimeError("future_feature_predictor is required")
        if not isinstance(
            self.denoising_network, (MemoryTransformer, OptimizedModule)
        ):
            raise TypeError("MemoryTransformer is required")
        denoising_network = cast(
            MemoryTransformer, cast(Any, self.denoising_network)
        )
        assert self.dynamic_memory_weight_update_config is not None
        assert self.prediction_error_queue_config is not None

        with torch.enable_grad():
            stored_bias = snapshot.retrieval_inputs.get(
                "history_attention_bias"
            )
            history_mask = snapshot.retrieval_inputs.get("history_mask")
            if stored_bias is None or history_mask is None:
                raise RuntimeError(
                    "dynamic-memory snapshot is missing history bias or mask"
                )
            read_bias = stored_bias.detach().float().clone().requires_grad_(True)
            decision_memory = denoising_network.retrieve_decision_memory(
                snapshot.retrieval_inputs,
                history_attention_bias=read_bias,
            )
            predicted_features = self.future_feature_predictor(
                current_observation_tokens=(
                    self._pool_future_prediction_context(decision_memory)
                ),
                action_chunk=snapshot.action_condition,
            )
            prediction_error = (
                self.future_feature_predictor.cosine_prediction_error(
                    predicted_features=predicted_features,
                    target_features=current_image_features.detach(),
                    detach_target=True,
                ).mean()
            )
            dynamic_gradient = torch.autograd.grad(
                outputs=prediction_error,
                inputs=read_bias,
                retain_graph=False,
                create_graph=False,
            )[0]

        valid_positions = torch.nonzero(
            history_mask[0].to(dtype=torch.bool), as_tuple=False
        ).flatten()
        if not (
            len(valid_positions)
            == len(snapshot.history_slot_ids)
            == len(snapshot.history_frame_indices)
        ):
            raise RuntimeError(
                "snapshot slot IDs/frame indices are not aligned with its "
                "history mask"
            )
        # Optionally keep every early episode-reference frame inside the
        # configured distance at the first-slot prior.
        dynamic_valid_mask = history_mask.to(dtype=torch.bool).clone()
        for frame_index, bias_position in zip(
            snapshot.history_frame_indices, valid_positions.tolist()
        ):
            if self._is_frozen_first_slot_frame(frame_index):
                dynamic_valid_mask[0, bias_position] = False

        stored_slot_weights = snapshot.stored_slot_weights.to(read_bias)
        if stored_slot_weights.shape != read_bias.shape:
            raise RuntimeError(
                "stored slot weights and attention bias are not aligned: "
                f"{tuple(stored_slot_weights.shape)} != {tuple(read_bias.shape)}"
            )
        updated_bias, record = apply_dynamic_memory_weight_update(
            stored_slot_weights,
            dynamic_gradient,
            dynamic_valid_mask,
            step_size=self.dynamic_memory_weight_update_config.step_size,
            weight_min=self.prediction_error_queue_config.weight_min,
            weight_max=self.prediction_error_queue_config.weight_max,
        )

        live_slot_ids = self.history_memory_slot_ids_dict.get(episode_idx, [])
        live_weights = self.history_memory_weights_dict.get(episode_idx, [])
        if len(live_slot_ids) != len(live_weights):
            raise RuntimeError(
                f"online slot IDs and weights are misaligned for episode {episode_idx}"
            )
        live_positions = {
            slot_id: idx for idx, slot_id in enumerate(live_slot_ids)
        }
        for slot_id, frame_index, bias_position in zip(
            snapshot.history_slot_ids,
            snapshot.history_frame_indices,
            valid_positions.tolist(),
        ):
            if slot_id not in live_positions:
                # This slot was evicted after the snapshot was created.
                continue
            live_idx = live_positions[slot_id]
            live_weights[live_idx] = self._freeze_first_slot_weight(
                updated_bias[0, bias_position].detach().clone(),
                frame_index,
            )
        return record

    def _trim_online_history_buffer(
        self, episode_idx: int, max_history_len: int
    ) -> None:
        """Evict aligned slots using minimum weight or standard FIFO."""

        actions = self.history_noisy_actions_dict.get(episode_idx, [])
        images = (
            self.history_img_features_dict.get(episode_idx, [])
            if self.history_img_feature_encoder is not None
            else None
        )
        if images is not None and len(actions) != len(images):
            raise RuntimeError(
                f"online action/image buffers are misaligned for episode {episode_idx}"
            )

        weights = self.history_memory_weights_dict.get(episode_idx)
        slot_ids = self.history_memory_slot_ids_dict.get(episode_idx)
        frame_indices = self.history_frame_indices_dict.get(episode_idx)
        if frame_indices is None:
            raise RuntimeError(
                f"online frame-index buffer is missing for episode {episode_idx}"
            )
        if len(actions) != len(frame_indices):
            raise RuntimeError(
                "online action/frame-index buffers are misaligned for episode "
                f"{episode_idx}"
            )
        if self.prediction_error_queue_enabled:
            if weights is None or slot_ids is None:
                raise RuntimeError(
                    f"online weight/ID buffer is missing for episode {episode_idx}"
                )
            if len(actions) != len(weights) or len(actions) != len(slot_ids):
                raise RuntimeError(
                    f"online history buffers are misaligned for episode {episode_idx}"
                )

        while len(actions) > max_history_len:
            evict_lowest = (
                self.dynamic_memory_weight_update_config is not None
                and self.dynamic_memory_weight_update_config
                .evict_lowest_weight_when_full
            )
            if evict_lowest:
                assert weights is not None
                eviction_idx = select_memory_slot_eviction_index(
                    weights,
                    evict_lowest_weight_when_full=True,
                    recent_slot_protection_num=(
                        self.dynamic_memory_weight_update_config
                        .recent_slot_protection_num
                    ),
                )
            else:
                eviction_idx = 0
            actions.pop(eviction_idx)
            frame_indices.pop(eviction_idx)
            if images is not None:
                images.pop(eviction_idx)
            if weights is not None:
                weights.pop(eviction_idx)
            if slot_ids is not None:
                slot_ids.pop(eviction_idx)

    @staticmethod
    def _summarize_prediction_error_records(
        records: list[PredictionErrorRecord],
        dynamic_records: list[DynamicMemoryWeightRecord],
        slot_weights: torch.Tensor,
        valid_slot_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid_slot_weights = slot_weights.masked_select(valid_slot_mask)
        valid_first_slot_mask = valid_slot_mask[:, 0]
        valid_first_slot_weights = slot_weights[:, 0].masked_select(
            valid_first_slot_mask
        )
        summary = {
            "all_slot_weight_mean": valid_slot_weights.mean(),
            "all_slot_weight_min": valid_slot_weights.min(),
            "all_slot_weight_max": valid_slot_weights.max(),
            "first_slot_weight_mean": valid_first_slot_weights.mean(),
        }
        if records:
            stacked = {
                key: torch.stack([record.as_dict()[key] for record in records])
                for key in records[0].as_dict()
            }
            summary.update({
                "current_error_mean": stacked["current_error"].mean(),
                "queue_mean": stacked["queue_mean"].mean(),
                "queue_std": stacked["queue_std"].mean(),
                "effective_std": stacked["effective_std"].mean(),
                "queue_size_mean": stacked["queue_size_before"].float().mean(),
                "queue_size_max": stacked["queue_size_before"].float().max(),
                "warmup_fraction": stacked["is_warmup"].float().mean(),
                "z_score_mean": stacked["z_score"].mean(),
                "z_score_min": stacked["z_score"].min(),
                "z_score_max": stacked["z_score"].max(),
                "initial_weight_mean": stacked["initial_weight"].mean(),
                "initial_weight_min": stacked["initial_weight"].min(),
                "initial_weight_max": stacked["initial_weight"].max(),
            })

        if dynamic_records:
            gradients = torch.cat(
                [record.gradient for record in dynamic_records]
            )
            weights_before = torch.cat(
                [record.weight_before for record in dynamic_records]
            )
            weights_after = torch.cat(
                [record.weight_after for record in dynamic_records]
            )
            was_clipped = torch.cat(
                [record.was_clipped for record in dynamic_records]
            )
            summary.update(
                {
                    "dynamic_gradient_mean": gradients.mean(),
                    "dynamic_gradient_min": gradients.min(),
                    "dynamic_gradient_max": gradients.max(),
                    "dynamic_gradient_abs_mean": gradients.abs().mean(),
                    "dynamic_weight_before_mean": weights_before.mean(),
                    "dynamic_weight_after_mean": weights_after.mean(),
                    "dynamic_weight_delta_mean": (
                        weights_after - weights_before
                    ).mean(),
                    "dynamic_weight_clip_fraction": was_clipped.float().mean(),
                }
            )
        return summary

    def _build_causal_training_memories(
        self,
        projected_input: dict[str, torch.Tensor],
        target: batch_type,
        valid_slot_mask: torch.Tensor,
        slot_frame_indices: torch.Tensor,
        pair_start_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[PredictionErrorRecord],
        list[DynamicMemoryWeightRecord],
        torch.Tensor | None,
        torch.Tensor,
    ]:
        """Build causal training memories in temporal anchor order.

        Only retrieval and future prediction are sequential. The returned
        decision memories are later sent through all DiT blocks as one flattened
        batch. When ``training_slot_weight`` is disabled, training uses raw
        attention and uniformly evicts one existing slot whenever the bank is
        full; prediction-error weighting remains available during inference.
        """

        if self.training_slot_weight and not self.prediction_error_queue_enabled:
            raise RuntimeError("causal memory construction requires error queue")
        if self.future_feature_predictor is None:
            raise RuntimeError("future_feature_predictor is required")
        if not isinstance(
            self.denoising_network, (MemoryTransformer, OptimizedModule)
        ):
            raise TypeError("MemoryTransformer is required")

        batch_size, traj_num = valid_slot_mask.shape
        if slot_frame_indices.shape != (batch_size, traj_num):
            raise ValueError(
                "slot_frame_indices must have shape "
                f"{(batch_size, traj_num)}, got "
                f"{tuple(slot_frame_indices.shape)}"
            )
        random_pair_training = pair_start_mask is not None
        if pair_start_mask is not None:
            if pair_start_mask.shape != (batch_size, traj_num):
                raise ValueError(
                    "pair_start_mask must have shape "
                    f"{(batch_size, traj_num)}, got "
                    f"{tuple(pair_start_mask.shape)}"
                )
            pair_start_mask = pair_start_mask.to(
                device=projected_input["x"].device,
                dtype=torch.bool,
            )
        flat_batch_size = batch_size * traj_num
        if projected_input["x"].shape[0] != flat_batch_size:
            raise ValueError("projected input has an unexpected flattened batch")

        history_len = self.denoising_network.max_history_len
        all_history_latents = projected_input.get("all_history_latents")
        if all_history_latents is None:
            raise KeyError("projected input is missing all_history_latents")
        if all_history_latents.shape[:2] != (batch_size, traj_num):
            raise ValueError(
                "all_history_latents must start with batch and trajectory "
                f"dimensions, got {tuple(all_history_latents.shape)}"
            )
        slot_weights = (
            torch.zeros(
                (batch_size, traj_num),
                dtype=torch.float32,
                device=projected_input["x"].device,
            )
            if self.training_slot_weight
            else None
        )
        queues = (
            [self._new_prediction_error_queue() for _ in range(batch_size)]
            if self.training_slot_weight
            else None
        )
        retained_slot_indices: list[list[int]] = [
            [] for _ in range(batch_size)
        ]
        if self.training_slot_weight and traj_num > 0:
            assert slot_weights is not None
            assert queues is not None
            if random_pair_training:
                assert pair_start_mask is not None
                pair_start_weights = queues[0].first_slot_weight(
                    slot_weights,
                    slot_frame_indices,
                )
                slot_weights = torch.where(
                    pair_start_mask & valid_slot_mask,
                    pair_start_weights,
                    slot_weights,
                )
            else:
                first_weight = queues[0].first_slot_weight(
                    slot_weights[:, 0], slot_frame_indices[:, 0]
                )
                slot_weights[:, 0] = torch.where(
                    valid_slot_mask[:, 0],
                    first_weight,
                    torch.zeros_like(first_weight),
                )

        decision_memory_by_time: list[torch.Tensor] = []
        predicted_future_by_time: list[torch.Tensor] = []
        records: list[PredictionErrorRecord] = []
        dynamic_records: list[DynamicMemoryWeightRecord] = []
        batch_offsets = (
            torch.arange(batch_size, device=projected_input["x"].device)
            * traj_num
        )

        record_entries = list(self.denoising_network.record_data_entries)
        collect_epoch_attention_stats = (
            self.denoising_network.training
            and getattr(
                self.denoising_network,
                "record_history_attention_epoch_stats",
                False,
            )
        )
        if collect_epoch_attention_stats:
            for entry in (
                "history_cross_attention_frame_logits",
                "history_cross_attention_slot_weights",
            ):
                if entry not in record_entries:
                    record_entries.append(entry)
        if record_entries:
            causal_retrieval_record_data: (
                dict[str, list[torch.Tensor]] | None
            ) = {entry: [] for entry in record_entries}
            if (
                "memory_gate_val" in causal_retrieval_record_data
                and "memory_gate_val" in projected_input
            ):
                causal_retrieval_record_data["memory_gate_val"].append(
                    projected_input["memory_gate_val"].clone()
                )
        else:
            causal_retrieval_record_data = None

        for traj_idx in range(traj_num):
            flat_indices = batch_offsets + traj_idx
            history_latent_rows: list[torch.Tensor] = []
            history_key_latent_rows: list[torch.Tensor] = []
            history_mask_rows: list[torch.Tensor] = []
            history_bias_rows: list[torch.Tensor] = []
            history_stored_weight_rows: list[torch.Tensor] = []
            for batch_idx, source_indices in enumerate(retained_slot_indices):
                retained_num = len(source_indices)
                padding_num = history_len - retained_num
                padding_latents = torch.zeros(
                    (padding_num, *all_history_latents.shape[2:]),
                    dtype=all_history_latents.dtype,
                    device=all_history_latents.device,
                )
                if retained_num > 0:
                    source_index_tensor = torch.tensor(
                        source_indices,
                        dtype=torch.long,
                        device=all_history_latents.device,
                    )
                    retained_latents = all_history_latents[
                        batch_idx, source_index_tensor
                    ]
                    if self.denoising_network.uses_absolute_history_time:
                        retained_time_embedding = (
                            self.denoising_network
                            .encode_history_frame_indices(
                                slot_frame_indices[
                                    batch_idx, source_index_tensor
                                ]
                            )
                        )
                    else:
                        relative_time_embedding = (
                            self.denoising_network.history_time_embedding
                        )
                        if relative_time_embedding is None:
                            raise RuntimeError(
                                "relative history time embedding is missing"
                            )
                        retained_time_embedding = relative_time_embedding[
                            0, -retained_num:, :
                        ]
                    positioned_retained_latents = (
                        self.denoising_network.add_history_key_encoding(
                            retained_latents.unsqueeze(0),
                            retained_time_embedding.unsqueeze(0),
                        ).squeeze(0)
                    )
                    if self.training_slot_weight:
                        assert slot_weights is not None
                        assert self.prediction_error_queue_config is not None
                        retained_stored_weights = torch.stack(
                            [
                                slot_weights[batch_idx, source_idx]
                                for source_idx in source_indices
                            ]
                        )
                        retained_frame_indices = slot_frame_indices[
                            batch_idx, source_index_tensor
                        ]
                        recent_slot_protection_num = (
                            self.dynamic_memory_weight_update_config
                            .recent_slot_protection_num
                            if self.dynamic_memory_weight_update_config
                            is not None
                            else 0
                        )
                        retained_biases = apply_recent_slot_attention_floor(
                            retained_stored_weights,
                            attention_bias_mode=(
                                self.history_attention_bias_mode
                            ),
                            recent_slot_protection_num=(
                                recent_slot_protection_num
                            ),
                            recent_slot_attention_floor=(
                                self.prediction_error_queue_config
                                .recent_slot_attention_floor
                            ),
                            fixed_slot_mask=self._frozen_first_slot_mask(
                                retained_frame_indices
                            ),
                        )
                else:
                    retained_latents = all_history_latents.new_zeros(
                        (0, *all_history_latents.shape[2:])
                    )
                    positioned_retained_latents = retained_latents
                    if self.training_slot_weight:
                        assert slot_weights is not None
                        retained_biases = slot_weights.new_zeros((0,))
                        retained_stored_weights = slot_weights.new_zeros((0,))
                history_latent_rows.append(
                    torch.cat(
                        [
                            padding_latents,
                            (
                                retained_latents
                                if self.denoising_network.memory_retrieval_mode
                                == "perception_patches"
                                else positioned_retained_latents
                            ),
                        ],
                        dim=0,
                    )
                )
                if self.denoising_network.memory_retrieval_mode == "perception_patches":
                    history_key_latent_rows.append(
                        torch.cat(
                            [padding_latents, positioned_retained_latents],
                            dim=0,
                        )
                    )
                history_mask_rows.append(
                    torch.cat(
                        [
                            torch.zeros(
                                padding_num,
                                dtype=torch.bool,
                                device=all_history_latents.device,
                            ),
                            torch.ones(
                                retained_num,
                                dtype=torch.bool,
                                device=all_history_latents.device,
                            ),
                        ]
                    )
                )
                if self.training_slot_weight:
                    assert slot_weights is not None
                    history_bias_rows.append(
                        torch.cat(
                            [
                                slot_weights.new_zeros((padding_num,)),
                                retained_biases,
                            ]
                        )
                    )
                    history_stored_weight_rows.append(
                        torch.cat(
                            [
                                slot_weights.new_zeros((padding_num,)),
                                retained_stored_weights,
                            ]
                        )
                    )

            history_latents = torch.stack(history_latent_rows)
            history_key_latents = (
                torch.stack(history_key_latent_rows)
                if self.denoising_network.memory_retrieval_mode
                == "perception_patches"
                else None
            )
            history_mask = torch.stack(history_mask_rows)
            history_attention_bias = (
                torch.stack(history_bias_rows).detach()
                if self.training_slot_weight
                else None
            )
            history_stored_weights = (
                torch.stack(history_stored_weight_rows).detach()
                if self.training_slot_weight
                else None
            )
            if (
                self.training_slot_weight
                and self.dynamic_memory_weight_update_enabled
            ):
                assert history_attention_bias is not None
                history_attention_bias.requires_grad_(True)

            def select(name: str) -> torch.Tensor | None:
                value = projected_input.get(name)
                return None if value is None else value[flat_indices]

            decision_memory = self.denoising_network.memory_retriever(
                global_cond=projected_input["global_cond"][flat_indices],
                global_cond_mask=select("global_cond_mask"),
                local_cond=select("local_cond"),
                local_cond_mask=select("local_cond_mask"),
                history_latents=history_latents,
                history_key_latents=history_key_latents,
                history_mask=history_mask,
                history_attention_bias=(
                    history_attention_bias
                    if self.training_slot_weight and self.attention_bias
                    else None
                ),
                memory_gate_val=select("memory_gate_val"),
                skip_history=self.denoising_network.skip_history_attn,
                record_data_dict=causal_retrieval_record_data,
            )
            decision_memory_by_time.append(decision_memory)

            if self.future_feature_predictor.use_action_condition:
                action_condition = target["future_action_condition"][:, traj_idx]
            else:
                action_condition = None
            predicted_future = self.future_feature_predictor(
                current_observation_tokens=(
                    self._pool_future_prediction_context(decision_memory)
                ),
                action_chunk=action_condition,
            )
            predicted_future_by_time.append(predicted_future)

            if self.training_slot_weight and traj_idx + 1 < traj_num:
                assert queues is not None
                assert slot_weights is not None
                future_token_mask = target["future_feature_mask"][:, traj_idx]
                valid_future = future_token_mask.any(dim=-1)
                if valid_future.any():
                    token_errors = (
                        self.future_feature_predictor.cosine_prediction_error(
                            predicted_features=predicted_future,
                            target_features=target["future_features"][:, traj_idx],
                            detach_target=True,
                        )
                    )
                    masked_token_errors = token_errors * future_token_mask.to(
                        token_errors.dtype
                    )
                    per_sample_errors = masked_token_errors.sum(dim=-1) / (
                        future_token_mask.sum(dim=-1).clamp_min(1)
                    )

                    if (
                        self.dynamic_memory_weight_update_enabled
                        and history_mask.any()
                    ):
                        assert history_attention_bias is not None
                        dynamic_error = per_sample_errors
                        if not self.attention_bias:
                            # Main retrieval is intentionally unbiased. Run a
                            # shadow read only to preserve the original
                            # prediction-error gradient used for slot scoring
                            # and eviction; it never reaches the action model or
                            # the future-prediction training loss.
                            shadow_decision_memory = (
                                self.denoising_network.memory_retriever(
                                    global_cond=projected_input["global_cond"][
                                        flat_indices
                                    ],
                                    global_cond_mask=select("global_cond_mask"),
                                    local_cond=select("local_cond"),
                                    local_cond_mask=select("local_cond_mask"),
                                    history_latents=history_latents,
                                    history_key_latents=history_key_latents,
                                    history_mask=history_mask,
                                    history_attention_bias=(
                                        history_attention_bias
                                    ),
                                    memory_gate_val=select("memory_gate_val"),
                                    skip_history=(
                                        self.denoising_network.skip_history_attn
                                    ),
                                    record_data_dict=None,
                                )
                            )
                            shadow_prediction = self.future_feature_predictor(
                                current_observation_tokens=(
                                    self._pool_future_prediction_context(
                                        shadow_decision_memory
                                    )
                                ),
                                action_chunk=action_condition,
                            )
                            shadow_token_errors = (
                                self.future_feature_predictor
                                .cosine_prediction_error(
                                    predicted_features=shadow_prediction,
                                    target_features=target["future_features"][
                                        :, traj_idx
                                    ],
                                    detach_target=True,
                                )
                            )
                            dynamic_error = (
                                shadow_token_errors
                                * future_token_mask.to(
                                    shadow_token_errors.dtype
                                )
                            ).sum(dim=-1) / (
                                future_token_mask.sum(dim=-1).clamp_min(1)
                            )
                        dynamic_gradient = torch.autograd.grad(
                            outputs=(
                                dynamic_error
                                * valid_future.to(dynamic_error.dtype)
                            ).sum(),
                            inputs=history_attention_bias,
                            retain_graph=self.attention_bias,
                            create_graph=False,
                        )[0]
                        dynamic_valid_mask = (
                            history_mask & valid_future[:, None]
                        )
                        # Frozen early reference frames keep the configured
                        # first-slot prior in every causal training mode.
                        for batch_idx, source_indices in enumerate(
                            retained_slot_indices
                        ):
                            retained_num = len(source_indices)
                            for relative_idx, source_idx in enumerate(
                                source_indices
                            ):
                                frame_index = int(
                                    slot_frame_indices[
                                        batch_idx, source_idx
                                    ].item()
                                )
                                if self._is_frozen_first_slot_frame(
                                    frame_index
                                ):
                                    dynamic_valid_mask[
                                        batch_idx,
                                        history_len
                                        - retained_num
                                        + relative_idx,
                                    ] = False
                        assert self.dynamic_memory_weight_update_config is not None
                        assert self.prediction_error_queue_config is not None
                        assert history_stored_weights is not None
                        updated_bias, dynamic_record = (
                            apply_dynamic_memory_weight_update(
                                history_stored_weights,
                                dynamic_gradient,
                                dynamic_valid_mask,
                                step_size=(
                                    self.dynamic_memory_weight_update_config.step_size
                                ),
                                weight_min=(
                                    self.prediction_error_queue_config.weight_min
                                ),
                                weight_max=(
                                    self.prediction_error_queue_config.weight_max
                                ),
                            )
                        )
                        if dynamic_record.gradient.numel() > 0:
                            dynamic_records.append(dynamic_record)
                        for batch_idx, source_indices in enumerate(
                            retained_slot_indices
                        ):
                            retained_num = len(source_indices)
                            if retained_num == 0 or not bool(
                                valid_future[batch_idx]
                            ):
                                continue
                            for relative_idx, source_idx in enumerate(
                                source_indices
                            ):
                                frame_index = int(
                                    slot_frame_indices[
                                        batch_idx, source_idx
                                    ].item()
                                )
                                slot_weights[batch_idx, source_idx] = (
                                    self._freeze_first_slot_weight(
                                        updated_bias[
                                            batch_idx,
                                            history_len
                                            - retained_num
                                            + relative_idx,
                                        ],
                                        frame_index,
                                    )
                                )

                    for batch_idx in range(batch_size):
                        if not bool(valid_future[batch_idx]):
                            continue
                        record = queues[batch_idx].score_and_push(
                            per_sample_errors[batch_idx]
                        )
                        next_frame_index = int(
                            slot_frame_indices[
                                batch_idx, traj_idx + 1
                            ].item()
                        )
                        slot_weights[batch_idx, traj_idx + 1] = (
                            self._freeze_first_slot_weight(
                                record.initial_weight.to(slot_weights.dtype),
                                next_frame_index,
                            )
                        )
                        records.append(record)

            # The current anchor becomes history only after its prediction and
            # all corrections to the previously queried slots are complete.
            for batch_idx in range(batch_size):
                if not bool(valid_slot_mask[batch_idx, traj_idx]):
                    continue
                source_indices = retained_slot_indices[batch_idx]
                if history_len <= 0:
                    continue
                if (
                    not self.training_slot_weight
                    and len(source_indices) >= history_len
                ):
                    # Evict only from the already-existing history. The current
                    # slot is appended afterwards and is therefore guaranteed
                    # to be visible to at least the next valid anchor.
                    eviction_idx = select_random_training_slot_eviction_index(
                        len(source_indices),
                        device=all_history_latents.device,
                    )
                    source_indices.pop(eviction_idx)
                source_indices.append(traj_idx)
                if self.training_slot_weight and len(source_indices) > history_len:
                    assert slot_weights is not None
                    if self.dynamic_memory_weight_update_config is None:
                        evict_lowest = False
                    else:
                        evict_lowest = (
                            self.dynamic_memory_weight_update_config
                            .evict_lowest_weight_when_full
                        )
                    eviction_idx = select_memory_slot_eviction_index(
                        [
                            slot_weights[batch_idx, source_idx]
                            for source_idx in source_indices
                        ],
                        evict_lowest_weight_when_full=evict_lowest,
                        recent_slot_protection_num=(
                            self.dynamic_memory_weight_update_config
                            .recent_slot_protection_num
                            if self.dynamic_memory_weight_update_config
                            is not None
                            else 0
                        ),
                    )
                    source_indices.pop(eviction_idx)

        decision_memories = torch.stack(decision_memory_by_time, dim=1)
        predicted_futures = torch.stack(predicted_future_by_time, dim=1)
        retained_source_slot_indices = torch.full(
            (batch_size, history_len),
            -1,
            dtype=torch.long,
            device=all_history_latents.device,
        )
        for batch_idx, source_indices in enumerate(retained_slot_indices):
            retained_num = len(source_indices)
            if retained_num == 0:
                continue
            retained_source_slot_indices[batch_idx, -retained_num:] = (
                torch.tensor(
                    source_indices,
                    dtype=torch.long,
                    device=all_history_latents.device,
                )
            )
        if causal_retrieval_record_data is not None:
            if "decision_memory" in causal_retrieval_record_data:
                causal_retrieval_record_data["decision_memory"] = [
                    einops.rearrange(
                        decision_memories, "b t n d -> (b t) n d"
                    )
                ]
            projected_input["_precomputed_memory_record_data"] = (
                causal_retrieval_record_data
            )
        return (
            decision_memories,
            predicted_futures,
            records,
            dynamic_records,
            slot_weights,
            retained_source_slot_indices,
        )

    def _encode_input_multi_traj(
        self, normalized_batch: batch_type
    ) -> tuple[batch_type, batch_type]:
        """
        Should be called only when training history cross-attention modules
        args:
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_num, data_length, 3, image_size, image_size)
                "robot0_wrist_camera_feature": (batch_size, traj_num, data_length, 768) [Optional]
                "robot0_10d": (batch_size, traj_num, data_length, 10)
                "action0_10d": (batch_size, traj_num, data_length, 10)
                "future_0_wrist_camera": (batch_size, traj_num, data_length, 3, image_size, image_size)
                "third_person_camera": (batch_size, traj_num, data_length, 3, image_size, image_size) # For table-bin scenario
        return:
            data_dict:
                "global_cond": (batch_size, traj_num, token_num, global_cond_dim)
                "local_cond": (batch_size, traj_num, token_num, local_cond_dim)
                "noisy_action": (batch_size, traj_num, data_length, action_dim) # Noisy action latents
                "history_noisy_actions": (batch_size, traj_num, history_action_num_per_chunk, action_dim) # History latents, the noise will be 1-inference-step less than "noisy_action", to match the inference scenarios
                "history_img_features": (batch_size, traj_num, token_num, history_img_features_dim) # History image features
                "history_frame_indices": (batch_size, traj_num) # Episode-absolute source-frame index
                "history_noisy_future_img_features": (batch_size, traj_num, token_num, feature_dim)
                "memory_gate_val": (batch_size, traj_num)
                "step": (batch_size,)
            target:
                "action": (batch_size, traj_num, data_length, 8)
        """

        batch_size = next(iter(normalized_batch.values())).shape[0]
        traj_num = next(iter(normalized_batch.values())).shape[1]
        data_dict: dict[str, torch.Tensor] = {}

        if "traj_idx" in normalized_batch:
            traj_indices = normalized_batch["traj_idx"].reshape(
                batch_size, traj_num, -1
            )
            if traj_indices.shape[2] != 1:
                raise ValueError(
                    "each trajectory must contain one scalar traj_idx, got "
                    f"shape {tuple(normalized_batch['traj_idx'].shape)}"
                )
            data_dict["history_frame_indices"] = traj_indices[:, :, 0].to(
                dtype=torch.long
            )

        global_cond_dict = {
            k: v
            for k, v in normalized_batch.items()
            if k in self.global_cond_encoder.data_entry_names
        }

        global_cond_dict_feature = {
            k: v
            for k, v in normalized_batch.items()
            if "feature" in k and k.replace("_feature", "") in self.global_cond_encoder.data_entry_names
        }
        global_cond_dict.update(global_cond_dict_feature)

        data_dict["global_cond"] = einops.rearrange(
            self.global_cond_encoder.forward(
                dict_apply(
                    global_cond_dict,
                    lambda x: einops.rearrange(x, "b t ... -> (b t) ..."),
                )
            ),
            "(b t) ... -> b t ...",
            b=batch_size,
        )

        target: dict[str, torch.Tensor] = {}

        if self.local_cond_encoder is not None:
            local_cond_dict = {
                k: v
                for k, v in normalized_batch.items()
                if k in self.local_cond_encoder.data_entry_names
            }
            data_dict["local_cond"] = einops.rearrange(
                self.local_cond_encoder.forward(
                    dict_apply(
                        local_cond_dict,
                        lambda x: einops.rearrange(x, "b t ... -> (b t) ..."),
                    )
                ),
                "(b t) ... -> b t ...",
                b=batch_size,
            )

        train_timesteps: int = self.noise_scheduler.train_step_num
        inference_timesteps = self.noise_scheduler.inference_step_num
        step_ratio = train_timesteps // inference_timesteps

        data_dict["step"] = self.noise_scheduler.sample_training_timesteps(
            batch_size=batch_size,
            device=self.device,
            generator=self.torch_rng,
        )

        trajectory = einops.rearrange(
            self.action_decoder.encode(
                dict_apply(
                    {
                        k: normalized_batch[k]
                        for k in self.action_decoder.data_entry_names
                    },
                    lambda x: einops.rearrange(x, "b t ... -> (b t) ..."),
                )
            ),
            "(b t) ... -> b t ...",
            b=batch_size,
        ) # (batch_size, traj_num, traj_length, action_dim)

        action_noise = torch.randn_like(
            trajectory,
        )  # (batch_size, traj_num, traj_length, action_dim)
        data_dict["noisy_action"], target["action"] = self.noise_scheduler.get_noisy_action_and_target(
            trajectory,
            action_noise,
            data_dict["step"],
        )
        if (
            self.future_feature_predictor is not None
            and self.future_feature_predictor.use_action_condition
        ):
            # Adjacent training anchors are history_action_num_per_chunk steps
            # apart, matching the number of actions actually executed online.
            # Conditioning on the full prediction horizon would expose actions
            # that occur after the future-feature target.
            target["future_action_condition"] = trajectory[
                :, :, : self.history_action_num_per_chunk
            ]

        history_latent_diffusion_step = self.noise_scheduler.get_less_noisy_timesteps(data_dict["step"])

        if self.train_history_action_noise_level == "none":
            data_dict["history_noisy_actions"] = trajectory
        elif self.train_history_action_noise_level == "random":

            flattened_traj = einops.rearrange(
                trajectory,
                "b t ... -> (b t) ...",
            )
            history_action_noise = torch.randn_like(
                flattened_traj,
            )
            rand_timesteps = torch.randint(
                0,
                train_timesteps,
                (batch_size * traj_num,),
                device=self.device,
                generator=self.torch_rng,
            )
            data_dict["history_noisy_actions"] = einops.rearrange(
                self.noise_scheduler.get_noisy_action_and_target(
                    flattened_traj,
                    history_action_noise,
                    cast(torch.IntTensor, rand_timesteps),
                )[0],
                "(b t) ... -> b t ...",
                b=batch_size,
            )
            
        elif self.train_history_action_noise_level == "last_step":
            history_action_noise = torch.randn_like(
                trajectory,
            )
            data_dict["history_noisy_actions"], _ = self.noise_scheduler.get_noisy_action_and_target(
                trajectory,
                history_action_noise,
                cast(torch.IntTensor, history_latent_diffusion_step),
            )
        else:
            raise ValueError(f"Unknown history action noise level: {self.train_history_action_noise_level}")

        # Truncate the history noisy actions to the number of history actions per chunk
        data_dict["history_noisy_actions"] = data_dict["history_noisy_actions"][:, :, :self.history_action_num_per_chunk]

        if self.memory_gate is not None:
            normalized_batch_without_text = {
                k: v
                for k, v in normalized_batch.items()
                if not isinstance(v[0], str)
            }
            flattened_data_dict = dict_apply(
                normalized_batch_without_text,
                lambda x: einops.rearrange(x, "b t ... -> (b t) ..."),
            )
            val = self.memory_gate.get_gate_value(
                flattened_data_dict
            )
            data_dict["memory_gate_val"] = einops.rearrange(
                val,
                "(b t) ... -> b t ...",
                b=batch_size,
            ) # (batch_size, traj_num) 
            # print(f"Memory gate val: {data_dict['memory_gate_val']}, {data_dict['memory_gate_val'].shape}, \ntraj_idx: {normalized_batch['traj_idx']}, {normalized_batch['traj_idx'].shape}")

        if (
            self.history_img_feature_encoder is not None
            and not self.stores_state_conditioned_slots
        ):
            img_dict = {
                k: normalized_batch[k]
                for k in self.history_img_feature_encoder.data_entry_names if k in normalized_batch
            }
            img_feature_dict = {
                f"{k}_feature": normalized_batch[f"{k}_feature"]
                for k in self.history_img_feature_encoder.data_entry_names if f"{k}_feature" in normalized_batch
            }
            img_dict.update(img_feature_dict)
            history_img_features = self.history_img_feature_encoder.forward(img_dict)
            data_dict["history_img_features"] = history_img_features

        # Memory slots may retain all visual patches while future prediction
        # remains a single stable MAP token from the frozen image encoder.
        if (
            self.future_feature_predictor is not None
            and not self.compact_multiview_slot
        ):
            if self.future_img_feature_encoder is None:
                raise RuntimeError("future_img_feature_encoder is required")
            if self.future_img_feature_encoder is self.history_img_feature_encoder:
                future_target_features = data_dict["history_img_features"]
            else:
                future_img_dict = {
                    k: normalized_batch[k]
                    for k in self.future_img_feature_encoder.data_entry_names
                    if k in normalized_batch
                }
                future_img_feature_dict = {
                    f"{k}_feature": normalized_batch[f"{k}_feature"]
                    for k in self.future_img_feature_encoder.data_entry_names
                    if f"{k}_feature" in normalized_batch
                }
                future_img_dict.update(future_img_feature_dict)
                future_target_features = (
                    self.future_img_feature_encoder.forward(future_img_dict)
                )

            # Anchor j predicts the representative image feature of anchor j+1.
            # MultiTrajDataset samples anchors in temporal order, but the final
            # anchor (and any anchor followed by padding) has no valid target.
            future_features = torch.zeros_like(future_target_features)
            future_features[:, :-1] = future_target_features[:, 1:]
            future_valid = torch.zeros(
                (batch_size, traj_num),
                dtype=torch.bool,
                device=future_target_features.device,
            )
            future_valid[:, :-1] = (
                ~normalized_batch["entire_traj_is_padding"][:, :-1]
                & ~normalized_batch["entire_traj_is_padding"][:, 1:]
            )
            future_valid = self._mask_invalid_future_transitions(
                future_valid,
                normalized_batch.get("future_transition_valid"),
            )
            target["future_features"] = future_features
            target["future_feature_mask"] = future_valid[:, :, None].expand(
                -1, -1, future_target_features.shape[2]
            )


        data_dict["entire_traj_is_padding"] = normalized_batch["entire_traj_is_padding"]

        if self.max_training_traj_num > 0:
            valid_traj_indices: list[torch.Tensor] = []
            for i in range(batch_size):
                valid_traj_num = int(torch.sum(~data_dict["entire_traj_is_padding"][i]))
                assert not torch.any(data_dict["entire_traj_is_padding"][i, :valid_traj_num]), f"entire_traj_is_padding must be False for the first few trajectories, but got {data_dict['entire_traj_is_padding'][i, :valid_traj_num]}"
                sampled_traj_indices = torch.randint(0, valid_traj_num, (self.max_training_traj_num,), device=data_dict["entire_traj_is_padding"].device)
                # aggregated_indices = sampled_traj_indices + i * self.max_training_traj_num
                valid_traj_indices.append(sampled_traj_indices)
            all_valid_traj_indices = torch.stack(valid_traj_indices, dim=0)
            # print(f"{all_valid_traj_indices.shape=}")
            data_dict["training_traj_indices"] = all_valid_traj_indices # (batch_size, max_training_traj_num)

            batch_idx = torch.arange(batch_size, device=self.device)
            for k, v in target.items():
                target[k] = target[k][batch_idx, all_valid_traj_indices]
                # print(f"{k}: {target[k].shape}")
            # data_dict will be processed in MemoryTransformer._project_to_latent_space_multi_traj
            
        return data_dict, target


    def _start_streaming_training_state(
        self,
        batch_size: int,
        first_frame_indices: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        if (
            self.denoising_network.include_action_history
            and self.train_history_action_noise_level != "none"
        ):
            raise ValueError(
                "streaming training supports action history only when "
                "train_history_action_noise_level='none'. Noisy action "
                "history needs one stored latent per diffusion timestep and "
                "cannot be carried safely across independently noised chunks."
            )
        history_weights: list[list[torch.Tensor]] | None = None
        next_slot_weights: list[torch.Tensor] | None = None
        queues: list[PredictionErrorQueue] | None = None
        if self.training_slot_weight:
            if not self.prediction_error_queue_enabled:
                raise RuntimeError(
                    "streaming slot weighting requires prediction_error_queue"
                )
            queues = [
                self._new_prediction_error_queue()
                for _ in range(batch_size)
            ]
            history_weights = [[] for _ in range(batch_size)]
            next_slot_weights = [
                queues[idx]
                .first_slot_weight(reference[idx], first_frame_indices[idx])
                .detach()
                .clone()
                for idx in range(batch_size)
            ]
        self._streaming_training_state = StreamingTrainingState(
            history_latents=[[] for _ in range(batch_size)],
            history_frame_indices=[[] for _ in range(batch_size)],
            history_weights=history_weights,
            next_slot_weights=next_slot_weights,
            queues=queues,
        )

    @staticmethod
    def _slice_streaming_projected_input(
        projected_input: dict[str, Any],
        *,
        batch_size: int,
        input_traj_num: int,
        supervised_traj_num: int,
    ) -> dict[str, Any]:
        """Keep only supervised anchors after encoding one look-ahead anchor."""

        flat_indices = (
            torch.arange(
                batch_size,
                device=projected_input["x"].device,
            )[:, None]
            * input_traj_num
            + torch.arange(
                supervised_traj_num,
                device=projected_input["x"].device,
            )[None, :]
        ).reshape(-1)
        sliced: dict[str, Any] = {}
        for key, value in projected_input.items():
            if not isinstance(value, torch.Tensor):
                sliced[key] = value
                continue
            if (
                value.ndim >= 2
                and tuple(value.shape[:2])
                == (batch_size, input_traj_num)
            ):
                sliced[key] = value[:, :supervised_traj_num]
            elif value.ndim >= 1 and value.shape[0] == (
                batch_size * input_traj_num
            ):
                sliced[key] = value[flat_indices]
            else:
                sliced[key] = value
        return sliced

    def _build_streaming_training_memories(
        self,
        projected_input: dict[str, torch.Tensor],
        target: batch_type,
        valid_slot_mask: torch.Tensor,
        slot_frame_indices: torch.Tensor,
        supervised_traj_num: int,
        reset_state: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[PredictionErrorRecord],
        list[DynamicMemoryWeightRecord],
        dict[str, torch.Tensor] | None,
    ]:
        """Advance one causal stream chunk and detach its terminal bank.

        The bank and prediction-error queues persist across calls, while the
        autograd graph is intentionally truncated at every chunk boundary.
        """

        if self.future_feature_predictor is None:
            raise RuntimeError(
                "streaming training requires future_feature_predictor"
            )
        if not isinstance(
            self.denoising_network, (MemoryTransformer, OptimizedModule)
        ):
            raise TypeError("MemoryTransformer is required")

        batch_size, input_traj_num = valid_slot_mask.shape
        if supervised_traj_num <= 0 or supervised_traj_num > input_traj_num:
            raise ValueError(
                "streaming supervised trajectory count must be in "
                f"[1, {input_traj_num}], got {supervised_traj_num}"
            )
        all_history_latents = projected_input.get("all_history_latents")
        if all_history_latents is None:
            raise KeyError("projected input is missing all_history_latents")
        if all_history_latents.shape[:2] != (
            batch_size,
            input_traj_num,
        ):
            raise ValueError(
                "streaming history latents do not match batch/trajectory "
                f"dimensions: {tuple(all_history_latents.shape)}"
            )

        if reset_state:
            self._start_streaming_training_state(
                batch_size=batch_size,
                first_frame_indices=slot_frame_indices[:, 0],
                reference=all_history_latents[:, 0, 0, 0].float(),
            )
        state = self._streaming_training_state
        if state is None:
            raise RuntimeError(
                "streaming state is missing; the first chunk must request reset"
            )
        if len(state.history_latents) != batch_size:
            raise ValueError(
                "streaming batch size changed before the stream was completed"
            )

        history_len = self.denoising_network.max_history_len
        flat_batch_offsets = (
            torch.arange(batch_size, device=all_history_latents.device)
            * input_traj_num
        )
        decision_memory_by_time: list[torch.Tensor] = []
        predicted_future_by_time: list[torch.Tensor] = []
        records: list[PredictionErrorRecord] = []
        dynamic_records: list[DynamicMemoryWeightRecord] = []

        record_entries = list(self.denoising_network.record_data_entries)
        collect_epoch_attention_stats = (
            self.denoising_network.training
            and getattr(
                self.denoising_network,
                "record_history_attention_epoch_stats",
                False,
            )
        )
        if collect_epoch_attention_stats:
            for entry in (
                "history_cross_attention_frame_logits",
                "history_cross_attention_slot_weights",
            ):
                if entry not in record_entries:
                    record_entries.append(entry)
        causal_retrieval_record_data = (
            {entry: [] for entry in record_entries}
            if record_entries
            else None
        )

        for traj_idx in range(supervised_traj_num):
            flat_indices = flat_batch_offsets + traj_idx
            current_slot_weights = (
                [
                    weight.detach().clone()
                    for weight in state.next_slot_weights
                ]
                if state.next_slot_weights is not None
                else None
            )
            history_latent_rows: list[torch.Tensor] = []
            history_key_rows: list[torch.Tensor] = []
            history_mask_rows: list[torch.Tensor] = []
            history_bias_rows: list[torch.Tensor] = []
            history_stored_weight_rows: list[torch.Tensor] = []

            for batch_idx in range(batch_size):
                retained_latent_list = state.history_latents[batch_idx]
                retained_frame_list = state.history_frame_indices[batch_idx]
                retained_num = len(retained_latent_list)
                if retained_num != len(retained_frame_list):
                    raise RuntimeError(
                        "streaming latent/frame buffers are misaligned"
                    )
                padding_num = history_len - retained_num
                if padding_num < 0:
                    raise RuntimeError(
                        "streaming history exceeded max_history_len"
                    )
                padding_latents = all_history_latents.new_zeros(
                    (padding_num, *all_history_latents.shape[2:])
                )
                if retained_num:
                    retained_latents = torch.stack(retained_latent_list)
                    retained_frames = torch.as_tensor(
                        retained_frame_list,
                        dtype=torch.long,
                        device=all_history_latents.device,
                    )
                    if self.denoising_network.uses_absolute_history_time:
                        retained_time_embedding = (
                            self.denoising_network
                            .encode_history_frame_indices(retained_frames)
                        )
                    else:
                        relative_time_embedding = (
                            self.denoising_network.history_time_embedding
                        )
                        if relative_time_embedding is None:
                            raise RuntimeError(
                                "relative history time embedding is missing"
                            )
                        retained_time_embedding = relative_time_embedding[
                            0, -retained_num:
                        ]
                    positioned_retained_latents = (
                        self.denoising_network.add_history_key_encoding(
                            retained_latents.unsqueeze(0),
                            retained_time_embedding.unsqueeze(0),
                        ).squeeze(0)
                    )
                    if self.training_slot_weight:
                        assert state.history_weights is not None
                        retained_stored_weights = torch.stack(
                            state.history_weights[batch_idx]
                        ).to(dtype=torch.float32)
                        if retained_stored_weights.shape != (retained_num,):
                            raise RuntimeError(
                                "streaming latent/weight buffers are misaligned"
                            )
                        assert self.prediction_error_queue_config is not None
                        recent_slot_protection_num = (
                            self.dynamic_memory_weight_update_config
                            .recent_slot_protection_num
                            if self.dynamic_memory_weight_update_config
                            is not None
                            else 0
                        )
                        retained_biases = apply_recent_slot_attention_floor(
                            retained_stored_weights,
                            attention_bias_mode=(
                                self.history_attention_bias_mode
                            ),
                            recent_slot_protection_num=(
                                recent_slot_protection_num
                            ),
                            recent_slot_attention_floor=(
                                self.prediction_error_queue_config
                                .recent_slot_attention_floor
                            ),
                            fixed_slot_mask=self._frozen_first_slot_mask(
                                retained_frames
                            ),
                        )
                else:
                    retained_latents = all_history_latents.new_zeros(
                        (0, *all_history_latents.shape[2:])
                    )
                    positioned_retained_latents = retained_latents
                    if self.training_slot_weight:
                        retained_stored_weights = torch.zeros(
                            0,
                            dtype=torch.float32,
                            device=all_history_latents.device,
                        )
                        retained_biases = retained_stored_weights

                history_latent_rows.append(
                    torch.cat(
                        [
                            padding_latents,
                            (
                                retained_latents
                                if self.denoising_network.memory_retrieval_mode
                                == "perception_patches"
                                else positioned_retained_latents
                            ),
                        ],
                        dim=0,
                    )
                )
                if (
                    self.denoising_network.memory_retrieval_mode
                    == "perception_patches"
                ):
                    history_key_rows.append(
                        torch.cat(
                            [padding_latents, positioned_retained_latents],
                            dim=0,
                        )
                    )
                history_mask_rows.append(
                    torch.cat(
                        [
                            torch.zeros(
                                padding_num,
                                dtype=torch.bool,
                                device=all_history_latents.device,
                            ),
                            torch.ones(
                                retained_num,
                                dtype=torch.bool,
                                device=all_history_latents.device,
                            ),
                        ]
                    )
                )
                if self.training_slot_weight:
                    history_bias_rows.append(
                        torch.cat(
                            [
                                torch.zeros(
                                    padding_num,
                                    dtype=torch.float32,
                                    device=all_history_latents.device,
                                ),
                                retained_biases,
                            ]
                        )
                    )
                    history_stored_weight_rows.append(
                        torch.cat(
                            [
                                torch.zeros(
                                    padding_num,
                                    dtype=torch.float32,
                                    device=all_history_latents.device,
                                ),
                                retained_stored_weights,
                            ]
                        )
                    )

            history_latents = torch.stack(history_latent_rows)
            history_key_latents = (
                torch.stack(history_key_rows)
                if self.denoising_network.memory_retrieval_mode
                == "perception_patches"
                else None
            )
            history_mask = torch.stack(history_mask_rows)
            history_attention_bias = (
                torch.stack(history_bias_rows).detach()
                if self.training_slot_weight
                else None
            )
            history_stored_weights = (
                torch.stack(history_stored_weight_rows).detach()
                if self.training_slot_weight
                else None
            )
            if (
                self.training_slot_weight
                and self.dynamic_memory_weight_update_enabled
            ):
                assert history_attention_bias is not None
                history_attention_bias.requires_grad_(True)

            def select(name: str) -> torch.Tensor | None:
                value = projected_input.get(name)
                return None if value is None else value[flat_indices]

            decision_memory = self.denoising_network.memory_retriever(
                global_cond=projected_input["global_cond"][flat_indices],
                global_cond_mask=select("global_cond_mask"),
                local_cond=select("local_cond"),
                local_cond_mask=select("local_cond_mask"),
                history_latents=history_latents,
                history_key_latents=history_key_latents,
                history_mask=history_mask,
                history_attention_bias=(
                    history_attention_bias
                    if self.training_slot_weight and self.attention_bias
                    else None
                ),
                memory_gate_val=select("memory_gate_val"),
                skip_history=self.denoising_network.skip_history_attn,
                record_data_dict=causal_retrieval_record_data,
            )
            decision_memory_by_time.append(decision_memory)

            action_condition = (
                target["future_action_condition"][:, traj_idx]
                if self.future_feature_predictor.use_action_condition
                else None
            )
            predicted_future = self.future_feature_predictor(
                current_observation_tokens=(
                    self._pool_future_prediction_context(decision_memory)
                ),
                action_chunk=action_condition,
            )
            predicted_future_by_time.append(predicted_future)

            future_token_mask = target["future_feature_mask"][:, traj_idx]
            valid_future = future_token_mask.any(dim=-1)
            if self.training_slot_weight and valid_future.any():
                assert state.queues is not None
                assert state.next_slot_weights is not None
                token_errors = (
                    self.future_feature_predictor.cosine_prediction_error(
                        predicted_features=predicted_future,
                        target_features=target["future_features"][:, traj_idx],
                        detach_target=True,
                    )
                )
                per_sample_errors = (
                    token_errors * future_token_mask.to(token_errors.dtype)
                ).sum(dim=-1) / future_token_mask.sum(dim=-1).clamp_min(1)

                if (
                    self.dynamic_memory_weight_update_enabled
                    and history_mask.any()
                ):
                    assert history_attention_bias is not None
                    dynamic_error = per_sample_errors
                    if not self.attention_bias:
                        shadow_decision_memory = (
                            self.denoising_network.memory_retriever(
                                global_cond=projected_input["global_cond"]
                                [flat_indices],
                                global_cond_mask=select("global_cond_mask"),
                                local_cond=select("local_cond"),
                                local_cond_mask=select("local_cond_mask"),
                                history_latents=history_latents,
                                history_key_latents=history_key_latents,
                                history_mask=history_mask,
                                history_attention_bias=history_attention_bias,
                                memory_gate_val=select("memory_gate_val"),
                                skip_history=(
                                    self.denoising_network.skip_history_attn
                                ),
                                record_data_dict=None,
                            )
                        )
                        shadow_prediction = self.future_feature_predictor(
                            current_observation_tokens=(
                                self._pool_future_prediction_context(
                                    shadow_decision_memory
                                )
                            ),
                            action_chunk=action_condition,
                        )
                        shadow_errors = (
                            self.future_feature_predictor
                            .cosine_prediction_error(
                                predicted_features=shadow_prediction,
                                target_features=(
                                    target["future_features"][:, traj_idx]
                                ),
                                detach_target=True,
                            )
                        )
                        dynamic_error = (
                            shadow_errors
                            * future_token_mask.to(shadow_errors.dtype)
                        ).sum(dim=-1) / future_token_mask.sum(
                            dim=-1
                        ).clamp_min(1)
                    dynamic_gradient = torch.autograd.grad(
                        outputs=(
                            dynamic_error
                            * valid_future.to(dynamic_error.dtype)
                        ).sum(),
                        inputs=history_attention_bias,
                        retain_graph=self.attention_bias,
                        create_graph=False,
                    )[0]
                    dynamic_valid_mask = (
                        history_mask & valid_future[:, None]
                    )
                    for batch_idx, retained_frames in enumerate(
                        state.history_frame_indices
                    ):
                        retained_num = len(retained_frames)
                        for relative_idx, frame_index in enumerate(
                            retained_frames
                        ):
                            if self._is_frozen_first_slot_frame(frame_index):
                                dynamic_valid_mask[
                                    batch_idx,
                                    history_len - retained_num + relative_idx,
                                ] = False
                    assert self.dynamic_memory_weight_update_config is not None
                    assert self.prediction_error_queue_config is not None
                    assert history_stored_weights is not None
                    updated_weights, dynamic_record = (
                        apply_dynamic_memory_weight_update(
                            history_stored_weights,
                            dynamic_gradient,
                            dynamic_valid_mask,
                            step_size=(
                                self.dynamic_memory_weight_update_config
                                .step_size
                            ),
                            weight_min=(
                                self.prediction_error_queue_config.weight_min
                            ),
                            weight_max=(
                                self.prediction_error_queue_config.weight_max
                            ),
                        )
                    )
                    if dynamic_record.gradient.numel() > 0:
                        dynamic_records.append(dynamic_record)
                    assert state.history_weights is not None
                    for batch_idx, retained_frames in enumerate(
                        state.history_frame_indices
                    ):
                        if not bool(valid_future[batch_idx]):
                            continue
                        retained_num = len(retained_frames)
                        for relative_idx, frame_index in enumerate(
                            retained_frames
                        ):
                            state.history_weights[batch_idx][relative_idx] = (
                                self._freeze_first_slot_weight(
                                    updated_weights[
                                        batch_idx,
                                        history_len
                                        - retained_num
                                        + relative_idx,
                                    ]
                                    .detach()
                                    .clone(),
                                    frame_index,
                                )
                            )

                for batch_idx in range(batch_size):
                    if not bool(valid_future[batch_idx]):
                        continue
                    record = state.queues[batch_idx].score_and_push(
                        per_sample_errors[batch_idx]
                    )
                    next_frame_index = int(
                        slot_frame_indices[
                            batch_idx, traj_idx + 1
                        ].item()
                    )
                    state.next_slot_weights[batch_idx] = (
                        self._freeze_first_slot_weight(
                            record.initial_weight.detach().clone(),
                            next_frame_index,
                        )
                    )
                    records.append(record)

            for batch_idx in range(batch_size):
                if not bool(valid_slot_mask[batch_idx, traj_idx]):
                    continue
                if history_len <= 0:
                    continue
                latents = state.history_latents[batch_idx]
                frames = state.history_frame_indices[batch_idx]
                if not self.training_slot_weight and len(latents) >= history_len:
                    eviction_idx = select_random_training_slot_eviction_index(
                        len(latents), device=all_history_latents.device
                    )
                    latents.pop(eviction_idx)
                    frames.pop(eviction_idx)
                latents.append(all_history_latents[batch_idx, traj_idx])
                frames.append(int(slot_frame_indices[batch_idx, traj_idx]))
                if self.training_slot_weight:
                    assert state.history_weights is not None
                    assert current_slot_weights is not None
                    weights = state.history_weights[batch_idx]
                    weights.append(
                        current_slot_weights[batch_idx]
                    )
                    if len(latents) > history_len:
                        if self.dynamic_memory_weight_update_config is None:
                            evict_lowest = False
                            recent_slot_protection_num = 0
                        else:
                            evict_lowest = (
                                self.dynamic_memory_weight_update_config
                                .evict_lowest_weight_when_full
                            )
                            recent_slot_protection_num = (
                                self.dynamic_memory_weight_update_config
                                .recent_slot_protection_num
                            )
                        eviction_idx = select_memory_slot_eviction_index(
                            weights,
                            evict_lowest_weight_when_full=evict_lowest,
                            recent_slot_protection_num=(
                                recent_slot_protection_num
                            ),
                        )
                        latents.pop(eviction_idx)
                        frames.pop(eviction_idx)
                        weights.pop(eviction_idx)

        decision_memories = torch.stack(decision_memory_by_time, dim=1)
        predicted_futures = torch.stack(predicted_future_by_time, dim=1)
        if causal_retrieval_record_data is not None:
            if "decision_memory" in causal_retrieval_record_data:
                causal_retrieval_record_data["decision_memory"] = [
                    einops.rearrange(
                        decision_memories, "b t n d -> (b t) n d"
                    )
                ]
            projected_input["_precomputed_memory_record_data"] = (
                causal_retrieval_record_data
            )

        # Truncated BPTT boundary: preserve values and queue state, not graphs.
        state.history_latents = [
            [latent.detach() for latent in row]
            for row in state.history_latents
        ]

        slot_table: dict[str, torch.Tensor] | None = None
        if self.training_slot_weight:
            assert state.history_weights is not None
            table_weight = torch.zeros(
                (batch_size, history_len),
                dtype=torch.float32,
                device=all_history_latents.device,
            )
            table_valid = torch.zeros_like(table_weight, dtype=torch.bool)
            table_frame = torch.full_like(
                table_weight, -1, dtype=torch.long
            )
            for batch_idx in range(batch_size):
                retained_num = len(state.history_weights[batch_idx])
                if retained_num == 0:
                    continue
                table_weight[batch_idx, -retained_num:] = torch.stack(
                    state.history_weights[batch_idx]
                ).to(table_weight)
                table_valid[batch_idx, -retained_num:] = True
                table_frame[batch_idx, -retained_num:] = torch.as_tensor(
                    state.history_frame_indices[batch_idx],
                    dtype=torch.long,
                    device=table_frame.device,
                )
            slot_table = {
                "weight": table_weight,
                "valid_mask": table_valid,
                "source_slot_index": table_frame.clone(),
                "anchor_frame_index": table_frame,
            }
        return (
            decision_memories,
            predicted_futures,
            records,
            dynamic_records,
            slot_table,
        )

    def _reproject_detached_history(
        self,
        raw_state: PrefixTrainingState | StreamingRawHistoryState,
        *,
        state_name: str,
    ) -> None:
        """Rebuild retained slots from detached frozen-encoder outputs."""

        stream_state = self._streaming_training_state
        if stream_state is None:
            raise RuntimeError(f"{state_name} training state is not initialized")
        if not isinstance(
            self.denoising_network, (MemoryTransformer, OptimizedModule)
        ):
            raise TypeError("MemoryTransformer is required")
        network = cast(
            MemoryTransformer, cast(Any, self.denoising_network)
        )
        projector = network.history_img_features_projector
        if projector is None:
            raise RuntimeError(
                f"{state_name} training requires a history image projector"
            )

        for batch_idx, frames in enumerate(
            stream_state.history_frame_indices
        ):
            retained_num = len(frames)
            if retained_num == 0:
                stream_state.history_latents[batch_idx] = []
                continue
            raw_images = raw_state.history_img_features[batch_idx]
            if len(raw_images) != retained_num:
                raise RuntimeError(
                    f"{state_name} raw image bank is not aligned with "
                    "frame indices"
                )
            latent_groups: list[torch.Tensor] = []
            if network.include_action_history:
                raw_actions = raw_state.history_actions[batch_idx]
                if len(raw_actions) != retained_num:
                    raise RuntimeError(
                        f"{state_name} raw action bank is not aligned with "
                        "frames"
                    )
                action_latents = network.action_projector(
                    torch.stack(raw_actions)
                )
                start_idx = (
                    network.input_pos_embedding.shape[1]
                    - network.action_token_num
                )
                end_idx = (
                    start_idx + network.history_action_num_per_chunk
                )
                action_latents = (
                    action_latents
                    + network.input_pos_embedding[
                        :, start_idx:end_idx, :
                    ]
                )
                latent_groups.append(action_latents)
            latent_groups.append(projector(torch.stack(raw_images)))
            projected = torch.cat(latent_groups, dim=1)
            stream_state.history_latents[batch_idx] = list(
                projected.unbind(dim=0)
            )

    def _reproject_prefix_history(self) -> None:
        """Rebuild prefix slots from detached raw encoder outputs."""

        prefix_state = self._prefix_training_state
        if prefix_state is None:
            raise RuntimeError("prefix training state is not initialized")
        self._reproject_detached_history(
            prefix_state,
            state_name="prefix",
        )

    def _streaming_history_reprojection_enabled(self) -> bool:
        """Whether streaming can safely persist raw visual features."""

        if self.stores_state_conditioned_slots:
            return False
        encoder = self.history_img_feature_encoder
        if encoder is None:
            return False
        if any(parameter.requires_grad for parameter in encoder.parameters()):
            return False
        if not isinstance(
            self.denoising_network, (MemoryTransformer, OptimizedModule)
        ):
            return False
        network = cast(MemoryTransformer, cast(Any, self.denoising_network))
        return network.history_img_features_projector is not None

    def compute_prefix_loss(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """Compute one anchor while keeping only detached raw slot state."""

        if self.skip_memory:
            raise ValueError("prefix training requires memory to be enabled")
        if self.stores_state_conditioned_slots:
            raise ValueError(
                "prefix currently requires raw visual slots; disable compact "
                "three-view/state-conditioned slot storage"
            )
        if self.history_img_feature_encoder is None:
            raise ValueError("prefix requires a history image encoder")
        if any(
            parameter.requires_grad
            for parameter in self.history_img_feature_encoder.parameters()
        ):
            raise ValueError(
                "prefix raw-feature state requires a frozen image encoder"
            )
        reset_state = bool(normalized_batch.pop("_prefix_reset"))
        final_anchor = bool(normalized_batch.pop("_prefix_final"))
        normalized_batch.pop("prefix_sequence", None)
        self.shared_model_manager.clear_cache()

        supervised_mask = normalized_batch.get("prefix_supervised_mask")
        feature_valid = normalized_batch.get("prefix_feature_valid")
        if (
            not isinstance(supervised_mask, torch.Tensor)
            or supervised_mask.ndim != 2
            or supervised_mask.shape[1] != 2
            or not isinstance(feature_valid, torch.Tensor)
            or feature_valid.shape != supervised_mask.shape
        ):
            raise ValueError(
                "each prefix anchor call requires current plus look-ahead "
                "masks with shape [batch, 2]"
            )
        current_valid = supervised_mask[:, 0].bool()
        next_feature_valid = feature_valid[:, 1].bool()

        action_key_names = self.action_decoder.data_entry_names
        action_traj_length = normalized_batch[action_key_names[0]].shape[2]
        data_dict, target = self._encode_input_multi_traj(normalized_batch)
        data_dict["_streaming_training"] = True
        if isinstance(self.denoising_network, OptimizedModule):
            self.denoising_network = cast(
                MemoryTransformer, cast(Any, self.denoising_network)
            )
        else:
            assert isinstance(self.denoising_network, MemoryTransformer)
        network = self.denoising_network
        projected_input = network.prepare_parallel_forward(data_dict)
        if "future_features" not in target:
            raise RuntimeError("prefix requires future feature targets")
        # The second trajectory is target-only and deliberately marked as
        # action padding. It is nevertheless a valid visual target.
        future_valid = current_valid & next_feature_valid
        target["future_feature_mask"][:, 0] = future_valid[:, None].expand(
            -1, target["future_feature_mask"].shape[-1]
        )

        batch_size = supervised_mask.shape[0]
        input_traj_num = supervised_mask.shape[1]
        traj_indices = normalized_batch["traj_idx"].reshape(
            batch_size, input_traj_num, -1
        )
        if traj_indices.shape[2] != 1:
            raise ValueError("each prefix anchor must have one traj_idx")
        slot_frame_indices = traj_indices[:, :, 0].to(
            device=projected_input["x"].device,
            dtype=torch.long,
        )
        all_history_latents = projected_input.get("all_history_latents")
        raw_image_features = data_dict.get("history_img_features")
        if all_history_latents is None or raw_image_features is None:
            raise RuntimeError(
                "prefix requires projected and raw history image features"
            )

        if reset_state:
            self._start_streaming_training_state(
                batch_size=batch_size,
                first_frame_indices=slot_frame_indices[:, 0],
                reference=all_history_latents[:, 0, 0, 0].float(),
            )
            self._prefix_training_state = PrefixTrainingState(
                history_img_features=[[] for _ in range(batch_size)],
                history_actions=[[] for _ in range(batch_size)],
            )
        stream_state = self._streaming_training_state
        prefix_state = self._prefix_training_state
        if stream_state is None or prefix_state is None:
            raise RuntimeError("prefix state is missing at a non-initial anchor")
        if len(stream_state.history_frame_indices) != batch_size:
            raise ValueError("prefix batch size changed within one sample")

        old_frames = [
            list(frames) for frames in stream_state.history_frame_indices
        ]
        old_image_maps = [
            {
                frame: feature
                for frame, feature in zip(
                    frames, prefix_state.history_img_features[batch_idx]
                )
            }
            for batch_idx, frames in enumerate(old_frames)
        ]
        old_action_maps = [
            {
                frame: action
                for frame, action in zip(
                    frames, prefix_state.history_actions[batch_idx]
                )
            }
            for batch_idx, frames in enumerate(old_frames)
        ]
        self._reproject_prefix_history()

        (
            decision_memory,
            predicted_future,
            prediction_error_records,
            dynamic_records,
            slot_table,
        ) = self._build_streaming_training_memories(
            projected_input=projected_input,
            target=target,
            valid_slot_mask=supervised_mask.bool(),
            slot_frame_indices=slot_frame_indices,
            supervised_traj_num=1,
            reset_state=False,
        )

        current_raw_images = raw_image_features[:, 0].detach()
        current_raw_actions = data_dict["history_noisy_actions"][:, 0].detach()
        for batch_idx in range(batch_size):
            image_map = old_image_maps[batch_idx]
            action_map = old_action_maps[batch_idx]
            if bool(current_valid[batch_idx]):
                frame = int(slot_frame_indices[batch_idx, 0].item())
                image_map[frame] = current_raw_images[batch_idx]
                action_map[frame] = current_raw_actions[batch_idx]
            retained_frames = stream_state.history_frame_indices[batch_idx]
            prefix_state.history_img_features[batch_idx] = [
                image_map[frame].detach() for frame in retained_frames
            ]
            prefix_state.history_actions[batch_idx] = [
                action_map[frame].detach() for frame in retained_frames
            ]

        projected_input = self._slice_streaming_projected_input(
            projected_input,
            batch_size=batch_size,
            input_traj_num=input_traj_num,
            supervised_traj_num=1,
        )
        projected_input["decision_memory"] = einops.rearrange(
            decision_memory, "b t n d -> (b t) n d"
        )
        model_output = network.finish_parallel_forward(
            projected_input,
            batch_size=batch_size,
            traj_num=1,
        )

        loss: dict[str, Any] = {}
        loss["future_prediction"] = (
            self.future_feature_predictor.cosine_prediction_loss(
                predicted_features=predicted_future,
                target_features=target["future_features"][:, :1],
                target_mask=target["future_feature_mask"][:, :1],
                detach_target=True,
            )
        )
        valid_anchor_mask = current_valid[:, None]
        memory_gate_val = data_dict.get("memory_gate_val")
        if memory_gate_val is not None:
            memory_gate_val = memory_gate_val[:, :1] * valid_anchor_mask
        action_loss = F.mse_loss(
            model_output["action"],
            target["action"][:, :1],
            reduction="none",
        )
        action_loss = einops.reduce(
            action_loss, "b t ... -> b t", "mean"
        )
        action_loss = action_loss * valid_anchor_mask

        if "action_is_error" in normalized_batch:
            traj_error_mask = torch.zeros(
                action_traj_length, device=self.device
            )
            traj_error_mask[
                self.action_no_error_range[0] : self.action_no_error_range[1]
            ] = 1
            traj_is_error = einops.reduce(
                normalized_batch["action_is_error"][:, :1]
                * traj_error_mask[None, None, :],
                "b t ... -> b t",
                "any",
            )
            action_loss = action_loss * (~traj_is_error)
            if memory_gate_val is not None:
                memory_gate_val = memory_gate_val * (~traj_is_error)

        if "action_is_critical" in normalized_batch:
            traj_is_critical = torch.any(
                normalized_batch["action_is_critical"][:, :1], dim=2
            ).squeeze(-1)
            critical_action_loss = action_loss * traj_is_critical
            if (critical_action_loss != 0).any():
                loss["critical_action"] = critical_action_loss.sum() / (
                    critical_action_loss != 0
                ).sum()
            else:
                loss["critical_action"] = critical_action_loss.sum()
            if memory_gate_val is not None:
                critical_memory_gate_val = (
                    memory_gate_val * traj_is_critical
                )
                critical_valid_mask = traj_is_critical & (action_loss != 0)
                if critical_valid_mask.any():
                    loss["critical_memory_gate_val"] = (
                        critical_memory_gate_val.sum()
                        / critical_valid_mask.sum()
                    )
                    loss["critical_binary_memory_gate_val"] = (
                        (critical_memory_gate_val > 0.5).sum()
                        / critical_valid_mask.sum()
                    )
                else:
                    loss["critical_memory_gate_val"] = (
                        critical_memory_gate_val.sum()
                    )
                    loss["critical_binary_memory_gate_val"] = (
                        (critical_memory_gate_val > 0.5).sum()
                    )

        if (action_loss != 0).any():
            loss["action"] = action_loss.sum() / (action_loss != 0).sum()
        else:
            loss["action"] = action_loss.sum() / action_loss.numel()
        if memory_gate_val is not None:
            memory_gate_valid_num = (action_loss != 0).sum()
            if memory_gate_valid_num == 0:
                memory_gate_valid_num = memory_gate_val.numel()
            loss["memory_gate_val"] = (
                memory_gate_val.sum() / memory_gate_valid_num
            )
            loss["binary_memory_gate_val"] = (
                (memory_gate_val > 0.5).sum() / memory_gate_valid_num
            )

        if (
            self.prediction_error_queue_config is not None
            and self.prediction_error_queue_config.record_training_stats
            and slot_table is not None
            and slot_table["valid_mask"].any()
        ):
            loss["prediction_error_queue_stats"] = (
                self._summarize_prediction_error_records(
                    records=prediction_error_records,
                    dynamic_records=dynamic_records,
                    slot_weights=slot_table["weight"],
                    valid_slot_mask=slot_table["valid_mask"],
                )
            )
        if (
            self.prediction_error_queue_config is not None
            and self.prediction_error_queue_config.record_slot_weights
            and slot_table is not None
        ):
            loss["prediction_error_slot_weight_table"] = slot_table

        if final_anchor:
            self._streaming_training_state = None
            self._prefix_training_state = None
        return loss

    def compute_streaming_loss(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """Compute one bounded chunk while carrying its causal memory state."""

        if self.skip_memory:
            raise ValueError("streaming training requires memory to be enabled")
        reset_state = bool(normalized_batch.pop("_streaming_reset"))
        final_chunk = bool(normalized_batch.pop("_streaming_final"))
        supervised_traj_num = int(
            normalized_batch.pop("_streaming_supervised_length")
        )
        normalized_batch.pop("streaming_sequence", None)
        self.shared_model_manager.clear_cache()

        action_key_names = self.action_decoder.data_entry_names
        action_traj_length = normalized_batch[action_key_names[0]].shape[2]
        global_key = self.global_cond_encoder.data_entry_names[0]
        global_key = (
            f"{global_key}_feature"
            if f"{global_key}_feature" in normalized_batch
            else global_key
        )
        batch_size, input_traj_num = normalized_batch[global_key].shape[:2]
        if supervised_traj_num > input_traj_num:
            raise ValueError(
                "streaming chunk needs at least as many encoded anchors as "
                "supervised anchors"
            )

        data_dict, target = self._encode_input_multi_traj(normalized_batch)
        data_dict["_streaming_training"] = True
        if isinstance(self.denoising_network, OptimizedModule):
            self.denoising_network = cast(
                MemoryTransformer, cast(Any, self.denoising_network)
            )
        else:
            assert isinstance(self.denoising_network, MemoryTransformer)
        projected_input = self.denoising_network.prepare_parallel_forward(
            data_dict
        )
        if self.compact_multiview_slot:
            self._set_compact_future_targets(
                target,
                projected_input,
                normalized_batch["entire_traj_is_padding"],
            )
        if "future_features" not in target:
            raise RuntimeError(
                "streaming training requires future feature targets"
            )

        traj_indices = normalized_batch["traj_idx"].reshape(
            batch_size, input_traj_num, -1
        )
        if traj_indices.shape[2] != 1:
            raise ValueError("each streaming anchor must have one traj_idx")
        slot_frame_indices = traj_indices[:, :, 0].to(
            device=projected_input["x"].device,
            dtype=torch.long,
        )
        reproject_history = self._streaming_history_reprojection_enabled()
        raw_history_state: StreamingRawHistoryState | None = None
        old_frames: list[list[int]] = []
        old_image_maps: list[dict[int, torch.Tensor]] = []
        old_action_maps: list[dict[int, torch.Tensor]] = []
        raw_image_features = data_dict.get("history_img_features")
        if reproject_history:
            if raw_image_features is None:
                raise RuntimeError(
                    "streaming history reprojection requires raw image "
                    "features"
                )
            if reset_state:
                raw_history_state = StreamingRawHistoryState(
                    history_img_features=[[] for _ in range(batch_size)],
                    history_actions=[[] for _ in range(batch_size)],
                )
                self._streaming_raw_history_state = raw_history_state
            else:
                raw_history_state = self._streaming_raw_history_state
                if raw_history_state is None:
                    raise RuntimeError(
                        "streaming raw history state is missing at a "
                        "non-initial chunk"
                    )
                stream_state = self._streaming_training_state
                if stream_state is None:
                    raise RuntimeError(
                        "streaming state is missing at a non-initial chunk"
                    )
                if len(raw_history_state.history_img_features) != batch_size:
                    raise ValueError(
                        "streaming batch size changed before the stream "
                        "was completed"
                    )
                old_frames = [
                    list(frames)
                    for frames in stream_state.history_frame_indices
                ]
                old_image_maps = [
                    {
                        frame: feature
                        for frame, feature in zip(
                            frames,
                            raw_history_state.history_img_features[batch_idx],
                        )
                    }
                    for batch_idx, frames in enumerate(old_frames)
                ]
                old_action_maps = [
                    {
                        frame: action
                        for frame, action in zip(
                            frames,
                            raw_history_state.history_actions[batch_idx],
                        )
                    }
                    for batch_idx, frames in enumerate(old_frames)
                ]
                self._reproject_detached_history(
                    raw_history_state,
                    state_name="streaming",
                )
        (
            decision_memory,
            predicted_future,
            prediction_error_records,
            dynamic_records,
            slot_table,
        ) = self._build_streaming_training_memories(
            projected_input=projected_input,
            target=target,
            valid_slot_mask=(
                ~normalized_batch["entire_traj_is_padding"]
            ),
            slot_frame_indices=slot_frame_indices,
            supervised_traj_num=supervised_traj_num,
            reset_state=reset_state,
        )
        if reproject_history:
            assert raw_history_state is not None
            assert raw_image_features is not None
            stream_state = self._streaming_training_state
            if stream_state is None:
                raise RuntimeError("streaming state disappeared during a chunk")
            if reset_state:
                old_frames = [[] for _ in range(batch_size)]
                old_image_maps = [{} for _ in range(batch_size)]
                old_action_maps = [{} for _ in range(batch_size)]
            current_raw_images = raw_image_features[
                :, :supervised_traj_num
            ].detach()
            current_raw_actions = (
                data_dict["history_noisy_actions"][
                    :, :supervised_traj_num
                ].detach()
                if self.denoising_network.include_action_history
                else None
            )
            valid_slot_mask = ~normalized_batch[
                "entire_traj_is_padding"
            ][:, :supervised_traj_num]
            for batch_idx in range(batch_size):
                image_map = old_image_maps[batch_idx]
                action_map = old_action_maps[batch_idx]
                for traj_idx in range(supervised_traj_num):
                    if not bool(valid_slot_mask[batch_idx, traj_idx]):
                        continue
                    frame = int(
                        slot_frame_indices[batch_idx, traj_idx].item()
                    )
                    image_map[frame] = current_raw_images[
                        batch_idx, traj_idx
                    ]
                    if current_raw_actions is not None:
                        action_map[frame] = current_raw_actions[
                            batch_idx, traj_idx
                        ]
                retained_frames = stream_state.history_frame_indices[
                    batch_idx
                ]
                raw_history_state.history_img_features[batch_idx] = [
                    image_map[frame].detach() for frame in retained_frames
                ]
                if self.denoising_network.include_action_history:
                    raw_history_state.history_actions[batch_idx] = [
                        action_map[frame].detach()
                        for frame in retained_frames
                    ]
        projected_input = self._slice_streaming_projected_input(
            projected_input,
            batch_size=batch_size,
            input_traj_num=input_traj_num,
            supervised_traj_num=supervised_traj_num,
        )
        projected_input["decision_memory"] = einops.rearrange(
            decision_memory, "b t n d -> (b t) n d"
        )
        model_output = self.denoising_network.finish_parallel_forward(
            projected_input,
            batch_size=batch_size,
            traj_num=supervised_traj_num,
        )

        loss: dict[str, Any] = {}
        loss["future_prediction"] = (
            self.future_feature_predictor.cosine_prediction_loss(
                predicted_features=predicted_future,
                target_features=target["future_features"][
                    :, :supervised_traj_num
                ],
                target_mask=target["future_feature_mask"][
                    :, :supervised_traj_num
                ],
                detach_target=True,
            )
        )

        valid_anchor_mask = ~normalized_batch[
            "entire_traj_is_padding"
        ][:, :supervised_traj_num]
        memory_gate_val = data_dict.get("memory_gate_val")
        if memory_gate_val is not None:
            memory_gate_val = (
                memory_gate_val[:, :supervised_traj_num]
                * valid_anchor_mask
            )
        action_loss = F.mse_loss(
            model_output["action"],
            target["action"][:, :supervised_traj_num],
            reduction="none",
        )
        action_loss = einops.reduce(
            action_loss, "b t ... -> b t", "mean"
        )
        action_loss = action_loss * valid_anchor_mask

        traj_is_error: torch.Tensor | None = None
        if "action_is_error" in normalized_batch:
            traj_error_mask = torch.zeros(
                action_traj_length, device=self.device
            )
            traj_error_mask[
                self.action_no_error_range[0] : self.action_no_error_range[1]
            ] = 1
            traj_is_error = einops.reduce(
                normalized_batch["action_is_error"][
                    :, :supervised_traj_num
                ]
                * traj_error_mask[None, None, :],
                "b t ... -> b t",
                "any",
            )
            action_loss = action_loss * (~traj_is_error)
            if memory_gate_val is not None:
                memory_gate_val = memory_gate_val * (~traj_is_error)

        traj_is_critical: torch.Tensor | None = None
        if "action_is_critical" in normalized_batch:
            traj_is_critical = torch.any(
                normalized_batch["action_is_critical"][
                    :, :supervised_traj_num
                ],
                dim=2,
            ).squeeze(-1)
            critical_action_loss = action_loss * traj_is_critical
            if (critical_action_loss != 0).any():
                loss["critical_action"] = critical_action_loss.sum() / (
                    critical_action_loss != 0
                ).sum()
            else:
                loss["critical_action"] = critical_action_loss.sum()
            if memory_gate_val is not None:
                critical_memory_gate_val = (
                    memory_gate_val * traj_is_critical
                )
                critical_valid_mask = (
                    traj_is_critical & (action_loss != 0)
                )
                if critical_valid_mask.any():
                    loss["critical_memory_gate_val"] = (
                        critical_memory_gate_val.sum()
                        / critical_valid_mask.sum()
                    )
                    loss["critical_binary_memory_gate_val"] = (
                        (critical_memory_gate_val > 0.5).sum()
                        / critical_valid_mask.sum()
                    )
                else:
                    loss["critical_memory_gate_val"] = (
                        critical_memory_gate_val.sum()
                    )
                    loss["critical_binary_memory_gate_val"] = (
                        (critical_memory_gate_val > 0.5).sum()
                    )

        if (action_loss != 0).any():
            loss["action"] = action_loss.sum() / (
                action_loss != 0
            ).sum()
        else:
            loss["action"] = action_loss.sum() / action_loss.numel()

        if memory_gate_val is not None:
            memory_gate_valid_num = (action_loss != 0).sum()
            if memory_gate_valid_num == 0:
                memory_gate_valid_num = memory_gate_val.numel()
            loss["memory_gate_val"] = (
                memory_gate_val.sum() / memory_gate_valid_num
            )
            loss["binary_memory_gate_val"] = (
                (memory_gate_val > 0.5).sum() / memory_gate_valid_num
            )

        if (
            self.prediction_error_queue_config is not None
            and self.prediction_error_queue_config.record_training_stats
            and slot_table is not None
            and slot_table["valid_mask"].any()
        ):
            loss["prediction_error_queue_stats"] = (
                self._summarize_prediction_error_records(
                    records=prediction_error_records,
                    dynamic_records=dynamic_records,
                    slot_weights=slot_table["weight"],
                    valid_slot_mask=slot_table["valid_mask"],
                )
            )
        if (
            self.prediction_error_queue_config is not None
            and self.prediction_error_queue_config.record_slot_weights
            and slot_table is not None
        ):
            loss["prediction_error_slot_weight_table"] = slot_table

        if final_chunk:
            self._streaming_training_state = None
            self._streaming_raw_history_state = None
        return loss



    def compute_loss(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """
        If self.skip_memory, will directly call the method in the superclass:
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
                "robot0_wrist_camera_feature": (batch_size, traj_length, 768) [Optional]
                "robot0_10d": (batch_size, traj_length, 8)
                "action0_10d": (batch_size, traj_length, 8)
                "future_0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
                "third_person_camera": (batch_size, traj_length, 3, image_size, image_size) # For table-bin scenario
                "action_is_error": (batch_size, traj_length)
                "action_is_critical": (batch_size, traj_length)

        If not self.skip_memory:
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size)
                "robot0_wrist_camera_feature": (batch_size, traj_num, traj_length, 768) [Optional]
                "robot0_10d": (batch_size, traj_num, traj_length, 8)
                "action0_10d": (batch_size, traj_num, traj_length, 8)
                "future_0_wrist_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size)
                "third_person_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size) # For table-bin scenario
                "entire_traj_is_padding": (batch_size, traj_num)
                "action_is_error": (batch_size, traj_num, traj_length)
                "action_is_critical": (batch_size, traj_num, traj_length) # Optional
        """
        if "_prefix_reset" in normalized_batch:
            return self.compute_prefix_loss(normalized_batch)
        if "_streaming_supervised_length" in normalized_batch:
            return self.compute_streaming_loss(normalized_batch)

        if self.skip_memory:
            return super().compute_loss(normalized_batch)

        # print(f"normalized_batch keys: {normalized_batch.keys()}")

        loss = {}

        self.shared_model_manager.clear_cache() # Clear the cache before every forward pass


        # print(normalized_batch["robot0_wrist_camera"][0,0].min(), normalized_batch["robot0_wrist_camera"][0,0].max())
        # img = normalized_batch["robot0_wrist_camera"][0,0].cpu().numpy()
        # cv2_img = img.squeeze(0).transpose(1, 2, 0)  # [image_size, image_size, 3]
        # cv2_img = (cv2_img) * 255
        # cv2_img = cv2.cvtColor(cv2_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"robot0_wrist_camera.png", cv2_img)

        # third_person_camera = normalized_batch["third_person_camera"][0,0].cpu().numpy()
        # print(third_person_camera.min(), third_person_camera.max())
        # cv2_img = third_person_camera.squeeze(0).transpose(1, 2, 0)  # [image_size, image_size, 3]
        # cv2_img = (cv2_img) * 255
        # cv2_img = cv2.cvtColor(cv2_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"third_person_camera.png", cv2_img)

        # exit()

        action_key_names = self.action_decoder.data_entry_names
        action_traj_length = normalized_batch[action_key_names[0]].shape[2]

        global_cond_key_names = self.global_cond_encoder.data_entry_names
        global_cond_valid_key_name = global_cond_key_names[0]
        if f"{global_cond_valid_key_name}_feature" in normalized_batch:
            global_cond_valid_key_name = f"{global_cond_valid_key_name}_feature"

        traj_num = normalized_batch[global_cond_valid_key_name].shape[1]
        batch_size = normalized_batch[global_cond_valid_key_name].shape[0]

        if "local_cond" in normalized_batch and len(normalized_batch["local_cond"]) > 0:
            assert (batch_size, traj_num) == next(
                iter(normalized_batch["local_cond"].values())
            ).shape[:2], "Please make sure you are using multi-trajectory dataset"

        assert (batch_size, traj_num) == normalized_batch[action_key_names[0]].shape[
            :2
        ], f"Please make sure you are using multi-trajectory dataset. (batch_size: {batch_size}, traj_num: {traj_num}, action_shape: {normalized_batch[action_key_names[0]].shape})"

        data_dict, target = self._encode_input_multi_traj(normalized_batch)

        # self._add_random_masks(data_dict)
        
        if isinstance(self.denoising_network, OptimizedModule):
            # After torch compile: Just fix the type of the denoising network for type checking.
            self.denoising_network = cast(MemoryTransformer, cast(Any, self.denoising_network))
        else:
            assert isinstance(
                self.denoising_network, MemoryTransformer
            ), "MemoryTransformer is required for memory-based policy"

        predicted_future: torch.Tensor | None = None
        prediction_error_records: list[PredictionErrorRecord] = []
        dynamic_memory_weight_records: list[DynamicMemoryWeightRecord] = []
        slot_weights: torch.Tensor | None = None
        retained_source_slot_indices: torch.Tensor | None = None
        slot_frame_indices: torch.Tensor | None = None
        projected_input: dict[str, torch.Tensor] | None = None
        if self.compact_multiview_slot:
            projected_input = self.denoising_network.prepare_parallel_forward(
                data_dict
            )
            self._set_compact_future_targets(
                target,
                projected_input,
                normalized_batch["entire_traj_is_padding"],
                normalized_batch.get("future_transition_valid"),
            )

        if self.prediction_error_queue_enabled or not self.training_slot_weight:
            if projected_input is None:
                projected_input = self.denoising_network.prepare_parallel_forward(
                    data_dict
                )
            valid_slot_mask = ~normalized_batch["entire_traj_is_padding"]
            if "traj_idx" not in normalized_batch:
                raise KeyError(
                    "causal training memory requires the dataset's absolute "
                    "traj_idx metadata"
                )
            traj_indices = normalized_batch["traj_idx"]
            if traj_indices.shape[:2] != (batch_size, traj_num):
                raise ValueError(
                    "traj_idx must start with batch and trajectory dimensions "
                    f"({batch_size}, {traj_num}), got {tuple(traj_indices.shape)}"
                )
            slot_frame_indices_3d = traj_indices.reshape(
                batch_size, traj_num, -1
            )
            if slot_frame_indices_3d.shape[2] != 1:
                raise ValueError(
                    "each trajectory must contain one scalar traj_idx, got "
                    f"shape {tuple(traj_indices.shape)}"
                )
            slot_frame_indices = slot_frame_indices_3d[:, :, 0].to(
                device=projected_input["x"].device,
                dtype=torch.long,
            )
            pair_start_mask = normalized_batch.get("pair_start_mask")
            (
                decision_memory,
                predicted_future,
                prediction_error_records,
                dynamic_memory_weight_records,
                slot_weights,
                retained_source_slot_indices,
            ) = self._build_causal_training_memories(
                projected_input=projected_input,
                target=target,
                valid_slot_mask=valid_slot_mask,
                slot_frame_indices=slot_frame_indices,
                pair_start_mask=pair_start_mask,
            )
            projected_input["decision_memory"] = einops.rearrange(
                decision_memory, "b t n d -> (b t) n d"
            )
            model_output = self.denoising_network.finish_parallel_forward(
                projected_input,
                batch_size=batch_size,
                traj_num=traj_num,
            )
        elif projected_input is not None:
            model_output = self.denoising_network.finish_parallel_forward(
                projected_input,
                batch_size=batch_size,
                traj_num=traj_num,
            )
        else:
            model_output = self.denoising_network.parallel_forward(data_dict)

        if self.future_feature_predictor is not None:
            if "future_features" not in target or "future_feature_mask" not in target:
                raise RuntimeError(
                    "Future prediction is enabled but future feature targets were "
                    "not constructed"
                )
            if predicted_future is None:
                decision_memory = model_output["decision_memory"]
                effective_traj_num = decision_memory.shape[1]
                flat_decision_memory = einops.rearrange(
                    decision_memory, "b t n d -> (b t) n d"
                )
                if self.future_feature_predictor.use_action_condition:
                    action_condition = einops.rearrange(
                        target["future_action_condition"],
                        "b t n d -> (b t) n d",
                    )
                else:
                    action_condition = None

                predicted_future = self.future_feature_predictor(
                    current_observation_tokens=(
                        self._pool_future_prediction_context(
                            flat_decision_memory
                        )
                    ),
                    action_chunk=action_condition,
                )
                predicted_future = einops.rearrange(
                    predicted_future,
                    "(b t) n d -> b t n d",
                    b=batch_size,
                    t=effective_traj_num,
                )
            loss["future_prediction"] = (
                self.future_feature_predictor.cosine_prediction_loss(
                    predicted_features=predicted_future,
                    target_features=target["future_features"],
                    target_mask=target["future_feature_mask"],
                    detach_target=True,
                )
            )

        if (
            self.prediction_error_queue_config is not None
            and self.prediction_error_queue_config.record_training_stats
            and slot_weights is not None
            and slot_frame_indices is not None
        ):
            loss["prediction_error_queue_stats"] = (
                self._summarize_prediction_error_records(
                    records=prediction_error_records,
                    dynamic_records=dynamic_memory_weight_records,
                    slot_weights=slot_weights,
                    valid_slot_mask=~normalized_batch["entire_traj_is_padding"],
                )
            )

        if (
            self.prediction_error_queue_config is not None
            and self.prediction_error_queue_config.record_slot_weights
            and slot_weights is not None
            and retained_source_slot_indices is not None
            and slot_frame_indices is not None
        ):
            loss["prediction_error_slot_weight_table"] = (
                build_active_slot_weight_table(
                    slot_weights=slot_weights,
                    retained_source_slot_indices=retained_source_slot_indices,
                    slot_frame_indices=slot_frame_indices,
                )
            )

        if "memory_gate_val" in self.denoising_network.recorded_data_dict:
            memory_gate_val = self.denoising_network.recorded_data_dict[
                "memory_gate_val"
            ]  # (batch_size, traj_num, transformer_layer_num, input_token_num)

            if len(memory_gate_val) > 0:
                memory_gate_val = einops.reduce(
                    memory_gate_val, "l (b t)-> b t", "mean", b=batch_size, t=traj_num
                )  # (batch_size, traj_num)
                memory_gate_val = memory_gate_val * (
                    ~normalized_batch["entire_traj_is_padding"]
                )  # Do not compute loss for padding trajectories
            else:
                memory_gate_val = None
        else:
            memory_gate_val = None

        critical_memory_gate_val = None

        action_loss = F.mse_loss(
            model_output["action"], target["action"], reduction="none"
        )

        action_loss = einops.reduce(
            action_loss, "b t ... -> b t", "mean"
        )  # mean over all dimensions except batch and traj_num # (batch_size, traj_num)
        if self.max_training_traj_num <= 0:
            action_loss = action_loss * (
                ~normalized_batch["entire_traj_is_padding"]
            )  # Do not compute loss for padding trajectories

        # img_feature_loss = None

        if "action_is_error" in normalized_batch and self.max_training_traj_num <= 0:
            traj_error_mask = torch.zeros(action_traj_length, device=self.device)
            traj_error_mask[
                self.action_no_error_range[0] : self.action_no_error_range[1]
            ] = 1
            traj_is_error = (
                normalized_batch["action_is_error"] * traj_error_mask[None, None, :]
            )  # (batch_size, traj_num, traj_length)
            traj_is_error = einops.reduce(
                traj_is_error, "b t ... -> b t", "any"
            )  # (batch_size, traj_num)
            action_loss = action_loss * (~traj_is_error)

            # if img_feature_loss is not None:
            #     img_feature_loss = img_feature_loss * (~traj_is_error)
            if memory_gate_val is not None:
                memory_gate_val = memory_gate_val * (~traj_is_error)

        critical_action_loss = None
        if "action_is_critical" in normalized_batch and self.max_training_traj_num <= 0:
            # Is based on the previous filtered loss (action_is_error and entire_traj_is_padding)
            single_action_is_critical = normalized_batch["action_is_critical"]
            # (batch_size, traj_num, traj_length)
            traj_action_is_critical = torch.any(
                single_action_is_critical, dim=2
            ).squeeze(
                -1
            )  # (batch_size, traj_num)
            critical_action_loss = action_loss * traj_action_is_critical

            if critical_action_loss.sum() > 0:
                loss["critical_action"] = (
                    critical_action_loss.sum() / (critical_action_loss != 0).sum()
                )
            else:
                loss["critical_action"] = critical_action_loss.sum()

            if memory_gate_val is not None:
                critical_memory_gate_val = memory_gate_val * traj_action_is_critical
                valid_mask = traj_action_is_critical * (action_loss != 0)
                if valid_mask.sum() > 0:
                    loss["critical_memory_gate_val"] = (
                        critical_memory_gate_val.sum()
                        / valid_mask.sum()
                    )
                    loss["critical_binary_memory_gate_val"] = (critical_memory_gate_val > 0.5).sum() / valid_mask.sum()
                else:
                    loss["critical_memory_gate_val"] = critical_memory_gate_val.sum()
                    loss["critical_binary_memory_gate_val"] = (critical_memory_gate_val > 0.5).sum()

        if memory_gate_val is not None:
            valid_num = (action_loss != 0).sum() if (action_loss != 0).sum() > 0 else memory_gate_val.sum()
            loss["memory_gate_val"] = memory_gate_val.sum() / valid_num
            loss["binary_memory_gate_val"] = (memory_gate_val > 0.5).sum() / valid_num
            
        if (action_loss != 0).sum() == 0:
            loss["action"] = action_loss.sum() / action_loss.numel()
        else:
            loss["action"] = action_loss.sum() / (action_loss != 0).sum()



        return loss

    def reset(self):
        super().reset()
        self.history_noisy_actions_dict = {}
        self.history_img_features_dict = {}
        self.history_memory_weights_dict = {}
        self.history_memory_slot_ids_dict = {}
        self.next_memory_slot_id_dict = {}
        self.history_frame_indices_dict = {}
        self.next_frame_index_dict = {}
        self.latest_online_memory_query_dict = {}
        self.prediction_error_queues = {}
        self.pending_future_predictions_dict = {}
        self.pending_dynamic_memory_updates_dict = {}
        self.recorded_data_dicts = {}
        self._streaming_training_state = None
        self._streaming_raw_history_state = None
        self._prefix_training_state = None
