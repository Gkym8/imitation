from functools import partial
from typing import Any, Callable, Mapping

import einops
import torch
import torch.nn.functional as F
from torch._dynamo.eval_frame import OptimizedModule

from imitation_learning.common.datatypes import batch_type
from imitation_learning.models.common.memory_gate import MemoryGate
from imitation_learning.models.denoising_networks.memory_transformer import MemoryTransformer
from imitation_learning.models.encoders.multi_token_encoder import MultiTokenEncoder
from imitation_learning.models.future_prediction.future_feature_predictor import FutureFeaturePredictor
from imitation_learning.models.memory.gate_memory import GateMemoryBank, gate_selectivity_penalty
from imitation_learning.policies.base_denoising_policy import BaseDenoisingPolicy
from robot_utils.data_utils import dict_apply
from robot_utils.torch_utils import aggregate_batch, split_batch


class HistoryDenoisingPolicy(BaseDenoisingPolicy):
    """Causal visual memory with observation-conditioned gate eviction."""

    def __init__(
        self,
        skip_memory: bool,
        history_img_feature_encoder: MultiTokenEncoder | None,
        execution_chunk_length: int | None = None,
        future_img_feature_encoder: MultiTokenEncoder | None = None,
        future_prediction_context_pooling: str = "none",
        use_cached_image_features: bool = True,
        future_feature_predictor: (
            FutureFeaturePredictor | Callable[..., FutureFeaturePredictor] | None
        ) = None,
        memory_eviction: Mapping[str, Any] | None = None,
        gate_selectivity_min_std: float = 0.1,
        # add_noise_to_history_img_features: bool,
        memory_gate: MemoryGate | None = None,
        max_training_traj_num: int = -1,
        future_prediction_mode: str | None = None,
        **kwargs,
    ):
        # Checkpoints without a mode retain their explicit legacy pooling and
        # target-encoder settings. New configs select both with one switch.
        if future_prediction_mode is not None:
            if future_prediction_mode not in {"pooled", "tokens"}:
                raise ValueError(
                    "future_prediction_mode must be 'pooled' or 'tokens', "
                    f"got {future_prediction_mode!r}"
                )
            future_prediction_context_pooling = (
                "mean" if future_prediction_mode == "pooled" else "none"
            )

        # Discard retired checkpoint metadata; memory slots contain images only.
        legacy_chunk_length = kwargs.pop("history_action_num_per_chunk", None)
        kwargs.pop("train_history_action_noise_level", None)
        kwargs.pop("eval_history_action_noise_level", None)
        if execution_chunk_length is None:
            execution_chunk_length = legacy_chunk_length

        if history_img_feature_encoder is not None:
            kwargs["denoising_network_partial"] = partial(
                kwargs["denoising_network_partial"],
                history_img_features_dim=history_img_feature_encoder.feature_dim,
                history_img_features_token_num=history_img_feature_encoder.token_num,
            )

        super().__init__(**kwargs)

        if (
            not isinstance(self.denoising_network, MemoryTransformer)
            and not skip_memory
        ):
            raise TypeError(
                "HistoryDenoisingPolicy requires MemoryTransformer"
            )
        self.future_prediction_mode = future_prediction_mode
        if future_prediction_context_pooling not in {"none", "mean"}:
            raise ValueError(
                "future_prediction_context_pooling must be 'none' or 'mean', "
                f"got {future_prediction_context_pooling!r}"
            )
        self.future_prediction_context_pooling = (
            future_prediction_context_pooling
        )
        self.use_cached_image_features = bool(use_cached_image_features)
        if future_img_feature_encoder is None:
            # Keep legacy checkpoints unchanged: the fallback is the already
            # registered history encoder, not a second nn.Module alias with a
            # duplicate state-dict prefix.
            self.__dict__["future_img_feature_encoder"] = (
                history_img_feature_encoder
            )
        else:
            self.future_img_feature_encoder = future_img_feature_encoder

        if (
            future_prediction_mode is not None
            and future_feature_predictor is not None
            and self.future_img_feature_encoder is not None
        ):
            expected_aggregations = (
                {"map", "cls"}
                if future_prediction_mode == "pooled"
                else {"patches"}
            )
            for name, encoder in self.future_img_feature_encoder.encoder_dict.items():
                aggregation = getattr(encoder, "feature_aggregation", None)
                if aggregation not in expected_aggregations:
                    raise ValueError(
                        f"future_prediction_mode={future_prediction_mode!r} "
                        "requires future image feature_aggregation in "
                        f"{sorted(expected_aggregations)}, got "
                        f"{aggregation!r} for {name!r}"
                    )

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
                output_feature_dim=self.future_img_feature_encoder.feature_dim,
                future_token_num=self.future_img_feature_encoder.token_num,
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
            if (
                self.future_feature_predictor.output_feature_dim
                != self.future_img_feature_encoder.feature_dim
                or self.future_feature_predictor.future_token_num
                != self.future_img_feature_encoder.token_num
            ):
                raise ValueError(
                    "Future predictor output shape must match the future image encoder"
                )
            if (
                self.future_feature_predictor.use_action_condition
                and self.future_feature_predictor.action_dim
                != self.action_decoder.latent_dim
            ):
                raise ValueError(
                    "Future predictor action_dim must match action latent_dim"
                )


        self.history_img_feature_encoder = history_img_feature_encoder
        self.memory_gate = memory_gate if isinstance(memory_gate, MemoryGate) else None
        self.skip_memory = skip_memory
        if skip_memory:
            self.enable_skip_memory()
        elif self.denoising_network.memory_retriever.slot_gate is None:
            raise ValueError("Gate memory requires use_conditioned_slot_gate=True")
        if max_training_traj_num > 0:
            raise ValueError("Gate memory requires ordered anchors, not trajectory subsampling")
        self.max_training_traj_num = max_training_traj_num
        eviction = dict(memory_eviction or {})
        if set(eviction).difference({"keep_first", "keep_recent"}):
            raise ValueError("memory_eviction only accepts keep_first and keep_recent")
        self.keep_first_memory = eviction.get("keep_first", True)
        self.keep_recent_memory = eviction.get("keep_recent", 2)
        if not skip_memory:
            self._new_memory_bank()  # Validate capacity and protection together.
        if not 0 <= gate_selectivity_min_std <= 0.5:
            raise ValueError("gate_selectivity_min_std must be in [0, 0.5]")
        self.gate_selectivity_min_std = float(gate_selectivity_min_std)

        if execution_chunk_length is None:
            execution_chunk_length = self.action_decoder.token_num
        if (
            isinstance(execution_chunk_length, bool)
            or not isinstance(execution_chunk_length, int)
            or not 1 <= execution_chunk_length <= self.action_decoder.token_num
        ):
            raise ValueError("execution_chunk_length must be in [1, action_length]")
        self.execution_chunk_length = execution_chunk_length
        self.history_storage_token_num = (
            history_img_feature_encoder.token_num if history_img_feature_encoder else 0
        )
        self.history_storage_feature_dim = (
            history_img_feature_encoder.feature_dim if history_img_feature_encoder else 0
        )
        self.online_memory_banks: dict[int, GateMemoryBank] = {}
        self.next_frame_index_dict: dict[int, int] = {}
        self.latest_online_memory_query_dict: dict[int, dict[str, Any]] = {}
        self.recorded_data_dicts: dict[int, list[dict[str, torch.Tensor]]] = {}
        self._streaming_training_state: list[GateMemoryBank] | None = None
        self.latest_training_slot_ids: tuple[tuple[int, ...], ...] = ()
        self.latest_training_slot_states: tuple[tuple[str, ...], ...] = ()

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

    def _pool_future_prediction_context(
        self, context_tokens: torch.Tensor
    ) -> torch.Tensor:
        if self.future_prediction_context_pooling == "mean":
            return context_tokens.mean(dim=-2, keepdim=True)
        return context_tokens

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
        if "variance_temperature" in normalized_batch:
            normalized_batch.pop(
                "variance_temperature"
            )  # Remove variance_temperature from meta

        batch_size = normalized_batch["episode_idx"].shape[0]


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
                "history_img_features": (batch_size, traj_num, token_num, history_img_features_dim) # History image features
                "history_noisy_future_img_features": (batch_size, traj_num, token_num, feature_dim)
                "memory_gate_val": (batch_size, traj_num)
                "step": (batch_size,)
            target:
                "action": (batch_size, traj_num, data_length, 8)
        """

        batch_size = next(iter(normalized_batch.values())).shape[0]
        traj_num = next(iter(normalized_batch.values())).shape[1]
        data_dict: dict[str, torch.Tensor] = {}

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
            # Adjacent training anchors are execution_chunk_length steps
            # apart, matching the number of actions actually executed online.
            # Conditioning on the full prediction horizon would expose actions
            # that occur after the future-feature target.
            target["future_action_condition"] = trajectory[
                :, :, : self.execution_chunk_length
            ]

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

        if self.history_img_feature_encoder is not None:
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

        # The frozen future encoder selects CLS/MAP or spatial patch targets
        # according to future_prediction_mode, independently of memory storage.
        if self.future_feature_predictor is not None:
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

        return data_dict, target

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


    def _new_memory_bank(self) -> GateMemoryBank:
        return GateMemoryBank(
            self.denoising_network.max_history_len,
            keep_first=self.keep_first_memory,
            keep_recent=self.keep_recent_memory,
            shadow_horizon=self.denoising_network.effective_shadow_horizon,
        )

    def _pack_history(
        self, banks: list[GateMemoryBank], reference: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        network = self.denoising_network
        # Training and online use the same configured capacity. With Shadow
        # disabled, use only the Active suffix of the checkpoint position table.
        capacity = max(bank.storage_capacity for bank in banks)
        rows, masks, active_masks, shadow_masks, shadow_ages = [], [], [], [], []
        frame_indices = []
        for bank in banks:
            padding = capacity - len(bank)
            empty = reference.new_zeros((padding, *reference.shape[1:]))
            retained = (
                torch.stack([slot.latent for slot in bank.slots])
                if len(bank) else reference.new_zeros((0, *reference.shape[1:]))
            )
            rows.append(torch.cat([empty, retained], dim=0))
            masks.append(torch.arange(capacity, device=reference.device) >= padding)
            active = torch.tensor(
                [False] * padding + [slot.state == "active" for slot in bank.slots],
                device=reference.device, dtype=torch.bool,
            )
            active_masks.append(active)
            shadow_masks.append(masks[-1] & ~active)
            shadow_ages.append(torch.tensor(
                [0] * padding + [bank.shadow_age(slot) for slot in bank.slots],
                device=reference.device, dtype=torch.long,
            ))
            frame_indices.append(torch.tensor(
                [0] * padding + [slot.frame_index for slot in bank.slots],
                device=reference.device, dtype=torch.long,
            ))
        history = torch.stack(rows)
        positioned = network.add_history_key_encoding(
            history, network.history_time_embedding[:, -capacity:],
            history_frame_indices=torch.stack(frame_indices),
        )
        result = {
            "history_mask": torch.stack(masks),
            "history_active_mask": torch.stack(active_masks),
            "history_shadow_mask": torch.stack(shadow_masks),
            "history_shadow_age": torch.stack(shadow_ages),
        }
        if network.memory_retrieval_mode == "perception_patches":
            result["history_latents"] = history
            result["history_key_latents"] = positioned
        else:
            result["history_latents"] = positioned
        return result

    def _retrieval_records(self, training: bool = False):
        entries = list(self.denoising_network.record_data_entries)
        if training and self.denoising_network.record_history_attention_epoch_stats:
            for name in ("history_cross_attention_frame_logits", "history_cross_attention_slot_weights"):
                if name not in entries:
                    entries.append(name)
        if training:
            entries.append("slot_gate_training_log_gates")
        return {name: [] for name in entries}

    def _current_slot_latents(self, raw_features, retrieval_inputs):
        network = self.denoising_network
        if network.compact_multiview_slot or network.state_conditioned_history_slot:
            return network.project_state_conditioned_slot_features(
                global_cond=retrieval_inputs["global_cond"],
                local_cond=retrieval_inputs.get("local_cond"),
                global_cond_mask=retrieval_inputs.get("global_cond_mask"),
                local_cond_mask=retrieval_inputs.get("local_cond_mask"),
            )
        return network.history_img_features_projector(raw_features)

    def _current_gate_scores(self, banks):
        gates = self.denoising_network.memory_retriever.latest_slot_gates
        if gates is None:
            raise RuntimeError("Memory retrieval did not produce slot gates")
        return [
            gates[i, gates.shape[1] - len(bank):].mean(dim=-1)
            for i, bank in enumerate(banks)
        ]

    def predict_single_traj(self, normalized_batch: batch_type) -> batch_type:
        """Read, decide, evict by the same read's gates, then write the new slot."""
        if "action" in normalized_batch:
            raise ValueError("Exclude action supervision during online inference")
        self.shared_model_manager.clear_cache()
        network = self.denoising_network
        data_dict, _ = self._encode_input_add_noise(normalized_batch, mode="eval")
        batch_size = data_dict["noisy_action"].shape[0]
        episode_ids = [int(value) for value in normalized_batch["episode_idx"]]
        if len(episode_ids) != batch_size or len(set(episode_ids)) != batch_size:
            raise ValueError("online batch must contain distinct episode IDs")
        banks = [
            self.online_memory_banks.setdefault(episode, self._new_memory_bank())
            for episode in episode_ids
        ]
        raw_features = self._encode_current_history_image_features(normalized_batch, batch_size)
        retrieval_inputs = network.prepare_single_memory_retrieval(data_dict)
        current_latents = self._current_slot_latents(raw_features, retrieval_inputs)
        retrieval_inputs.update(self._pack_history(banks, current_latents))
        if self.memory_gate is not None:
            retrieval_inputs["memory_gate_val"] = self.memory_gate.get_gate_value(normalized_batch)
        records = self._retrieval_records()
        decision_memory = network.retrieve_decision_memory(
            retrieval_inputs, record_data_dict=records or None,
        )
        gate_scores = self._current_gate_scores(banks)
        gates = network.memory_retriever.latest_slot_gates
        effective_gates = network.memory_retriever.latest_effective_slot_gates
        self.latest_online_memory_query_dict = {}
        for i, (episode, bank) in enumerate(zip(episode_ids, banks)):
            values = gates[i, gates.shape[1] - len(bank):].float().cpu().tolist()
            self.latest_online_memory_query_dict[episode] = {
                "query_frame_index": self.next_frame_index_dict.get(episode, 0),
                "query_anchor_index": bank.next_anchor_index,
                "active_slot_count": bank.active_count,
                "shadow_slot_count": bank.shadow_count,
                "slot_ids": tuple(slot.slot_id for slot in bank.slots),
                "slot_frame_indices": tuple(slot.frame_index for slot in bank.slots),
                "slot_states": tuple(slot.state for slot in bank.slots),
                "slot_expire_at": tuple(slot.expire_at for slot in bank.slots),
                "slot_gates": tuple(tuple(heads) for heads in values),
                "slot_effective_gates": tuple(
                    tuple(heads) for heads in effective_gates[i, gates.shape[1] - len(bank):].float().cpu().tolist()
                ),
                "gate_scores": tuple(sum(heads) / len(heads) for heads in values),
            }
            if records:
                self.recorded_data_dicts.setdefault(episode, []).append({
                    key: torch.stack(values, dim=1)[i:i + 1].detach().clone()
                    for key, values in records.items() if values
                })

        data_dict["decision_memory"] = decision_memory
        if records:
            data_dict["_precomputed_memory_record_data"] = records
        for timestep in self.noise_scheduler.get_inference_timesteps():
            data_dict["step"] = torch.full(
                (batch_size,), timestep, device=self.device, dtype=data_dict["step"].dtype
            )
            output = network(data_dict)
            data_dict.pop("_precomputed_memory_record_data", None)
            data_dict["noisy_action"] = self.noise_scheduler.step(
                output["action"], int(timestep), data_dict["noisy_action"]
            )
        valid = ~normalized_batch.get(
            "entire_traj_is_padding",
            torch.zeros(batch_size, dtype=torch.bool, device=self.device),
        ).reshape(batch_size).bool()
        for i, (episode, bank) in enumerate(zip(episode_ids, banks)):
            if not bool(valid[i]):
                continue
            frame = self.next_frame_index_dict.get(episode, 0)
            bank.append(
                current_latents[i].detach(), frame, gate_scores[i],
                raw_features=raw_features[i].detach(),
            )
            self.next_frame_index_dict[episode] = frame + self.execution_chunk_length
        return self.action_decoder(data_dict["noisy_action"])

    def _build_gate_training_memories(
        self, projected, raw_features, valid_mask, frame_indices, supervised, banks
    ):
        network = self.denoising_network
        all_latents = projected["all_history_latents"]
        batch_size, input_length = all_latents.shape[:2]
        offsets = torch.arange(batch_size, device=all_latents.device) * input_length
        memories, penalties, eligible_rows, history_rows = [], [], [], []
        records = self._retrieval_records(training=True)
        current_keys = ("global_cond", "global_cond_mask", "local_cond", "local_cond_mask", "memory_gate_val")
        can_reproject = not (network.compact_multiview_slot or network.state_conditioned_history_slot)
        if can_reproject:
            for bank in banks:
                bank.reproject(network.history_img_features_projector)
        for anchor in range(supervised):
            flat_indices = offsets + anchor
            inputs = {
                name: projected[name][flat_indices]
                for name in current_keys if name in projected
            }
            inputs.update(self._pack_history(banks, all_latents[:, anchor]))
            # Inspect the bank before inserting this observation. This also
            # handles nonzero stream starts, padding, and carried chunk state.
            history_rows.append(inputs["history_mask"].any(dim=-1))
            memory = network.retrieve_decision_memory(inputs, record_data_dict=records)
            memories.append(memory)
            log_gates = records["slot_gate_training_log_gates"].pop()
            penalty, eligible = gate_selectivity_penalty(
                log_gates, inputs["history_active_mask"],
                network.max_history_len, self.gate_selectivity_min_std,
            )
            penalties.append(penalty)
            eligible_rows.append(eligible & valid_mask[:, anchor])
            scores = self._current_gate_scores(banks)
            for i, bank in enumerate(banks):
                if bool(valid_mask[i, anchor]):
                    bank.append(
                        all_latents[i, anchor], int(frame_indices[i, anchor]), scores[i],
                        raw_features=raw_features[i, anchor] if can_reproject else None,
                    )
        records.pop("slot_gate_training_log_gates")
        memories = torch.stack(memories, dim=1)
        penalties = torch.stack(penalties, dim=1)
        eligible = torch.stack(eligible_rows, dim=1)
        # An unfilled bank contributes zero. Averaging over valid anchors keeps
        # the regularizer invariant to the trainer's chunk boundaries.
        gate_loss = (penalties * eligible).sum() / valid_mask[:, :supervised].sum().clamp_min(1)
        if "decision_memory" in records:
            records["decision_memory"] = [
                einops.rearrange(memories.detach(), "b t n d -> (b t) n d")
            ]
        if records:
            projected["_precomputed_memory_record_data"] = records
        self.latest_training_slot_ids = tuple(
            tuple(slot.slot_id for slot in bank.slots) for bank in banks
        )
        self.latest_training_slot_states = tuple(
            tuple(slot.state for slot in bank.slots) for bank in banks
        )
        for bank in banks:
            bank.detach()
        return memories, gate_loss, torch.stack(history_rows, dim=1)

    def compute_streaming_loss(self, normalized_batch: batch_type) -> batch_type:
        if "_streaming_supervised_length" not in normalized_batch:
            raise ValueError("streaming metadata is required")
        return self.compute_loss(normalized_batch)

    def compute_loss(self, normalized_batch: batch_type) -> batch_type:
        if self.skip_memory:
            return super().compute_loss(normalized_batch)
        batch = dict(normalized_batch)
        streaming = "_streaming_supervised_length" in batch
        if streaming:
            supervised = int(batch.pop("_streaming_supervised_length"))
            reset = bool(batch.pop("_streaming_reset"))
            final = bool(batch.pop("_streaming_final"))
        else:
            supervised = batch["entire_traj_is_padding"].shape[1]
            reset, final = True, True
        batch.pop("streaming_sequence", None)
        self.shared_model_manager.clear_cache()
        data_dict, target = self._encode_input_multi_traj(batch)
        valid = ~batch["entire_traj_is_padding"].bool()
        batch_size, input_length = valid.shape
        if not 0 < supervised <= input_length:
            raise ValueError("invalid supervised anchor count")
        if "traj_idx" not in batch:
            raise KeyError("Gate memory requires absolute traj_idx metadata")
        frames = batch["traj_idx"].reshape(batch_size, input_length, -1)
        if frames.shape[-1] != 1:
            raise ValueError("each anchor requires a scalar traj_idx")
        if reset:
            banks = [self._new_memory_bank() for _ in range(batch_size)]
        else:
            banks = self._streaming_training_state
            if banks is None or len(banks) != batch_size:
                raise RuntimeError("missing or misaligned streaming memory banks")
        data_dict["_streaming_training"] = True
        data_dict["history_frame_indices"] = frames[:, :, 0]
        projected = self.denoising_network.prepare_parallel_forward(data_dict)
        decision_memory, gate_loss, history_available = self._build_gate_training_memories(
            projected, data_dict["history_img_features"], valid,
            frames[:, :, 0], supervised, banks,
        )
        projected = self._slice_streaming_projected_input(
            projected, batch_size=batch_size, input_traj_num=input_length,
            supervised_traj_num=supervised,
        )
        projected["decision_memory"] = einops.rearrange(
            decision_memory, "b t n d -> (b t) n d"
        )
        output = self.denoising_network.finish_parallel_forward(
            projected, batch_size=batch_size, traj_num=supervised,
        )
        anchor_valid = valid[:, :supervised]
        action_error = F.mse_loss(
            output["action"], target["action"][:, :supervised], reduction="none"
        ).flatten(start_dim=2).mean(dim=-1)
        loss = {
            "action": (action_error * anchor_valid).sum() / anchor_valid.sum().clamp_min(1),
            "gate_selectivity": gate_loss,
        }
        if self.future_feature_predictor is not None:
            future_mask = target["future_feature_mask"][:, :supervised]
            predict_rows = (
                history_available & anchor_valid & future_mask.any(dim=-1)
            )
            if bool(predict_rows.any()):
                flat_rows = predict_rows.flatten()
                actions = None
                if self.future_feature_predictor.use_action_condition:
                    actions = target["future_action_condition"][:, :supervised][predict_rows]
                # Do not call the predictor for observations without past
                # memory, rather than merely masking their loss afterwards.
                prediction = self.future_feature_predictor(
                    current_observation_tokens=self._pool_future_prediction_context(
                        projected["decision_memory"][flat_rows]
                    ),
                    action_chunk=actions,
                )
                # Preserve each mode's objective, averaging only eligible
                # future targets (not the action-only first observation).
                prediction_loss = (
                    self.future_feature_predictor.mse_prediction_loss
                    if self.future_prediction_mode == "tokens"
                    else self.future_feature_predictor.cosine_prediction_loss
                )
                loss["future_prediction"] = prediction_loss(
                    prediction, target["future_features"][:, :supervised][predict_rows],
                    target_mask=future_mask[predict_rows],
                    detach_target=True,
                )
            else:
                # Action-only chunks still support ordinary backward/DDP.
                # The existing trainer permits unused predictor parameters.
                loss["future_prediction"] = projected["decision_memory"].sum() * 0.0
        if "action_is_critical" in batch:
            critical = batch["action_is_critical"][:, :supervised].any(dim=2).squeeze(-1)
            critical = critical & anchor_valid
            loss["critical_action"] = (action_error * critical).sum() / critical.sum().clamp_min(1)
        if streaming:
            self._streaming_training_state = None if final else banks
        return loss

    def reset(self):
        super().reset()
        self.online_memory_banks = {}
        self.next_frame_index_dict = {}
        self.latest_online_memory_query_dict = {}
        self.recorded_data_dicts = {}
        self._streaming_training_state = None
        self.latest_training_slot_ids = ()
        self.latest_training_slot_states = ()
        if isinstance(self.denoising_network, (MemoryTransformer, OptimizedModule)):
            self.denoising_network.memory_retriever.latest_slot_gates = None
            self.denoising_network.memory_retriever.latest_effective_slot_gates = None
