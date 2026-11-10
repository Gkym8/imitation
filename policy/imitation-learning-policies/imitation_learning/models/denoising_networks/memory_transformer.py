import copy
import math
import einops
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, RmsNorm

from imitation_learning.models.common.modules import Projector
from imitation_learning.models.common.patch_merger import SpatialPatchMerger
from imitation_learning.models.common.spatial_merge import SpatialMergeSize, spatial_merge_shape
from imitation_learning.models.common.state_query_merger import StateQueryPatchMerger
from imitation_learning.models.common.learned_query_merger import LearnedQueryPatchMerger
from imitation_learning.models.common.slot_gate import ObservationConditionedSlotGate
from imitation_learning.models.common.pos_embeddings import (
    get_1d_sincos_pos_embed_from_grid,
)
from imitation_learning.models.denoising_networks.conditional_transformer import (
    ConditionalTransformer, ConditionalTransformerBlock)
from imitation_learning.models.denoising_networks.modules import (
    CrossAttention, HistoryCrossAttention)


class BinaryGatingSTEv1(torch.autograd.Function):
    """
    Straight-through estimator for binary gating.
    Backward assumes y = x * gate
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, gate)
        # Return 1.0 if x > 0.5, else 0.0
        return x * (gate > 0.5).float()


    @staticmethod
    def backward(ctx, grad_outputs):
        # Identity gradient: let the MLP think it's continuous
        x, gate = ctx.saved_tensors
        grad_x = grad_outputs * gate
        grad_gate = grad_outputs * x
        return grad_x, grad_gate


class BinaryGatingSTEv2(torch.autograd.Function):
    """
    Straight-through estimator for binary gating.
    Backward uses binary mask to compute gradient of x
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        mask = (gate > 0.5).float()
        ctx.save_for_backward(x, mask)
        # Return 1.0 if x > 0.5, else 0.0
        return x * mask


    @staticmethod
    def backward(ctx, grad_outputs):
        # Identity gradient: let the MLP think it's continuous
        x, mask = ctx.saved_tensors
        grad_x = grad_outputs * mask
        grad_gate = grad_outputs * x
        return grad_x, grad_gate

class BinaryGatingSTEv3(torch.autograd.Function):
    """
    Straight-through estimator for binary gating.
    Backward uses binary mask
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        mask = (gate > 0.5).float()
        ctx.save_for_backward(x, mask)
        return x * mask


    @staticmethod
    def backward(ctx, grad_outputs):
        # Identity gradient: let the MLP think it's continuous
        x, mask = ctx.saved_tensors
        grad_x = grad_outputs * mask
        grad_gate = grad_outputs
        return grad_x, grad_gate


class TokenGateFusion(nn.Module):
    """Learn a feature-wise blend of current and history-aware tokens."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.proj.bias, mean=0.0, std=1e-3)

    def forward(
        self, current_tokens: torch.Tensor, retrieved_tokens: torch.Tensor
    ) -> torch.Tensor:
        gate = torch.sigmoid(
            self.proj(torch.cat([current_tokens, retrieved_tokens], dim=-1))
        )
        return gate * current_tokens + (1.0 - gate) * retrieved_tokens


class PerceptionHistoryRetrievalBlock(nn.Module):
    """Use every current image patch to retrieve all retained history patches."""

    def __init__(
        self,
        hidden_dim: int,
        head_num: int,
        history_attention_type: str,
        ssmax_scaling_param: float | None,
    ):
        super().__init__()
        self.history_norm = RmsNorm(hidden_dim, eps=1e-6)
        self.history_cross_attn = HistoryCrossAttention(
            dim=hidden_dim,
            head_num=head_num,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
            attention_type=history_attention_type,
            ssmax_scaling_param=ssmax_scaling_param,
        )
        self.attn_output_norm = RmsNorm(hidden_dim, eps=1e-6)
        self.ff = Mlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim * 4,
            act_layer=nn.GELU,
        )
        self.ff_output_norm = RmsNorm(hidden_dim, eps=1e-6)

    def forward(
        self,
        query: torch.Tensor,
        history_values: torch.Tensor,
        history_keys: torch.Tensor | None,
        history_mask: torch.Tensor | None,
        record_data_dict: dict[str, list[torch.Tensor]] | None = None,
        history_log_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        retrieved = self.history_cross_attn(
            x=self.history_norm(query),
            history_latents=history_values,
            history_key_latents=history_keys,
            history_mask=history_mask,
            record_data_dict=record_data_dict,
            history_log_gate=history_log_gate,
        )
        query = self.attn_output_norm(query + retrieved)
        return self.ff_output_norm(query + self.ff(query))


class ObservationDrivenMemoryRetriever(nn.Module):
    """Build a compact decision memory from the current observation and history.

    The learned query tokens first read the current image/proprioception tokens and
    then use the resulting observation representation to retrieve the raw memory
    slots.  The residual path guarantees that, when no history is available,
    ``decision_memory`` still contains the current observation.
    """

    def __init__(
        self,
        hidden_dim: int,
        head_num: int,
        decision_memory_token_num: int,
        history_attention_type: str,
        ssmax_scaling_param: float | None,
        binary_gating: bool,
        straight_through: str,
        retrieval_mode: str = "learned_queries",
        perception_retrieval_layers: int = 2,
        compact_multiview_slot: bool = False,
        use_conditioned_slot_gate: bool = False,
        slot_gate_hidden_dim: int = 128,
        slot_gate_bias_init: float = 0.0,
        shadow_gate_scale: float = 0.01,
        state_as_dit_token: bool = False,
    ):
        super().__init__()
        if retrieval_mode not in {"learned_queries", "perception_patches"}:
            raise ValueError(
                "retrieval_mode must be 'learned_queries' or "
                f"'perception_patches', got {retrieval_mode!r}"
            )
        if perception_retrieval_layers <= 0:
            raise ValueError("perception_retrieval_layers must be positive")
        if decision_memory_token_num <= 1:
            raise ValueError(
                "decision_memory_token_num must be greater than 1 so DiT blocks "
                "can selectively attend to decision memory"
            )

        self.hidden_dim = hidden_dim
        self.decision_memory_token_num = decision_memory_token_num
        self.binary_gating = binary_gating
        self.straight_through = straight_through
        self.retrieval_mode = retrieval_mode
        self.compact_multiview_slot = compact_multiview_slot
        self.state_as_dit_token = state_as_dit_token
        if not math.isfinite(shadow_gate_scale) or not 0 < shadow_gate_scale <= 1:
            raise ValueError("shadow_gate_scale must be finite and in (0, 1]")
        self.shadow_gate_scale = float(shadow_gate_scale)
        if use_conditioned_slot_gate and ssmax_scaling_param is not None:
            raise ValueError("Slot gates require standard Softmax attention")
        self.slot_gate = (
            ObservationConditionedSlotGate(
                feature_dim=hidden_dim,
                head_num=head_num,
                hidden_dim=slot_gate_hidden_dim,
                bias_init=slot_gate_bias_init,
                use_state_condition=state_as_dit_token,
            )
            if use_conditioned_slot_gate
            else None
        )
        self.latest_slot_gates: torch.Tensor | None = None
        self.latest_effective_slot_gates: torch.Tensor | None = None

        self.query_tokens = nn.Parameter(
            torch.empty(1, decision_memory_token_num, hidden_dim)
        )
        self.observation_norm = RmsNorm(hidden_dim, eps=1e-6)
        self.observation_cross_attn = CrossAttention(
            dim=hidden_dim,
            head_num=head_num,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
        )
        self.query_self_norm = RmsNorm(hidden_dim, eps=1e-6)
        self.query_self_attn = Attention(
            dim=hidden_dim,
            num_heads=head_num,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
        )
        self.query_ff_norm = RmsNorm(hidden_dim, eps=1e-6)
        self.query_ff = Mlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim * 4,
            act_layer=nn.GELU,
        )

        self.history_norm = RmsNorm(hidden_dim, eps=1e-6)
        self.history_cross_attn = HistoryCrossAttention(
            dim=hidden_dim,
            head_num=head_num,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
            attention_type=history_attention_type,
            ssmax_scaling_param=ssmax_scaling_param,
        )
        self.output_ff_norm = RmsNorm(hidden_dim, eps=1e-6)
        self.output_ff = Mlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim * 4,
            act_layer=nn.GELU,
        )

        # Keep legacy learned-query checkpoints loadable without missing
        # perception-module parameters.  These modules exist only when the new
        # retrieval mode is explicitly selected by configuration.
        if retrieval_mode == "perception_patches":
            if not state_as_dit_token:
                self.state_condition_norm = RmsNorm(hidden_dim, eps=1e-6)
                self.state_condition_attn = CrossAttention(
                    dim=hidden_dim,
                    head_num=head_num,
                    qkv_bias=True,
                    qk_norm=True,
                    norm_layer=RmsNorm,
                )
                self.state_condition_ff_norm = RmsNorm(hidden_dim, eps=1e-6)
                self.state_condition_ff = Mlp(
                    in_features=hidden_dim,
                    hidden_features=hidden_dim * 4,
                    act_layer=nn.GELU,
                )
            self.perception_retrieval_blocks = nn.ModuleList(
                [
                    PerceptionHistoryRetrievalBlock(
                        hidden_dim=hidden_dim,
                        head_num=head_num,
                        history_attention_type=history_attention_type,
                        ssmax_scaling_param=ssmax_scaling_param,
                    )
                    for _ in range(perception_retrieval_layers)
                ]
            )
            self.perception_gate = TokenGateFusion(hidden_dim)
            if compact_multiview_slot:
                self.compact_history_query_norm = RmsNorm(
                    hidden_dim, eps=1e-6
                )
                self.compact_history_query_mlp = Mlp(
                    in_features=hidden_dim,
                    hidden_features=hidden_dim * 4,
                    out_features=hidden_dim,
                    act_layer=nn.GELU,
                )

            # These modules only implement the legacy learned-query route.
            # Keeping them in the state layout is convenient, but they must
            # not be trainable or DDP will report unused-gradient parameters.
            self.query_tokens.requires_grad_(False)
            for legacy_module in (
                self.observation_norm,
                self.observation_cross_attn,
                self.query_self_norm,
                self.query_self_attn,
                self.query_ff_norm,
                self.query_ff,
                self.history_norm,
                self.history_cross_attn,
                self.output_ff_norm,
                self.output_ff,
            ):
                legacy_module.requires_grad_(False)

        nn.init.normal_(self.query_tokens, std=0.02)

    def condition_current_tokens(
        self,
        global_cond: torch.Tensor,
        local_cond: torch.Tensor | None = None,
        local_cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return image tokens, with legacy state conditioning when selected."""

        if self.state_as_dit_token:
            return global_cond
        current_tokens = global_cond
        if local_cond is not None:
            if (
                local_cond.ndim != 3
                or local_cond.shape[0] != global_cond.shape[0]
                or local_cond.shape[-1] != self.hidden_dim
            ):
                raise ValueError(
                    "local_cond must have shape [batch, token, hidden_dim], "
                    f"got {tuple(local_cond.shape)}"
                )
            current_tokens = current_tokens + self.state_condition_attn(
                self.state_condition_norm(current_tokens),
                local_cond,
                local_cond_mask,
            )
            current_tokens = current_tokens + self.state_condition_ff(
                self.state_condition_ff_norm(current_tokens)
            )
        elif local_cond_mask is not None:
            raise ValueError("local_cond_mask was provided without local_cond")
        return current_tokens

    def _apply_memory_gate(
        self, retrieved_history: torch.Tensor, memory_gate_val: torch.Tensor
    ) -> torch.Tensor:
        gate = memory_gate_val[:, None, None]
        if not self.binary_gating:
            return retrieved_history * gate
        if self.straight_through == "v1":
            return BinaryGatingSTEv1.apply(retrieved_history, gate)
        if self.straight_through == "v2":
            return BinaryGatingSTEv2.apply(retrieved_history, gate)
        if self.straight_through == "v3":
            return BinaryGatingSTEv3.apply(retrieved_history, gate)
        return retrieved_history * (gate > 0.5).to(retrieved_history.dtype)

    def _compute_slot_log_gate(
        self,
        current_tokens: torch.Tensor,
        history_latents: torch.Tensor,
        history_mask: torch.Tensor | None,
        current_mask: torch.Tensor | None,
        record_data_dict: dict[str, list[torch.Tensor]] | None,
        history_shadow_mask: torch.Tensor | None = None,
        history_shadow_steps: torch.Tensor | None = None,
        state_tokens: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.slot_gate is None:
            return None
        log_gate = self.slot_gate(
            history_tokens=history_latents,
            current_tokens=current_tokens,
            history_mask=history_mask,
            current_mask=current_mask,
            state_tokens=state_tokens,
            state_mask=state_mask,
        )
        if record_data_dict is not None and "slot_gate_training_log_gates" in record_data_dict:
            record_data_dict["slot_gate_training_log_gates"].append(log_gate)
        gates = log_gate.detach().exp()
        if history_mask is not None:
            gates = gates.masked_fill(~history_mask[:, :, None].bool(), 0)
        self.latest_slot_gates = gates
        if record_data_dict is not None and "history_slot_gates" in record_data_dict:
            record_data_dict["history_slot_gates"].append(gates.clone())
        if history_shadow_mask is not None:
            if history_shadow_mask.shape != log_gate.shape[:2]:
                raise ValueError("history_shadow_mask must have shape [batch, slot]")
            shadow_mask = history_shadow_mask.to(device=log_gate.device, dtype=torch.bool)
            if history_mask is not None and (shadow_mask & ~history_mask.bool()).any():
                raise ValueError("history_shadow_mask may only mark valid history slots")
            if history_shadow_steps is None:
                if shadow_mask.any():
                    raise ValueError("history_shadow_steps is required for Shadow slots")
                history_shadow_steps = torch.zeros_like(shadow_mask, dtype=torch.long)
            if history_shadow_steps.shape != log_gate.shape[:2]:
                raise ValueError("history_shadow_steps must have shape [batch, slot]")
            if history_shadow_steps.is_floating_point() or history_shadow_steps.dtype == torch.bool:
                raise ValueError("history_shadow_steps must contain integer read counts")
            steps = history_shadow_steps.to(device=log_gate.device)
            if (steps < 0).any() or not torch.equal(steps > 0, shadow_mask):
                raise ValueError("history_shadow_steps must be positive only for Shadow slots")
            # Recompute from the raw gate: g_eff = g_raw * rate**k.
            # Log space avoids repeatedly multiplying already-decayed gates.
            log_gate = log_gate + steps[:, :, None].to(log_gate.dtype) * math.log(self.shadow_gate_scale)
        elif history_shadow_steps is not None:
            raise ValueError("history_shadow_steps requires history_shadow_mask")
        effective_gates = log_gate.detach().exp()
        if history_mask is not None:
            effective_gates = effective_gates.masked_fill(~history_mask[:, :, None].bool(), 0)
        self.latest_effective_slot_gates = effective_gates
        if record_data_dict is not None and "history_effective_slot_gates" in record_data_dict:
            record_data_dict["history_effective_slot_gates"].append(effective_gates.clone())
        return log_gate

    def forward(
        self,
        global_cond: torch.Tensor,
        global_cond_mask: torch.Tensor | None = None,
        local_cond: torch.Tensor | None = None,
        local_cond_mask: torch.Tensor | None = None,
        history_latents: torch.Tensor | None = None,
        history_key_latents: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        memory_gate_val: torch.Tensor | None = None,
        skip_history: bool = False,
        record_data_dict: dict[str, list[torch.Tensor]] | None = None,
        history_shadow_mask: torch.Tensor | None = None,
        history_shadow_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.latest_slot_gates = None
        self.latest_effective_slot_gates = None
        if (history_shadow_mask is not None or history_shadow_steps is not None) and self.slot_gate is None:
            raise ValueError("Shadow memory requires an observation-conditioned slot gate")
        if global_cond.ndim != 3 or global_cond.shape[-1] != self.hidden_dim:
            raise ValueError(
                "global_cond must have shape [batch, token, hidden_dim], got "
                f"{tuple(global_cond.shape)}"
            )

        if self.retrieval_mode == "perception_patches":
            current_tokens = self.condition_current_tokens(
                global_cond, local_cond, local_cond_mask
            )

            retrieved_tokens = current_tokens
            if (
                history_latents is not None
                and history_latents.shape[1] > 0
                and not skip_history
            ):
                if history_mask is None:
                    has_valid_history = torch.ones(
                        global_cond.shape[0],
                        dtype=torch.bool,
                        device=global_cond.device,
                    )
                else:
                    has_valid_history = history_mask.to(dtype=torch.bool).any(
                        dim=-1
                    )
                # Score against the pure current visual/state representation,
                # before any history is retrieved. Reuse the same per-head
                # prior in every retrieval layer of this decision.
                history_log_gate = self._compute_slot_log_gate(
                    current_tokens, history_latents, history_mask,
                    global_cond_mask, record_data_dict,
                    history_shadow_mask, history_shadow_steps,
                    state_tokens=local_cond if self.state_as_dit_token else None,
                    state_mask=local_cond_mask if self.state_as_dit_token else None,
                )
                if self.compact_multiview_slot:
                    # S_t is kept pure for storage and future supervision.  A
                    # separate MLP produces the content query used only for
                    # reading history.
                    retrieved_tokens = current_tokens + (
                        self.compact_history_query_mlp(
                            self.compact_history_query_norm(current_tokens)
                        )
                    )
                last_idx = len(self.perception_retrieval_blocks) - 1
                for block_idx, block in enumerate(
                    self.perception_retrieval_blocks
                ):
                    # Keep diagnostics compatible with the original one-read
                    # layout by recording only the final retrieval layer.
                    layer_record_data = (
                        record_data_dict if block_idx == last_idx else None
                    )
                    retrieved_tokens = block(
                        query=retrieved_tokens,
                        history_values=history_latents,
                        history_keys=history_key_latents,
                        history_mask=history_mask,
                        record_data_dict=layer_record_data,
                        history_log_gate=history_log_gate,
                    )
                # Padded samples with no retained slot must reduce exactly to
                # their current (state-conditioned) perception tokens; the
                # retrieval block's FFN must not alter those samples.
                retrieved_tokens = torch.where(
                    has_valid_history[:, None, None],
                    retrieved_tokens,
                    current_tokens,
                )
                if memory_gate_val is not None:
                    history_delta = retrieved_tokens - current_tokens
                    history_delta = self._apply_memory_gate(
                        history_delta, memory_gate_val
                    )
                    retrieved_tokens = current_tokens + history_delta

            decision_memory = self.perception_gate(
                current_tokens, retrieved_tokens
            )
            if (
                record_data_dict is not None
                and "decision_memory" in record_data_dict
            ):
                record_data_dict["decision_memory"].append(
                    decision_memory.detach().clone()
                )
            return decision_memory

        observation_groups = [global_cond]
        mask_groups: list[torch.Tensor] = []
        any_mask = global_cond_mask is not None or (
            local_cond_mask is not None and not self.state_as_dit_token
        )
        if any_mask:
            if global_cond_mask is None:
                global_cond_mask = torch.ones(
                    global_cond.shape[:2], dtype=torch.bool, device=global_cond.device
                )
            mask_groups.append(global_cond_mask.to(dtype=torch.bool))

        if local_cond is not None and not self.state_as_dit_token:
            if (
                local_cond.ndim != 3
                or local_cond.shape[0] != global_cond.shape[0]
                or local_cond.shape[-1] != self.hidden_dim
            ):
                raise ValueError(
                    "local_cond must have shape [batch, token, hidden_dim], got "
                    f"{tuple(local_cond.shape)}"
                )
            observation_groups.append(local_cond)
            if any_mask:
                if local_cond_mask is None:
                    local_cond_mask = torch.ones(
                        local_cond.shape[:2],
                        dtype=torch.bool,
                        device=local_cond.device,
                    )
                mask_groups.append(local_cond_mask.to(dtype=torch.bool))
        elif local_cond is None and local_cond_mask is not None:
            raise ValueError("local_cond_mask was provided without local_cond")

        observation_tokens = torch.cat(observation_groups, dim=1)
        observation_mask = torch.cat(mask_groups, dim=1) if any_mask else None

        query = self.query_tokens.expand(global_cond.shape[0], -1, -1)
        query = query + self.observation_cross_attn(
            self.observation_norm(query), observation_tokens, observation_mask
        )
        query = query + self.query_self_attn(self.query_self_norm(query))
        query = query + self.query_ff(self.query_ff_norm(query))

        if (
            history_latents is not None
            and history_latents.shape[1] > 0
            and not skip_history
        ):
            history_log_gate = self._compute_slot_log_gate(
                query, history_latents, history_mask, None, record_data_dict,
                history_shadow_mask, history_shadow_steps,
                state_tokens=local_cond if self.state_as_dit_token else None,
                state_mask=local_cond_mask if self.state_as_dit_token else None,
            )
            retrieved_history = self.history_cross_attn(
                x=self.history_norm(query),
                history_latents=history_latents,
                history_mask=history_mask,
                record_data_dict=record_data_dict,
                history_log_gate=history_log_gate,
            )
            if memory_gate_val is not None:
                retrieved_history = self._apply_memory_gate(
                    retrieved_history, memory_gate_val
                )
            query = query + retrieved_history

        decision_memory = query + self.output_ff(self.output_ff_norm(query))
        if record_data_dict is not None and "decision_memory" in record_data_dict:
            record_data_dict["decision_memory"].append(decision_memory.detach().clone())
        return decision_memory


class MemoryTransformerBlock(ConditionalTransformerBlock):
    def __init__(
        self,
        hidden_dim: int,
        head_num: int,
        input_token_num: int,
        add_additional_self_attn: bool,
    ):
        super().__init__(hidden_dim, head_num)
        self.add_additional_self_attn:bool = add_additional_self_attn
        self.decision_memory_norm: nn.Module = RmsNorm(hidden_dim, eps=1e-6)
        self.decision_memory_cross_attn: CrossAttention = CrossAttention(
            dim=hidden_dim,
            head_num=head_num,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
        )
        if self.add_additional_self_attn:
            self.history_norm3: nn.Module = RmsNorm(hidden_dim, eps=1e-6)
            self.history_self_attn: Attention = Attention(
                dim=hidden_dim,
                num_heads=head_num,
                qkv_bias=True,
                qk_norm=True,
                norm_layer=RmsNorm,
            )
        self.input_token_num: int = input_token_num

    def forward(
        self,
        x: torch.Tensor,
        global_cond: torch.Tensor,
        decision_memory: torch.Tensor,
        global_cond_mask: torch.Tensor | None = None,
        decision_memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (batch_size, token_num, hidden_dim)
        global_cond: (batch_size, current_observation_token_num, hidden_dim)
        decision_memory: (batch_size, decision_memory_token_num, hidden_dim)
        """
        assert (
            x.shape[1] == self.input_token_num
        ), f"x.shape[1] ({x.shape[1]}) must be equal to input_token_num ({self.input_token_num})"
        attn: torch.Tensor = self.attn(self.norm1(x))
        x = x + attn
        # Restore the original per-layer current-image conditioning path.  The
        # positional encoding is already present in global_cond, so it must not
        # be added again inside the block.
        current_observation_attn: torch.Tensor = self.cross_attn(
            self.norm2(x), global_cond, global_cond_mask
        )
        x = x + current_observation_attn
        decision_memory_attn: torch.Tensor = self.decision_memory_cross_attn(
            self.decision_memory_norm(x),
            decision_memory,
            decision_memory_mask,
        )
        x = x + decision_memory_attn
        if self.add_additional_self_attn:
            x = x + self.history_self_attn(self.history_norm3(x))

        x = x + self.ff(self.norm3(x))

        return x


class MemoryTransformer(ConditionalTransformer):
    def __init__(
        self,
        max_history_len: int,
        freeze_non_history_modules: bool,
        history_attention_type: str,
        record_data_entries: list[str],  # For debugging
        ssmax_scaling_param: float | None,
        decision_memory_token_num: int,
        skip_history_attn: bool,  # For ablation study
        add_memory_gate_token: bool = False, # For compatibility with previous checkpoints
        binary_gating: bool = True, # For compatibility with previous checkpoints
        straight_through: str = "", # "v1", "v2", "v3", or ""
        add_additional_self_attn: bool = True,
        history_img_features_dim: int = 0,
        history_img_features_token_num: int = 0,
        record_history_attention_epoch_stats: bool = False,
        memory_retrieval_mode: str = "learned_queries",
        perception_retrieval_layers: int = 2,
        compact_multiview_slot: bool = False,
        state_conditioned_history_slot: bool = False,
        history_view_num: int = 0,
        shadow_horizon: int = 0,
        shadow_gate_scale: float = 0.01,
        use_conditioned_slot_gate: bool = False,
        slot_gate_hidden_dim: int = 128,
        slot_gate_bias_init: float = 0.0,
        patch_merge: bool = False,
        patch_merge_size: SpatialMergeSize = 1,
        patch_merge_patch_token_num: int = 0,
        state_as_dit_token: bool = False,
        state_query_merge: bool = False,
        learned_query_merge: bool = False,
        **kwargs,
    ):
        # Old image-only checkpoints may still contain these retired settings.
        kwargs.pop("include_action_history", None)
        kwargs.pop("history_action_num_per_chunk", None)
        kwargs["name"] = "memory_transformer"

        if not isinstance(state_as_dit_token, bool):
            raise ValueError("state_as_dit_token must be boolean")
        if state_as_dit_token and (compact_multiview_slot or state_conditioned_history_slot):
            raise ValueError("state_as_dit_token requires plain image history slots")

        if not isinstance(patch_merge, bool):
            raise ValueError("patch_merge must be boolean")
        if not isinstance(state_query_merge, bool):
            raise ValueError("state_query_merge must be boolean")
        if patch_merge and state_query_merge:
            raise ValueError("patch_merge and state_query_merge are mutually exclusive")
        if not isinstance(learned_query_merge, bool):
            raise ValueError("learned_query_merge must be boolean")
        if learned_query_merge and (patch_merge or state_query_merge):
            raise ValueError("learned_query_merge, patch_merge and state_query_merge are mutually exclusive")
        if state_query_merge and kwargs.get("local_cond_token_num", 0) <= 0:
            raise ValueError("state_query_merge requires robot state tokens")
        if patch_merge or state_query_merge or learned_query_merge:
            if compact_multiview_slot or state_conditioned_history_slot:
                raise ValueError("patch_merge requires plain perception patch slots")
            merge_height, merge_width = spatial_merge_shape(patch_merge_size, "patch_merge_size")
            if patch_merge_patch_token_num <= 0:
                raise ValueError("patch_merge requires positive merge size and DINO patch count")
            if kwargs["global_cond_dim"] != history_img_features_dim:
                raise ValueError("patch_merge requires matching current/history feature widths")
            factor = merge_height * merge_width
            for token_num in (kwargs["global_cond_token_num"], history_img_features_token_num):
                if token_num <= 0 or token_num % patch_merge_patch_token_num:
                    raise ValueError("patch_merge requires complete current/history DINO patch grids")
                if token_num % factor:
                    raise ValueError("patch_merge_size must divide the DINO patch grid")
            kwargs["global_cond_token_num"] //= factor
            history_img_features_token_num //= factor

        if compact_multiview_slot:
            kwargs["global_cond_pos_emb_type"] = "none"

        super().__init__(**kwargs)

        self.patch_merge = patch_merge
        self.state_query_merge = state_query_merge
        self.learned_query_merge = learned_query_merge
        # Missing in older checkpoints: retain their state-to-image route.
        self.state_as_dit_token = state_as_dit_token
        self.binary_gating: bool = binary_gating
        self.skip_history_attn: bool = skip_history_attn
        self.memory_retrieval_mode = memory_retrieval_mode
        self.compact_multiview_slot = compact_multiview_slot
        self.state_conditioned_history_slot = (
            state_conditioned_history_slot
        )
        self.history_view_num = history_view_num
        if isinstance(shadow_horizon, bool) or not isinstance(shadow_horizon, int) or shadow_horizon < 0:
            raise ValueError("shadow_horizon must be a non-negative integer")
        if shadow_horizon and not use_conditioned_slot_gate:
            raise ValueError("Shadow memory requires use_conditioned_slot_gate=True")
        self.shadow_horizon = shadow_horizon
        self.max_history_storage_len = max_history_len + shadow_horizon
        if use_conditioned_slot_gate and (max_history_len <= 0 or skip_history_attn):
            raise ValueError("Slot gates require an enabled, non-empty history bank")
        if compact_multiview_slot:
            if memory_retrieval_mode != "perception_patches":
                raise ValueError(
                    "compact_multiview_slot requires perception_patches retrieval"
                )
            if history_view_num <= 1:
                raise ValueError(
                    "compact_multiview_slot requires at least two views"
                )
            if self.global_cond_token_num != history_view_num:
                raise ValueError(
                    "compact multi-view current tokens must equal history_view_num "
                    f"({self.global_cond_token_num} != {history_view_num})"
                )
        if state_conditioned_history_slot:
            if memory_retrieval_mode != "perception_patches":
                raise ValueError(
                    "state-conditioned history slots require "
                    "perception_patches retrieval"
                )
            if self.global_cond_token_num != history_img_features_token_num:
                raise ValueError(
                    "state-conditioned history slots require current and "
                    "history image token counts to match "
                    f"({self.global_cond_token_num} != "
                    f"{history_img_features_token_num})"
                )
        self.decision_memory_token_num: int = (
            self.global_cond_token_num
            if memory_retrieval_mode == "perception_patches"
            else decision_memory_token_num
        )

        self.memory_retriever = ObservationDrivenMemoryRetriever(
            hidden_dim=self.hidden_dim,
            head_num=self.head_num,
            decision_memory_token_num=decision_memory_token_num,
            history_attention_type=history_attention_type,
            ssmax_scaling_param=ssmax_scaling_param,
            binary_gating=binary_gating,
            straight_through=straight_through,
            retrieval_mode=memory_retrieval_mode,
            perception_retrieval_layers=perception_retrieval_layers,
            compact_multiview_slot=compact_multiview_slot,
            use_conditioned_slot_gate=use_conditioned_slot_gate,
            slot_gate_hidden_dim=slot_gate_hidden_dim,
            slot_gate_bias_init=slot_gate_bias_init,
            shadow_gate_scale=shadow_gate_scale,
            state_as_dit_token=state_as_dit_token,
        )

        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                MemoryTransformerBlock(
                    hidden_dim=self.hidden_dim,
                    head_num=self.head_num,
                    input_token_num=1
                    + self.action_token_num
                    + (self.local_cond_token_num if state_as_dit_token else 0)
                    + (1 if add_memory_gate_token else 0),
                    add_additional_self_attn=add_additional_self_attn,
                )
                for _ in range(self.layer_num)
            ]
        )

        self.max_history_len: int = max_history_len

        self.record_data_entries: list[str] = record_data_entries
        self.recorded_data_dict: dict[str, list[torch.Tensor]] = {}
        self.record_history_attention_epoch_stats: bool = (
            record_history_attention_epoch_stats
        )
        # Plain CPU tensors on purpose: these are runtime diagnostics and must
        # not become part of model/workspace checkpoints.
        self._history_attention_epoch_stats: dict[
            str, dict[str, torch.Tensor]
        ] = {}

        self.history_time_embedding: nn.Parameter = nn.Parameter(
            torch.zeros(
                1, self.max_history_storage_len, self.hidden_dim
            )  # Active capacity + Shadow horizon
        )

        self.initialize_memory_weights()

        if freeze_non_history_modules:
            for name, param in self.named_parameters():
                if (
                    "history" not in name
                    and "memory_retriever" not in name
                    and "decision_memory" not in name
                ):
                    param.requires_grad = False

        self.history_img_features_dim: int = history_img_features_dim
        self.history_img_features_token_num: int = history_img_features_token_num
        print(
            f"history_img_features_dim: {self.history_img_features_dim}, history_img_features_token_num: {self.history_img_features_token_num}"
        )
        self.history_img_features_projector: nn.Module | None = None
        if (
            self.history_img_features_dim > 0
            and self.history_img_features_token_num > 0
        ):
            if learned_query_merge:
                self.global_cond_projector = LearnedQueryPatchMerger(
                    self.global_cond_dim, self.hidden_dim,
                    patch_merge_patch_token_num, patch_merge_size, self.head_num,
                )
            elif state_query_merge:
                self.global_cond_projector = StateQueryPatchMerger(
                    self.global_cond_dim, self.hidden_dim,
                    patch_merge_patch_token_num, patch_merge_size, self.head_num,
                )
            elif patch_merge:
                # Register one merger under the current-image projector. All
                # history projection calls reuse it without a second alias in
                # the state dict or a second set of trainable parameters.
                self.global_cond_projector = SpatialPatchMerger(
                    self.global_cond_dim, self.hidden_dim,
                    patch_merge_patch_token_num, patch_merge_size,
                )
            elif compact_multiview_slot or state_conditioned_history_slot:
                if (
                    compact_multiview_slot
                    and self.history_img_features_token_num != history_view_num
                ):
                    raise ValueError(
                        "compact history image tokens must equal "
                        "history_view_num"
                    )
                # The policy stores S_t after global projection and
                # proprioceptive modulation, already in hidden_dim space.
                self.history_img_features_dim = self.hidden_dim
                self.history_img_features_projector = nn.Identity()
            else:
                self.history_img_features_projector = Projector(
                    self.projector_type,
                    self.history_img_features_dim,
                    self.hidden_dim,
                )

        if compact_multiview_slot:
            self.history_view_embedding: nn.Parameter | None = nn.Parameter(
                torch.empty(1, history_view_num, self.hidden_dim)
            )
            nn.init.normal_(self.history_view_embedding, std=0.02)
        else:
            self.register_parameter("history_view_embedding", None)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"MemoryTransformer trainable parameters: {trainable_params}, total parameters: {total_params}"
        )
        if max_history_len > 0 and (
            self.history_img_features_dim <= 0 or self.history_img_features_token_num <= 0
        ):
            raise ValueError("memory slots require image features and image tokens")

        if add_memory_gate_token:
            assert binary_gating, "Memory gate token is only supported when binary gating is enabled"
            self.memory_gate_tokens: nn.Parameter | None = nn.Parameter(
                torch.randn(2, self.hidden_dim)
            ) # Two tokens: [0] when memory gate == 0, [1] when memory gate == 1
        else:
            self.memory_gate_tokens = None

    def _project_global_cond(self, global_cond, local_cond=None, local_cond_mask=None):
        if self.state_query_merge or self.learned_query_merge:
            return self.project_history_image_features(global_cond, local_cond, local_cond_mask)
        return super()._project_global_cond(global_cond, local_cond, local_cond_mask)

    def project_history_image_features(
        self, features: torch.Tensor, state_features: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.learned_query_merge:
            return self.global_cond_projector(features)
        if self.state_query_merge:
            if state_features is None:
                raise ValueError("state_query_merge requires each image's own state")
            if state_mask is not None:
                if state_mask.shape != state_features.shape[:-1]:
                    raise ValueError("state_query_merge state_mask must match state tokens")
                state_features = state_features.masked_fill(~state_mask.bool()[..., None], 0)
            return self.global_cond_projector(
                features, self.local_cond_projector(state_features), state_mask,
            )
        if self.patch_merge:
            return self.global_cond_projector(features)
        if self.history_img_features_projector is None:
            raise ValueError("history image features require an image projector")
        return self.history_img_features_projector(features)

    def _update_history_attention_metric(
        self, name: str, values: torch.Tensor
    ) -> None:
        """Accumulate sample-level moments without retaining full batches.

        values has shape [layer, sample, slot] or
        [layer, sample, slot, token-within-slot].  NaNs denote padded slots.
        Heads and action-query tokens have already been averaged per sample.
        """
        values = values.detach().float()
        finite = torch.isfinite(values)
        safe_values = torch.where(finite, values, torch.zeros_like(values))

        batch_stats = {
            "sum": safe_values.double().sum(dim=1).cpu(),
            "sum_sq": safe_values.double().square().sum(dim=1).cpu(),
            "count": finite.sum(dim=1).to(dtype=torch.int64).cpu(),
            "min": torch.where(
                finite, values, torch.full_like(values, float("inf"))
            ).amin(dim=1).cpu(),
            "max": torch.where(
                finite, values, torch.full_like(values, -float("inf"))
            ).amax(dim=1).cpu(),
        }

        if name not in self._history_attention_epoch_stats:
            self._history_attention_epoch_stats[name] = batch_stats
            return

        accumulated = self._history_attention_epoch_stats[name]
        accumulated["sum"] += batch_stats["sum"]
        accumulated["sum_sq"] += batch_stats["sum_sq"]
        accumulated["count"] += batch_stats["count"]
        accumulated["min"] = torch.minimum(
            accumulated["min"], batch_stats["min"]
        )
        accumulated["max"] = torch.maximum(
            accumulated["max"], batch_stats["max"]
        )

    def _accumulate_history_attention_epoch_stats(
        self, record_data_dict: dict[str, list[torch.Tensor]]
    ) -> None:
        frame_entries = record_data_dict.get(
            "history_cross_attention_frame_logits", []
        )
        weight_entries = record_data_dict.get(
            "history_cross_attention_slot_weights", []
        )
        if not frame_entries or not weight_entries:
            return

        # History is retrieved once per anchor by learned decision-memory query
        # tokens. Treat anchors as samples and expose one retrieval layer.
        # [1, sample, head, query, slot, token-within-slot]
        frame_logits = torch.cat(frame_entries, dim=0).unsqueeze(0).float()
        # [1, sample, head, query, slot]
        slot_weights = torch.cat(weight_entries, dim=0).unsqueeze(0).float()

        # One content logit per slot while preserving the exact softmax slot
        # marginal: logsumexp over tokens belonging to that slot.
        slot_logits_per_query = torch.logsumexp(frame_logits, dim=-1)
        valid_slots = torch.isfinite(slot_logits_per_query)
        safe_slot_logits = slot_logits_per_query.masked_fill(
            ~valid_slots, -float("inf")
        )
        slot_log_probability_per_query = safe_slot_logits - torch.logsumexp(
            safe_slot_logits, dim=-1, keepdim=True
        )
        slot_log_probability_per_query = slot_log_probability_per_query.masked_fill(
            ~valid_slots, float("nan")
        )
        slot_weights = slot_weights.masked_fill(~valid_slots, float("nan"))

        # Keep samples separate for correct epoch-level variance; only heads
        # and action query tokens are averaged here.
        frame_logits_per_sample = torch.nanmean(frame_logits, dim=(2, 3))
        slot_logits_per_sample = torch.nanmean(
            slot_logits_per_query, dim=(2, 3)
        )
        slot_log_probability_per_sample = torch.nanmean(
            slot_log_probability_per_query, dim=(2, 3)
        )
        slot_weights_per_sample = torch.nanmean(slot_weights, dim=(2, 3))

        self._update_history_attention_metric(
            "frame_logit", frame_logits_per_sample
        )
        self._update_history_attention_metric(
            "slot_logit", slot_logits_per_sample
        )
        self._update_history_attention_metric(
            "slot_log_probability", slot_log_probability_per_sample
        )
        self._update_history_attention_metric(
            "slot_attention_weight", slot_weights_per_sample
        )

    def pop_history_attention_epoch_stats(
        self,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Return finalized epoch statistics and reset the accumulator."""
        finalized: dict[str, dict[str, torch.Tensor]] = {}
        for name, stats in self._history_attention_epoch_stats.items():
            count = stats["count"]
            safe_count = count.clamp_min(1).double()
            mean = stats["sum"] / safe_count
            variance = stats["sum_sq"] / safe_count - mean.square()
            std = variance.clamp_min(0).sqrt()
            has_data = count > 0

            finalized[name] = {
                "mean": torch.where(
                    has_data, mean, torch.full_like(mean, float("nan"))
                ).float(),
                "std": torch.where(
                    has_data, std, torch.full_like(std, float("nan"))
                ).float(),
                "min": torch.where(
                    has_data,
                    stats["min"],
                    torch.full_like(stats["min"], float("nan")),
                ).float(),
                "max": torch.where(
                    has_data,
                    stats["max"],
                    torch.full_like(stats["max"], float("nan")),
                ).float(),
                "count": count,
            }

        self._history_attention_epoch_stats = {}
        return finalized

    def set_skip_history_attn(self, skip_history_attn: bool):
        self.skip_history_attn = skip_history_attn

    def initialize_memory_weights(self):
        # Initialize transformer
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.blocks.apply(_basic_init)
        self.memory_retriever.apply(_basic_init)
        if self.memory_retriever.slot_gate is not None:
            self.memory_retriever.slot_gate.reset_parameters()
        if self.memory_retrieval_mode == "perception_patches":
            nn.init.normal_(
                self.memory_retriever.perception_gate.proj.weight,
                mean=0.0,
                std=1e-3,
            )
            nn.init.normal_(
                self.memory_retriever.perception_gate.proj.bias,
                mean=0.0,
                std=1e-3,
            )

        history_time_embedding = get_1d_sincos_pos_embed_from_grid(
            embed_dim=self.hidden_dim,
            pos=np.arange(self.max_history_storage_len),
        )
        self.history_time_embedding.data.copy_(
            torch.from_numpy(history_time_embedding).unsqueeze(0)
        )

    def add_history_key_encoding(
        self,
        history_latents: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Add frame/view metadata to history keys, never to stored values."""

        if history_latents.ndim != 4:
            raise ValueError(
                "history_latents must have shape [batch, slot, token, hidden]"
            )
        expected_time_tail = (
            history_latents.shape[1],
            self.hidden_dim,
        )
        if (
            time_embedding.ndim != 3
            or time_embedding.shape[0] not in {1, history_latents.shape[0]}
            or time_embedding.shape[1:] != expected_time_tail
        ):
            raise ValueError(
                "time_embedding must match (or broadcast over) history "
                "batch/slot dimensions, got "
                f"{tuple(time_embedding.shape)} for "
                f"{tuple(history_latents.shape)}"
            )
        history_keys = history_latents + time_embedding[:, :, None, :]
        if self.history_view_embedding is not None:
            if history_latents.shape[2] != self.history_view_num:
                raise ValueError(
                    "compact slots require exactly one image token per view, "
                    f"got {history_latents.shape[2]} tokens for {self.history_view_num} views"
                )
            history_keys = history_keys + self.history_view_embedding[:, None]
        return history_keys

    def project_state_conditioned_slot_features(
        self,
        global_cond: torch.Tensor,
        local_cond: torch.Tensor | None = None,
        global_cond_mask: torch.Tensor | None = None,
        local_cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project one observation into the exact S_t representation banked online."""

        if not (
            self.compact_multiview_slot
            or self.state_conditioned_history_slot
        ):
            raise RuntimeError(
                "state-conditioned slot projection is not configured"
            )
        gc = self.global_cond_projector(global_cond)
        gc = gc + self.global_cond_pos_embedding

        if local_cond is None:
            lc = None
        else:
            lc = self.local_cond_projector(local_cond)
            lc = lc + self.input_pos_embedding[
                :, 1 : 1 + self.local_cond_token_num
            ]
            if local_cond_mask is not None:
                lc = lc.clone()
                lc[local_cond_mask == 0] = self.mask_token.type_as(lc)
        return self.memory_retriever.condition_current_tokens(
            gc, lc, local_cond_mask
        )

    def project_compact_slot_features(
        self,
        global_cond: torch.Tensor,
        local_cond: torch.Tensor | None = None,
        global_cond_mask: torch.Tensor | None = None,
        local_cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backward-compatible name for compact multi-view checkpoints."""

        if not self.compact_multiview_slot:
            raise RuntimeError(
                "project_compact_slot_features requires compact_multiview_slot"
            )
        return self.project_state_conditioned_slot_features(
            global_cond=global_cond,
            local_cond=local_cond,
            global_cond_mask=global_cond_mask,
            local_cond_mask=local_cond_mask,
        )

    def _project_to_latent_space(
        self, data_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Project one decision point and prepare one observation-driven read."""

        decision_memory = data_dict.pop("decision_memory", None)
        precomputed_record_data = data_dict.pop(
            "_precomputed_memory_record_data", None
        )
        history_latents: torch.Tensor | None = None
        history_state_features = data_dict.pop("history_state_features", None)
        history_state_mask = data_dict.pop("history_state_mask", None)
        history_mask = data_dict.pop("history_mask", None)
        history_shadow_mask = data_dict.pop("history_shadow_mask", None)
        history_shadow_steps = data_dict.pop("history_shadow_steps", None)
        memory_gate_val = data_dict.pop("memory_gate_val", None)

        if "history_img_features" in data_dict:
            history_img_features = data_dict.pop("history_img_features")
            history_len = history_img_features.shape[1]
            if history_len > self.max_history_storage_len:
                raise ValueError("history_len exceeds Active plus Shadow storage capacity")
            history_latents = self.project_history_image_features(
                history_img_features, history_state_features, history_state_mask,
            )
            if history_len == 0:
                positioned_history_latents = history_latents
            else:
                time_embedding = self.history_time_embedding[:, -history_len:]
                positioned_history_latents = self.add_history_key_encoding(
                    history_latents, time_embedding
                )

        input_dict = self._project_current_inputs(data_dict)
        if history_latents is not None:
            if self.memory_retrieval_mode == "perception_patches":
                input_dict["history_latents"] = history_latents
                input_dict["history_key_latents"] = positioned_history_latents
            else:
                input_dict["history_latents"] = positioned_history_latents
        if history_mask is not None:
            input_dict["history_mask"] = history_mask
        if history_shadow_mask is not None:
            input_dict["history_shadow_mask"] = history_shadow_mask
        if history_shadow_steps is not None:
            input_dict["history_shadow_steps"] = history_shadow_steps
        if memory_gate_val is not None:
            input_dict["memory_gate_val"] = memory_gate_val
        if decision_memory is not None:
            if (
                decision_memory.ndim != 3
                or decision_memory.shape[0] != input_dict["x"].shape[0]
                or decision_memory.shape[1] != self.decision_memory_token_num
                or decision_memory.shape[2] != self.hidden_dim
            ):
                raise ValueError(
                    "decision_memory must have shape "
                    f"[batch, {self.decision_memory_token_num}, {self.hidden_dim}], "
                    f"got {tuple(decision_memory.shape)}"
                )
            input_dict["decision_memory"] = decision_memory
        if precomputed_record_data is not None:
            input_dict["_precomputed_memory_record_data"] = (
                precomputed_record_data
            )

        self._add_memory_gate_token(input_dict)
        return input_dict

    def _project_current_inputs(
        self, data_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Project tokens and route state to DiT or the legacy retrieval path."""

        local_cond_mask = data_dict.get("local_cond_mask")
        if (self.patch_merge or self.state_query_merge or self.learned_query_merge) and "global_cond_mask" in data_dict:
            data_dict = dict(data_dict)
            data_dict["global_cond_mask"] = self.global_cond_projector.merge_mask(
                data_dict["global_cond_mask"]
            )
        input_dict = super()._project_to_latent_space(data_dict)
        local_token_num = self.local_cond_token_num

        if local_token_num > 0:
            x = input_dict["x"]
            expected_token_num = 1 + local_token_num + self.action_token_num
            if x.shape[1] != expected_token_num:
                raise ValueError(
                    f"Expected {expected_token_num} projected input tokens, got "
                    f"{x.shape[1]}"
                )
            input_dict["local_cond"] = x[:, 1 : 1 + local_token_num]
            if not self.state_as_dit_token:
                input_dict["x"] = torch.cat(
                    [x[:, :1], x[:, 1 + local_token_num :]], dim=1
                )
            if local_cond_mask is not None:
                input_dict["local_cond_mask"] = local_cond_mask

        return input_dict

    def _add_memory_gate_token(self, input_dict: dict[str, torch.Tensor]) -> None:
        if self.memory_gate_tokens is None:
            return
        if "memory_gate_val" not in input_dict:
            raise ValueError(
                "memory_gate_val must be provided when memory gate tokens are used"
            )
        binary_gate_val = (input_dict["memory_gate_val"] > 0.5).long()
        batch_memory_gate_tokens = self.memory_gate_tokens[
            binary_gate_val, None, :
        ]
        input_dict["x"] = torch.cat(
            [batch_memory_gate_tokens, input_dict["x"]], dim=1
        )

    def _project_to_latent_space_multi_traj(
        self, data_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Project history image features into memory slots
        Will also call the super class's _process_data_dict to process the rest of the data
        This function is only called during training when multiple supervised trajectories are provided at the same time
        data_dict: {
            # Same as no-history version
            "noisy_action": (batch, traj_num, data_length, action_dim),
            "step": (batch,),
            "global_cond": (batch, traj_num, global_cond_token_num, global_cond_dim),
            "local_cond": (batch, traj_num, local_cond_token_num, local_cond_dim),
            "global_cond_mask": (batch, traj_num, global_cond_token_num),
            "local_cond_mask": (batch, traj_num, local_cond_token_num),
            "noisy_traj_mask": (batch, traj_num, data_length),
            "step_mask": (batch, traj_num),
            # History-specific
            "history_img_features": (batch, traj_num, data_length, feature_dim), # Optional
            "history_mask": (batch, traj_num),
            "memory_gate_val": (batch, traj_num), # Optional
            "training_traj_indices": (batch, max_training_traj_num), # Optional
        }

        return: {
            "x": (batch*traj_num, local_cond_token_num + action_token_num + 1(denoising step) + 1(memory gate token, optional), hidden_dim),
            "global_cond": (batch*traj_num, global_cond_token_num, hidden_dim),
            "global_cond_mask": (batch*traj_num, global_cond_token_num),

            "step": (batch*traj_num,),
            "history_latents": (batch*traj_num, max_history_len, token_num, hidden_dim),
            "history_mask": (batch*traj_num, max_history_len),
            "memory_gate_val": (batch*traj_num,),
        }
        if self.max_training_traj_num > 0, traj_num will be replaced by self.max_training_traj_num if exceeds
        """

        streaming_training = bool(data_dict.pop("_streaming_training", False))
        traj_num = data_dict["noisy_action"].shape[1]
        batch_size = data_dict["noisy_action"].shape[0]
        device = data_dict["noisy_action"].device

        if not streaming_training:
            assert (
                traj_num >= self.max_history_len + 1
            ), f"To make sure all the history time embeddings are trained,traj_num ({traj_num}) must be at least equal to max_history_len + 1 ({self.max_history_len + 1}) in each training step"
        # Project all image slots before constructing the causal memory bank.
        history_latents: torch.Tensor | None = None

        if (
            (self.patch_merge or self.state_query_merge or self.learned_query_merge or self.history_img_features_projector is not None)
            and not self.compact_multiview_slot
            and not self.state_conditioned_history_slot
        ):
            projected_history_img_features_latent = self.project_history_image_features(
                data_dict["history_img_features"],
                data_dict.get("local_cond") if self.state_query_merge else None,
                data_dict.get("local_cond_mask") if self.state_query_merge else None,
            )  # (batch, traj_num, traj_length, hidden_dim)
            history_latents = projected_history_img_features_latent

        # Step 2: Determine effective trajectories and sample non-history tensors early
        has_training_traj_indices = "training_traj_indices" in data_dict
        if has_training_traj_indices:
            traj_indices = data_dict["training_traj_indices"]  # (batch, effective_traj_num)
            effective_traj_num = traj_indices.shape[1]
        else:
            traj_indices = torch.arange(traj_num, device=device).unsqueeze(0).expand(batch_size, -1)
            effective_traj_num = traj_num

        reshaped_data_dict: dict[str, torch.Tensor] = {}
        non_history_keys = [
            "noisy_action",
            "global_cond",
            "local_cond",
            "global_cond_mask",
            "local_cond_mask",
            "noisy_traj_mask",
            "step_mask",
        ]
        for key in non_history_keys:
            if key in data_dict:
                tensor = data_dict[key]
                if has_training_traj_indices:
                    batch_idx_2d = torch.arange(batch_size, device=device)[:, None]
                    tensor = tensor[batch_idx_2d, traj_indices]
                reshaped_data_dict[key] = einops.rearrange(
                    tensor, "batch traj_num ... -> (batch traj_num) ..."
                )

        reshaped_data_dict["step"] = (
            data_dict["step"].repeat_interleave(effective_traj_num).view(-1)
        )

        # Step 3: Project non-history tensors (only effective trajectories go through the projectors)
        input_dict = self._project_current_inputs(reshaped_data_dict)
        input_dict["step"] = reshaped_data_dict["step"]
        if (
            self.compact_multiview_slot
            or self.state_conditioned_history_slot
        ):
            if has_training_traj_indices:
                raise ValueError(
                    "state-conditioned slots require all ordered trajectories; "
                    "disable max_training_traj_num subsampling"
                )
            flat_slot_features = (
                self.memory_retriever.condition_current_tokens(
                    input_dict["global_cond"],
                    input_dict.get("local_cond"),
                    input_dict.get("local_cond_mask"),
                )
            )
            all_current_slot_features = einops.rearrange(
                flat_slot_features,
                "(batch traj_num) token hidden -> batch traj_num token hidden",
                batch=batch_size,
                traj_num=traj_num,
            )
            history_latents = all_current_slot_features
            if self.compact_multiview_slot:
                input_dict["all_current_slot_features"] = (
                    all_current_slot_features
                )
        if history_latents is None:
            raise ValueError("history image features are required by the configured memory")
        # (batch, traj_num, token_num, hidden_dim)
        # Keep one un-positioned latent group per source anchor. The causal
        # policy uses this tensor to construct a retained-slot buffer that can
        # be non-contiguous when gate-based eviction is enabled.
        input_dict["all_history_latents"] = history_latents
        # The flag was consumed above; the causal bank supplies retained slots
        # later, so streaming does not need a second, sliding-window history.
        if streaming_training:
            return input_dict

        # Step 4: Construct history windows via vectorized gather (replaces Python for-loop)
        # For trajectory index i, history window position k (0..max_history_len-1)
        # maps to source index: i - max_history_len + k
        window_offsets = torch.arange(self.max_history_len, device=device)
        # source_indices[b, j, k] = traj_indices[b, j] - max_history_len + k
        source_indices = traj_indices[:, :, None] - self.max_history_len + window_offsets[None, None, :]
        # (batch, effective_traj_num, max_history_len)
        valid_mask = source_indices >= 0
        source_indices_clamped = source_indices.clamp(min=0)

        batch_idx_3d = torch.arange(batch_size, device=device)[:, None, None]
        merged_history_latents = history_latents[batch_idx_3d, source_indices_clamped]
        # (batch, effective_traj_num, max_history_len, token_num, hidden_dim)
        merged_history_latents = merged_history_latents * valid_mask[:, :, :, None, None]

        if "history_mask" in data_dict:
            full_history_mask = data_dict["history_mask"]  # (batch, traj_num)
            merged_history_masks = full_history_mask[batch_idx_3d, source_indices_clamped]
            # (batch, effective_traj_num, max_history_len)
            merged_history_masks = merged_history_masks & valid_mask
        else:
            merged_history_masks = valid_mask

        unpositioned_history_latents = einops.rearrange(
            merged_history_latents, "batch traj_num ... -> (batch traj_num) ... "
        )  # (batch*effective_traj_num, max_history_len, token_num, hidden_dim)

        positioned_history_latents = self.add_history_key_encoding(
            unpositioned_history_latents, self.history_time_embedding[:, -self.max_history_len:]
        )
        if self.memory_retrieval_mode == "perception_patches":
            input_dict["history_latents"] = unpositioned_history_latents
            input_dict["history_key_latents"] = positioned_history_latents
        else:
            input_dict["history_latents"] = positioned_history_latents
        input_dict["history_mask"] = einops.rearrange(
            merged_history_masks, "batch traj_num ... -> (batch traj_num) ... "
        )

        if "memory_gate_val" in data_dict:
            memory_gate_val = data_dict["memory_gate_val"]
            if has_training_traj_indices:
                batch_idx_2d = torch.arange(batch_size, device=device)[:, None]
                memory_gate_val = memory_gate_val[batch_idx_2d, traj_indices]
            input_dict["memory_gate_val"] = einops.rearrange(
                memory_gate_val, "batch traj_num ... -> (batch traj_num) ..."
            )
        self._add_memory_gate_token(input_dict)

        return input_dict

    def parallel_forward(
        self, data_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Only used when training with multiple ground-truth trajectories
        data_dict: {
            "noisy_action": (batch, traj_num, traj_length, action_dim),
            "step": (batch,),
            "global_cond": (batch, traj_num, global_cond_token_num, global_cond_dim),
            "local_cond": (batch, traj_num, local_cond_token_num, local_cond_dim),
            "history_mask": (batch, traj_num),
            "memory_gate_val": (batch, traj_num),
            "training_traj_indices": (batch, max_training_traj_num), # Optional, if provided, will only use the indexed trajectories for training
        }
        return: {
            "action": (batch, traj_num, traj_length, action_dim),
        }
        if "training_traj_indices" in data_dict, traj_num will be replaced by max_training_traj_num
        """

        batch_size = data_dict["noisy_action"].shape[0]
        if "training_traj_indices" in data_dict:
            traj_num = data_dict["training_traj_indices"].shape[1]
        else:
            # raise RuntimeError("training_traj_indices is not provided")
            traj_num = data_dict["noisy_action"].shape[1]
        input_dict = self.prepare_parallel_forward(data_dict)
        return self.finish_parallel_forward(input_dict, batch_size, traj_num)

    def prepare_parallel_forward(
        self, data_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Project a multi-anchor batch before memory retrieval and DiT blocks.

        Keeping this stage separate allows the policy to construct decision
        memories and gate-based eviction causally before the DiT action blocks.
        """

        return self._project_to_latent_space_multi_traj(data_dict)

    def prepare_single_memory_retrieval(
        self, data_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Project the current observation for a single causal memory read."""

        input_dict = self._project_to_latent_space(copy.deepcopy(data_dict))
        retrieval_keys = (
            "global_cond",
            "global_cond_mask",
            "local_cond",
            "local_cond_mask",
            "history_latents",
            "history_key_latents",
            "history_mask",
            "history_shadow_mask",
            "history_shadow_steps",
            "memory_gate_val",
        )
        return {
            key: input_dict[key]
            for key in retrieval_keys
            if key in input_dict
        }

    def retrieve_decision_memory(
        self,
        retrieval_inputs: dict[str, torch.Tensor],
        record_data_dict: dict[str, list[torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Run only observation-driven history retrieval, without DiT blocks."""

        return self.memory_retriever(
            global_cond=retrieval_inputs["global_cond"],
            global_cond_mask=retrieval_inputs.get("global_cond_mask"),
            local_cond=retrieval_inputs.get("local_cond"),
            local_cond_mask=retrieval_inputs.get("local_cond_mask"),
            history_latents=retrieval_inputs.get("history_latents"),
            history_key_latents=retrieval_inputs.get("history_key_latents"),
            history_mask=retrieval_inputs.get("history_mask"),
            history_shadow_mask=retrieval_inputs.get("history_shadow_mask"),
            history_shadow_steps=retrieval_inputs.get("history_shadow_steps"),
            memory_gate_val=retrieval_inputs.get("memory_gate_val"),
            skip_history=self.skip_history_attn,
            record_data_dict=record_data_dict,
        )

    def finish_parallel_forward(
        self,
        input_dict: dict[str, torch.Tensor],
        batch_size: int,
        traj_num: int,
    ) -> dict[str, torch.Tensor]:
        """Run all DiT blocks and restore the batch/anchor dimensions."""

        expected_flat_batch = batch_size * traj_num
        if input_dict["x"].shape[0] != expected_flat_batch:
            raise ValueError(
                "Projected input batch does not match batch_size * traj_num: "
                f"{input_dict['x'].shape[0]} != {batch_size} * {traj_num}"
            )
        if "decision_memory" in input_dict and (
            input_dict["decision_memory"].shape[0] != expected_flat_batch
        ):
            raise ValueError(
                "decision_memory must have the same flattened batch dimension "
                "as x"
            )

        parallel_results = self._get_results(self._run_blocks(input_dict))
        results: dict[str, torch.Tensor] = {
            "action": einops.rearrange(
                parallel_results["action"],
                "(batch traj_num) ... -> batch traj_num ...",
                batch=batch_size,
                traj_num=traj_num,
            ),
            "decision_memory": einops.rearrange(
                parallel_results["decision_memory"],
                "(batch traj_num) ... -> batch traj_num ...",
                batch=batch_size,
                traj_num=traj_num,
            ),
        }
        return results

    def _run_blocks(self, input_dict: dict[str, torch.Tensor]):

        record_entries = list(self.record_data_entries)
        collect_epoch_stats = (
            self.training and self.record_history_attention_epoch_stats
        )
        if collect_epoch_stats:
            for entry in [
                "history_cross_attention_frame_logits",
                "history_cross_attention_slot_weights",
            ]:
                if entry not in record_entries:
                    record_entries.append(entry)

        precomputed_record_data = input_dict.pop(
            "_precomputed_memory_record_data", None
        )
        if precomputed_record_data is not None:
            record_data_dict = precomputed_record_data
        elif len(record_entries) > 0:
            record_data_dict: dict[str, list[torch.Tensor]] | None = {
                entry: [] for entry in record_entries
            }
            if "memory_gate_val" in input_dict and "memory_gate_val" in record_data_dict:
                record_data_dict["memory_gate_val"].append(input_dict["memory_gate_val"].clone())
        else:
            record_data_dict = None

        if "decision_memory" in input_dict:
            decision_memory = input_dict["decision_memory"]
        else:
            decision_memory = self.memory_retriever(
                global_cond=input_dict["global_cond"],
                global_cond_mask=input_dict.get("global_cond_mask"),
                local_cond=input_dict.get("local_cond"),
                local_cond_mask=input_dict.get("local_cond_mask"),
                history_latents=input_dict.get("history_latents"),
                history_key_latents=input_dict.get("history_key_latents"),
                history_mask=input_dict.get("history_mask"),
                history_shadow_mask=input_dict.get("history_shadow_mask"),
                history_shadow_steps=input_dict.get("history_shadow_steps"),
                memory_gate_val=input_dict.get("memory_gate_val"),
                skip_history=self.skip_history_attn,
                record_data_dict=record_data_dict,
            )
            input_dict["decision_memory"] = decision_memory

        for block in self.blocks:
            input_dict["x"] = block(
                x=input_dict["x"],
                global_cond=input_dict["global_cond"],
                decision_memory=decision_memory,
                global_cond_mask=input_dict.get("global_cond_mask"),
            )

        if record_data_dict is not None:
            self.recorded_data_dict = record_data_dict
            if collect_epoch_stats:
                self._accumulate_history_attention_epoch_stats(record_data_dict)

        return input_dict

    def _get_results(
        self, input_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        results = super()._get_results(input_dict)
        results["decision_memory"] = input_dict["decision_memory"]
        return results

    def forward(
        self,
        data_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        data_dict = copy.deepcopy(data_dict)  # Avoid modifying the original data_dict

        input_dict = self._project_to_latent_space(data_dict)
        input_dict = self._run_blocks(input_dict)
        return self._get_results(input_dict)
