"""Adapted from diffusion policy codebase (https://github.com/real-stanford/diffusion_policy)"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import use_fused_attn



class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, group_num: int = 8
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(group_num, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim: int = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, 1)
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)  # TODO: why 10000?
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SSMax(nn.Module):
    def __init__(self, scaling_param: float):
        super().__init__()
        self.scaling_param: float = scaling_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SSMax to the last dimension of x.
        https://arxiv.org/pdf/2501.19399
        """
        input_shape = x.shape
        x = x.reshape(-1, input_shape[-1])
        log_n = torch.log(torch.tensor(x.shape[-1], device=x.device))
        n_powers = (self.scaling_param * x * log_n).exp()  # (..., n)
        x = n_powers / (n_powers.sum(dim=-1, keepdim=True) + 1e-6)
        x = x.reshape(input_shape)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_num: int,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer: type[nn.Module],
    ):
        super().__init__()

        self.dim: int = dim
        self.head_num: int = head_num
        self.head_dim: int = dim // head_num
        self.attn_scale: float = 1.0 / math.sqrt(self.head_dim)
        self.qkv_bias: bool = qkv_bias
        self.qk_norm: bool = qk_norm
        self.norm_layer: type[nn.Module] = norm_layer
        self.fused_attn: bool = use_fused_attn()

        self.q: nn.Linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv: nn.Linear = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm: nn.Module = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm: nn.Module = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.linear: nn.Linear = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, cond_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: (batch_size, token_num, dim)
        cond: (batch_size, cond_token_num, dim)
        cond_mask: (batch_size, cond_token_num)
        """
        assert len(x.shape) == 3
        batch_size, token_num, hidden_dim = x.shape
        assert hidden_dim == self.dim
        assert (
            len(cond.shape) == 3
            and cond.shape[0] == batch_size
            and cond.shape[2] == hidden_dim
        ), f"cond.shape: {cond.shape}, batch_size: {batch_size}, hidden_dim: {hidden_dim}"
        cond_token_num = cond.shape[1]

        q = (
            self.q.forward(x)  # (batch_size, token_num, dim)
            .reshape(batch_size, token_num, self.head_num, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # (batch_size, head_num, token_num, head_dim)
        kv = (
            self.kv.forward(cond)
            .reshape(batch_size, cond_token_num, 2, self.head_num, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )  # (2, batch_size, head_num, cond_token_num, head_dim)
        k, v = kv.unbind(dim=0)  # (batch_size, head_num, cond_token_num, head_dim)

        q: torch.Tensor = self.q_norm(q)
        k: torch.Tensor = self.k_norm(k)

        if cond_mask is not None:
            assert cond_mask.shape == (batch_size, cond_token_num)
            attn_mask = cond_mask[:, None, None, :].expand(
                batch_size, self.head_num, token_num, cond_token_num
            )
        else:
            attn_mask = None

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False
            )
        else:
            attn = (
                q @ k.transpose(-2, -1)
            ) * self.attn_scale  # (batch_size, head_num, token_num, cond_token_num)
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask.logical_not(), -float("inf"))
            attn = attn.softmax(dim=-1)
            x = attn @ v  # (batch_size, head_num, token_num, head_dim)

        x = x.transpose(1, 2).reshape(
            batch_size, token_num, self.dim
        )  # (batch_size, token_num, dim)
        return self.linear.forward(x)


class HistoryCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_num: int,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer: type[nn.Module],
        attention_type: str,
        ssmax_scaling_param: float | None,
    ):
        super().__init__()

        self.dim: int = dim
        self.head_num: int = head_num
        self.head_dim: int = dim // head_num
        self.qkv_bias: bool = qkv_bias
        self.qk_norm: bool = qk_norm
        self.norm_layer: type[nn.Module] = norm_layer
        self.fused_attn: bool = use_fused_attn()

        self.q: nn.Linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv: nn.Linear = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm: nn.Module = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm: nn.Module = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.linear: nn.Linear = nn.Linear(
            dim, dim, bias=False
        )  # No bias, so when the history masks are all 0, the output will be 0

        assert attention_type in ["token_wise", "block_wise"]
        self.attention_type: str = attention_type

        # self.projector: nn.Module = Projector(projector_type, latent_token_num*self.head_dim, dim)
        if ssmax_scaling_param is not None:
            self.ssmax: nn.Module | None = SSMax(ssmax_scaling_param)
        else:
            self.ssmax = None

    def forward(
        self,
        x: torch.Tensor,
        history_latents: torch.Tensor,
        history_key_latents: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        record_data_dict: dict[str, list[torch.Tensor]] | None = None,
        history_log_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (batch_size, token_num, dim)
        history_latents: (batch_size, history_len, token_num, dim), values
        history_key_latents: optional keys with the same shape. This permits
            adding frame-position embeddings to keys without contaminating
            the stored visual values, as in MemoryVLA's perception memory.
        history_mask: (batch_size, history_len), 1 for valid, 0 for invalid
        history_log_gate: (batch_size, history_len, head_num), log of the
            observation-conditioned slot gate, shared by a slot's patch tokens

        Calculate the attention score of x and each history latent. Each history latent should be treated as a whole.
        The attention score should have shape (batch_size, history_len)
        The output should be a weighted sum of the history latents, weighted by the attention score, with shape (batch_size, token_num, dim)
        """

        # print(f"{x.shape=}, {history_latents.shape=}, {history_mask.shape=}")
        # print(f"{history_mask=}")
        # print(f"{x.mean(dim=-1)=}, {x.std(dim=-1)=}")
        # print(f"{history_latents.squeeze(2).mean(dim=-1)=}, {history_latents.squeeze(2).std(dim=-1)=}")
        # mean_history_latents = history_latents.squeeze(2).mean(dim=-1)
        # first_row_history_latents = mean_history_latents[0]
        # print(f"{torch.allclose(mean_history_latents, first_row_history_latents)=}")
        assert len(x.shape) == 3
        batch_size, token_num, hidden_dim = x.shape
        assert hidden_dim == self.dim

        assert (
            len(history_latents.shape) == 4
            and history_latents.shape[0] == batch_size
            and history_latents.shape[3] == hidden_dim
        ), f"history_latents.shape: {history_latents.shape}, batch_size: {batch_size}, token_num: {token_num}, hidden_dim: {hidden_dim}"
        if history_key_latents is None:
            history_key_latents = history_latents
        if history_key_latents.shape != history_latents.shape:
            raise ValueError(
                "history_key_latents must match history_latents, got "
                f"{tuple(history_key_latents.shape)} and "
                f"{tuple(history_latents.shape)}"
            )
        if self.attention_type == "block_wise":
            assert (
                history_latents.shape[2] == token_num
            ), f"history_latents.shape: {history_latents.shape}, batch_size: {batch_size}, token_num: {token_num}, hidden_dim: {hidden_dim}"
            history_token_num = token_num
        else:
            history_token_num = history_latents.shape[2]

        history_len = history_latents.shape[1]

        # Multi-head cross attention:
        # head is applied to the feature dimension, and will be permuted to the dimension right after batch dimension
        # all the tokens of each latent will be merged into a single token for attention score calculation
        q = (
            self.q.forward(x)  # (batch_size, token_num, dim)
            .reshape(batch_size, token_num, self.head_num, self.head_dim)
            .permute(0, 2, 1, 3)  # (batch_size, head_num, token_num, head_dim)
        )
        if history_key_latents is history_latents:
            kv = (
                self.kv.forward(history_latents)
                .reshape(
                    batch_size,
                    history_len,
                    history_token_num,
                    2,
                    self.head_num,
                    self.head_dim,
                )
                .permute(3, 0, 4, 1, 2, 5)
            )
            k, v = kv.unbind(dim=0)
        else:
            # Keep the original joint KV projector/checkpoint layout while
            # allowing position-aware keys and content-only values.
            kv_bias = self.kv.bias
            projected_keys = F.linear(
                history_key_latents,
                self.kv.weight[: self.dim],
                None if kv_bias is None else kv_bias[: self.dim],
            )
            projected_values = F.linear(
                history_latents,
                self.kv.weight[self.dim :],
                None if kv_bias is None else kv_bias[self.dim :],
            )
            k = projected_keys.reshape(
                batch_size,
                history_len,
                history_token_num,
                self.head_num,
                self.head_dim,
            ).permute(0, 3, 1, 2, 4)
            v = projected_values.reshape(
                batch_size,
                history_len,
                history_token_num,
                self.head_num,
                self.head_dim,
            ).permute(0, 3, 1, 2, 4)

        q: torch.Tensor = self.q_norm(q)
        k: torch.Tensor = self.k_norm(k)

        if self.attention_type == "block_wise":

            q = q.reshape(batch_size, self.head_num, 1, token_num * self.head_dim)
            k = k.reshape(
                batch_size, self.head_num, history_len, token_num * self.head_dim
            )
            v = v.reshape(
                batch_size, self.head_num, history_len, token_num * self.head_dim
            )

            if history_mask is not None:
                assert history_mask.shape == (batch_size, history_len)
                attn_mask = (
                    history_mask[:, None, :]
                    .expand(batch_size, self.head_num, history_len)
                    .unsqueeze(2)
                )  # (batch_size, head_num, 1, history_len)
            else:
                attn_mask = None
            attn_scale = 1.0 / math.sqrt(self.head_dim * token_num)

        elif self.attention_type == "token_wise":
            k = k.view(
                batch_size,
                self.head_num,
                history_len * history_token_num,
                self.head_dim,
            )
            v = v.view(
                batch_size,
                self.head_num,
                history_len * history_token_num,
                self.head_dim,
            )

            if history_mask is not None:
                assert history_mask.shape == (batch_size, history_len)
                attn_mask = (
                    history_mask[:, None, None, :, None]
                    .expand(
                        batch_size,
                        self.head_num,
                        token_num,
                        history_len,
                        history_token_num,
                    )
                    .reshape(
                        batch_size,
                        self.head_num,
                        token_num,
                        history_len * history_token_num,
                    )
                )
            else:
                attn_mask = None

            attn_scale = 1.0 / math.sqrt(self.head_dim)

        else:
            raise ValueError(
                f"Invalid attention type: {self.attention_type}. Only token_wise and chunk_wise are supported."
            )

        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #         q, k, v, attn_mask=attn_mask, is_causal=False
        #     )
        #     # token_wise: (batch_size, head_num, token_num, head_dim)
        #     # chunk_wise: (batch_size, head_num, 1, token_num * head_dim)
        #     x = torch.nan_to_num(
        #         x, nan=0.0, posinf=0.0, neginf=0.0
        #     )  # x will be nan if all the attention score is -inf
        # else:

        attn = (q @ k.transpose(-2, -1)) * attn_scale
        # Chunk-wise: (batch_size, head_num, 1, history_len)
        # Token-wise: (batch_size, head_num, token_num, history_len*history_token_num)

        if history_log_gate is not None:
            if history_log_gate.shape != (batch_size, history_len, self.head_num):
                raise ValueError(
                    "history_log_gate must match [batch, slot, head]"
                )
            if self.ssmax is not None:
                raise ValueError("Slot gates require standard Softmax attention")
            gate_logits = history_log_gate.permute(0, 2, 1).to(attn)
            if self.attention_type == "token_wise":
                gate_logits = gate_logits.repeat_interleave(history_token_num, dim=-1)
            attn = attn + gate_logits[:, :, None, :]

        # Keep the final pre-softmax values for offline history-attention
        # analysis. Restoring the history axes gives an unambiguous
        # [slot, token-within-slot] representation. Invalid/padded slots are
        # stored as NaN so they do not contaminate epoch statistics.
        if (
            record_data_dict is not None
            and "history_cross_attention_frame_logits" in record_data_dict
            and self.attention_type == "token_wise"
        ):
            frame_logits = attn.reshape(
                batch_size,
                self.head_num,
                token_num,
                history_len,
                history_token_num,
            )
            if history_mask is not None:
                valid_slots = history_mask[:, None, None, :, None].bool()
                frame_logits = frame_logits.masked_fill(
                    ~valid_slots, float("nan")
                )
            record_data_dict["history_cross_attention_frame_logits"].append(
                frame_logits.detach()
            )

        if (
            record_data_dict is not None
            and "history_attention_logits" in record_data_dict
        ):
            record_data_dict["history_attention_logits"].append(
                attn.detach().clone()
            )

        if attn_mask is not None:
            masked_attn = attn.masked_fill(attn_mask.logical_not(), -float("inf"))
            # All-padding samples must also have finite backwards.
            masked_attn = torch.where(
                attn_mask.any(dim=-1, keepdim=True), masked_attn,
                torch.zeros_like(masked_attn),
            )
            if self.ssmax is not None:
                attn = self.ssmax(masked_attn)
            else:
                attn = masked_attn.softmax(dim=-1)
            attn = attn.masked_fill(
                attn_mask.logical_not(), 0
            )  # set attn to 0 if masked_attn is all -inf
        elif self.ssmax is not None:
            attn = self.ssmax(attn)
        else:
            attn = attn.softmax(dim=-1)

        if (
            record_data_dict is not None
            and "history_cross_attention_slot_weights" in record_data_dict
        ):
            if self.attention_type == "token_wise":
                slot_weights = attn.reshape(
                    batch_size,
                    self.head_num,
                    token_num,
                    history_len,
                    history_token_num,
                ).sum(dim=-1)
            else:
                # Block-wise attention already has one probability per slot:
                # [batch, head, 1, history_len].
                slot_weights = attn
            record_data_dict["history_cross_attention_slot_weights"].append(
                slot_weights.detach()
            )

        if (
            record_data_dict is not None
            and "history_cross_attention" in record_data_dict
        ):
            record_data_dict["history_cross_attention"].append(attn.detach().clone())

        x = attn @ v
        # token_wise: (batch_size, head_num, token_num, head_dim)
        # chunk_wise: (batch_size, head_num, 1, token_num * head_dim)

        x = x.reshape(batch_size, self.head_num, token_num, self.head_dim)
        x = x.permute(0, 2, 1, 3)  # (batch_size, token_num, head_num, head_dim)
        x = x.reshape(batch_size, token_num, self.dim)

        return self.linear.forward(x)
