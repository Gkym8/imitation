"""State-modulated learned queries pool independent local image regions."""

from math import isqrt

import torch
from torch import nn
from timm.models.vision_transformer import RmsNorm

from imitation_learning.models.denoising_networks.modules import CrossAttention


class StateQueryPatchMerger(nn.Module):
    """One learned query per spatial region, shared across views and time.

    State only modulates queries: ``q = (1 + gamma(state)) * Q + beta(state)``.
    Keys and values come from that region's image patches. No query residual
    is added to the output. Views are grouped independently, in row-major order.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, patch_token_num: int,
        merge_size: int, head_num: int,
    ):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim), ("hidden_dim", hidden_dim),
            ("patch_token_num", patch_token_num), ("merge_size", merge_size),
            ("head_num", head_num),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        grid = isqrt(patch_token_num)
        if grid * grid != patch_token_num or grid % merge_size:
            raise ValueError("state_query_merge requires a square patch grid divisible by merge_size")
        if hidden_dim % head_num:
            raise ValueError("state_query_merge hidden_dim must be divisible by head_num")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.patch_token_num = patch_token_num
        self.grid = grid
        self.merge_size = merge_size
        self.output_token_num = patch_token_num // merge_size**2
        self.query_tokens = nn.Parameter(torch.empty(self.output_token_num, hidden_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.state_modulation = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.image_norm = nn.LayerNorm(input_dim)
        self.image_projector = nn.Linear(input_dim, hidden_dim)
        self.attention = CrossAttention(
            dim=hidden_dim, head_num=head_num, qkv_bias=True,
            qk_norm=True, norm_layer=RmsNorm,
        )
        # Start from ordinary learned-query pooling, then learn state changes.
        nn.init.zeros_(self.state_modulation[-1].weight)
        nn.init.zeros_(self.state_modulation[-1].bias)

    def _group(self, patches: torch.Tensor) -> torch.Tensor:
        token_num, dim = patches.shape[-2:]
        if token_num <= 0 or token_num % self.patch_token_num:
            raise ValueError("state_query_merge input must contain complete per-view patch grids")
        views = token_num // self.patch_token_num
        side = self.grid // self.merge_size
        grouped = patches.reshape(
            *patches.shape[:-2], views, side, self.merge_size,
            side, self.merge_size, dim,
        ).transpose(-4, -3)
        return grouped.reshape(
            *patches.shape[:-2], views * self.output_token_num,
            self.merge_size**2, dim,
        )

    def merge_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return self._group(mask.bool().unsqueeze(-1)).squeeze(-1).all(dim=-1)

    def forward(
        self, patches: torch.Tensor, state_tokens: torch.Tensor,
        state_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if patches.ndim < 3 or patches.shape[-1] != self.input_dim:
            raise ValueError(f"state_query_merge expects [..., tokens, {self.input_dim}]")
        if (
            state_tokens.ndim != patches.ndim
            or state_tokens.shape[:-2] != patches.shape[:-2]
            or state_tokens.shape[-1] != self.hidden_dim
            or state_tokens.shape[-2] <= 0
        ):
            raise ValueError("state_query_merge state tokens must match image batch dimensions and hidden_dim")
        if state_mask is None:
            state = state_tokens.mean(dim=-2)
        else:
            if state_mask.shape != state_tokens.shape[:-1]:
                raise ValueError("state_query_merge state_mask must match state tokens")
            mask = state_mask.bool()
            state = state_tokens.masked_fill(~mask[..., None], 0).sum(dim=-2)
            state = state / mask.sum(dim=-1, keepdim=True).clamp_min(1)
        gamma, beta = self.state_modulation(state).chunk(2, dim=-1)
        regions = self._group(self.image_projector(self.image_norm(patches)))
        region_num = regions.shape[-3]
        views = region_num // self.output_token_num
        queries = self.query_tokens.repeat(views, 1)
        queries = (1 + gamma[..., None, :]) * queries + beta[..., None, :]
        output = self.attention(
            queries.reshape(-1, 1, self.hidden_dim),
            regions.reshape(-1, self.merge_size**2, self.hidden_dim),
        )
        return output.reshape(*patches.shape[:-2], region_num, self.hidden_dim)
