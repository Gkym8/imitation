"""Local SigLIP-style MAP pooling."""

from math import isqrt

import torch
from torch import nn

from imitation_learning.models.common.spatial_merge import SpatialMergeSize, spatial_merge_shape


class LearnedQueryPatchMerger(nn.Module):
    """One query per local region, shared across views and time; no state fusion."""

    def __init__(
        self, input_dim: int, hidden_dim: int, patch_token_num: int,
        merge_size: SpatialMergeSize, head_num: int, cls_token: bool = False,
    ):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim), ("hidden_dim", hidden_dim),
            ("patch_token_num", patch_token_num), ("head_num", head_num),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_dim % head_num:
            raise ValueError("learned_query_merge hidden_dim must be divisible by head_num")
        if not isinstance(cls_token, bool):
            raise ValueError("cls_token must be boolean")
        self.cls_token = cls_token
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.patch_token_num = patch_token_num
        grid = isqrt(patch_token_num)
        height, width = spatial_merge_shape(merge_size)
        if grid * grid != patch_token_num or grid % height or grid % width:
            raise ValueError("learned_query_merge requires a square patch grid divisible by spatial_merge_size")
        self.grid = grid
        self.merge_height, self.merge_width = height, width
        self.merge_area = height * width
        self.output_token_num = patch_token_num // self.merge_area
        self.query_tokens = nn.Parameter(torch.empty(self.output_token_num, hidden_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        # Retain the existing DINO-to-policy feature adapter. The MAP head
        # below uses ordinary multihead attention, without extra Q/K norms.
        self.image_norm = nn.LayerNorm(input_dim)
        self.image_projector = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, head_num, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_dim, eps=1.0e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def _group(self, patches: torch.Tensor) -> torch.Tensor:
        token_num, dim = patches.shape[-2:]
        if token_num <= 0 or token_num % self.patch_token_num:
            raise ValueError("learned_query_merge input must contain complete per-view patch grids")
        views = token_num // self.patch_token_num
        rows, columns = self.grid // self.merge_height, self.grid // self.merge_width
        grouped = patches.reshape(
            *patches.shape[:-2], views, rows, self.merge_height,
            columns, self.merge_width, dim,
        ).transpose(-4, -3)
        return grouped.reshape(
            *patches.shape[:-2], views * self.output_token_num,
            self.merge_area, dim,
        )

    def _split_cls(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        per_view = self.patch_token_num + 1
        if tokens.shape[-2] <= 0 or tokens.shape[-2] % per_view:
            raise ValueError("CLS image input must contain complete per-view patches plus CLS")
        views = tokens.shape[-2] // per_view
        grouped = tokens.reshape(*tokens.shape[:-2], views, per_view, tokens.shape[-1])
        patches = grouped[..., :-1, :].reshape(*tokens.shape[:-2], views * self.patch_token_num, tokens.shape[-1])
        return patches, grouped[..., -1:, :]

    def merge_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if self.cls_token:
            patches, cls = self._split_cls(mask.bool().unsqueeze(-1))
            grouped = self._group(patches).squeeze(-1).all(dim=-1)
            grouped = grouped.reshape(*mask.shape[:-1], cls.shape[-3], self.output_token_num)
            return torch.cat([grouped, cls.squeeze(-1)], dim=-1).flatten(-2)
        return self._group(mask.bool().unsqueeze(-1)).squeeze(-1).all(dim=-1)

    def _pool_images(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim < 3 or patches.shape[-1] != self.input_dim:
            raise ValueError(f"learned_query_merge expects [..., tokens, {self.input_dim}]")
        regions = self._group(self.image_projector(self.image_norm(patches)))
        region_num = regions.shape[-3]
        views = region_num // self.output_token_num
        queries = self.query_tokens.repeat(views, 1).expand(
            *regions.shape[:-3], region_num, self.hidden_dim,
        ).reshape(-1, 1, self.hidden_dim)
        region_patches = regions.reshape(-1, self.merge_area, self.hidden_dim)
        pooled = self.attention(queries, region_patches, region_patches)[0]
        output = pooled + self.mlp(self.layernorm(pooled))
        return output.reshape(*patches.shape[:-2], region_num, self.hidden_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if not self.cls_token:
            return self._pool_images(patches)
        patches, cls = self._split_cls(patches)
        pooled = self._pool_images(patches)
        views = cls.shape[-3]
        pooled = pooled.reshape(*pooled.shape[:-2], views, self.output_token_num, self.hidden_dim)
        # CLS bypasses local attention and the pooling MLP; adapt its width
        # with the same per-token image adapter as the spatial features.
        cls = self.image_projector(self.image_norm(cls))
        return torch.cat([pooled, cls], dim=-2).flatten(-3, -2)
