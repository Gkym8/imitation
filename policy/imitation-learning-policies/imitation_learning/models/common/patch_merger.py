"""Learnable local patch merging after a frozen visual backbone."""

from math import isqrt

import torch
from torch import nn


class SpatialPatchMerger(nn.Module):
    """Concatenate spatial neighbours, then project through a narrow MLP.

    Input tokens are row-major patches, with views/frames concatenated in
    token order. Each view is grouped independently. Backbone features can
    be detached without preventing this module from receiving gradients.
    """

    def __init__(self, input_dim: int, hidden_dim: int, patch_token_num: int, merge_size: int):
        super().__init__()
        for name, value in (("input_dim", input_dim), ("hidden_dim", hidden_dim),
                            ("patch_token_num", patch_token_num), ("merge_size", merge_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        grid = isqrt(patch_token_num)
        if grid * grid != patch_token_num or grid % merge_size:
            raise ValueError("patch_merge requires a square patch grid divisible by merge_size")
        self.input_dim = input_dim
        self.patch_token_num = patch_token_num
        self.grid = grid
        self.merge_size = merge_size
        self.output_token_num = patch_token_num // merge_size**2
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * merge_size**2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _group(self, patches: torch.Tensor) -> torch.Tensor:
        token_num, dim = patches.shape[-2:]
        if token_num % self.patch_token_num:
            raise ValueError("patch_merge input must contain complete per-view patch grids")
        views = token_num // self.patch_token_num
        side = self.grid // self.merge_size
        grouped = patches.reshape(
            *patches.shape[:-2], views, side, self.merge_size, side, self.merge_size, dim
        ).transpose(-4, -3)
        return grouped.reshape(
            *patches.shape[:-2], views * side**2, self.merge_size**2 * dim
        )

    def merge_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return self._group(mask.bool().unsqueeze(-1)).all(dim=-1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim < 2 or patches.shape[-1] != self.input_dim:
            raise ValueError(f"patch_merge expects [..., tokens, {self.input_dim}]")
        return self.mlp(self._group(self.norm(patches)))
