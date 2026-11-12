"""Global image MAP pooling followed by a residual read of robot state."""

from math import isqrt

import torch
from torch import nn

from imitation_learning.models.common.spatial_merge import SpatialMergeSize, spatial_merge_shape


class LearnedQueryPatchMerger(nn.Module):
    """Learned queries shared across views and time.

    With ``num_queries``, every query reads the full patch grid of one view.
    Omitting it preserves the original one-query-per-region checkpoint path.
    After adapting image features to the policy width, pooling uses the
    SigLIP MAP head structure: ``h = MHA(Q, X, X)`` followed by
    ``h + MLP(LayerNorm(h))``. Global MAP outputs then query their own
    frame's state: ``LN(softmax(Q(R) K(S)^T / sqrt(D)) V(S) + R)``.
    Legacy regional pooling keeps its original state-independent computation.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, patch_token_num: int,
        merge_size: SpatialMergeSize, head_num: int, num_queries: int | None = None,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.patch_token_num = patch_token_num
        self.num_queries = num_queries
        if num_queries is None:
            grid = isqrt(patch_token_num)
            height, width = spatial_merge_shape(merge_size, "merge_size")
            if grid * grid != patch_token_num or grid % height or grid % width:
                raise ValueError("learned_query_merge requires a square patch grid divisible by merge_size")
            self.grid = grid
            self.merge_height, self.merge_width = height, width
            self.merge_area = height * width
            self.output_token_num = patch_token_num // self.merge_area
        else:
            if isinstance(num_queries, bool) or not isinstance(num_queries, int) or num_queries <= 0:
                raise ValueError("num_queries must be a positive integer")
            self.output_token_num = num_queries
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
        if num_queries is not None:
            # Follow the state-read equation directly: one attention matrix
            # scaled by sqrt(D), no extra output projection, MLP or Q/K norm.
            self.state_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.state_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.state_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.state_norm = nn.LayerNorm(hidden_dim, eps=1.0e-6)

    def _group(self, patches: torch.Tensor) -> torch.Tensor:
        token_num, dim = patches.shape[-2:]
        if token_num <= 0 or token_num % self.patch_token_num:
            raise ValueError("learned_query_merge input must contain complete per-view patch grids")
        views = token_num // self.patch_token_num
        if self.num_queries is not None:
            return patches.reshape(*patches.shape[:-2], views, self.patch_token_num, dim)
        rows, columns = self.grid // self.merge_height, self.grid // self.merge_width
        grouped = patches.reshape(
            *patches.shape[:-2], views, rows, self.merge_height,
            columns, self.merge_width, dim,
        ).transpose(-4, -3)
        return grouped.reshape(
            *patches.shape[:-2], views * self.output_token_num,
            self.merge_area, dim,
        )

    def merge_mask(self, mask: torch.Tensor) -> torch.Tensor:
        grouped = self._group(mask.bool().unsqueeze(-1)).squeeze(-1).all(dim=-1)
        if self.num_queries is not None:
            # Encoder masks mark complete camera observations. All query
            # outputs inherit their own view's validity, as regional pooling
            # inherits the validity of every patch in its region.
            return grouped.unsqueeze(-1).expand(*grouped.shape, self.num_queries).flatten(-2)
        return grouped

    def _pool_images(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim < 3 or patches.shape[-1] != self.input_dim:
            raise ValueError(f"learned_query_merge expects [..., tokens, {self.input_dim}]")
        regions = self._group(self.image_projector(self.image_norm(patches)))
        if self.num_queries is not None:
            views = regions.shape[-3]
            queries = self.query_tokens.expand(
                *regions.shape[:-2], self.num_queries, self.hidden_dim,
            ).reshape(-1, self.num_queries, self.hidden_dim)
            image_patches = regions.reshape(-1, self.patch_token_num, self.hidden_dim)
            pooled = self.attention(queries, image_patches, image_patches)[0]
            output = pooled + self.mlp(self.layernorm(pooled))
            return output.reshape(*patches.shape[:-2], views * self.num_queries, self.hidden_dim)
        region_num = regions.shape[-3]
        views = region_num // self.output_token_num
        queries = self.query_tokens.repeat(views, 1).expand(
            *regions.shape[:-3], region_num, self.hidden_dim,
        ).reshape(-1, 1, self.hidden_dim)
        region_patches = regions.reshape(-1, self.merge_area, self.hidden_dim)
        pooled = self.attention(queries, region_patches, region_patches)[0]
        output = pooled + self.mlp(self.layernorm(pooled))
        return output.reshape(*patches.shape[:-2], region_num, self.hidden_dim)

    def forward(
        self, patches: torch.Tensor, state_features: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual_tokens = self._pool_images(patches)
        if self.num_queries is None:
            return visual_tokens
        if state_features is None:
            raise ValueError("learned_query_merge requires each image's own state")
        if (
            state_features.ndim != visual_tokens.ndim
            or state_features.shape[:-2] != visual_tokens.shape[:-2]
            or state_features.shape[-1] != self.hidden_dim
            or state_features.shape[-2] <= 0
        ):
            raise ValueError("learned_query_merge state must match image batch dimensions and hidden width")
        if state_mask is not None:
            if state_mask.shape != state_features.shape[:-1]:
                raise ValueError("learned_query_merge state_mask must match state tokens")
            state_mask = state_mask.bool()
            state_features = state_features.masked_fill(~state_mask[..., None], 0)
        q = self.state_query(visual_tokens)
        k = self.state_key(state_features)
        v = self.state_value(state_features)
        logits = (q @ k.transpose(-2, -1)) * (self.hidden_dim ** -0.5)
        if state_mask is not None:
            visible = state_mask[..., None, :]
            logits = logits.masked_fill(~visible, -torch.inf)
            # An entirely masked state supplies zero information. Avoid the
            # all-minus-infinity softmax without a CPU/GPU synchronization.
            logits = torch.where(visible.any(dim=-1, keepdim=True), logits, 0)
        weights = logits.softmax(dim=-1)
        if state_mask is not None:
            weights = weights.masked_fill(~visible, 0)
        return self.state_norm(visual_tokens + weights @ v)
