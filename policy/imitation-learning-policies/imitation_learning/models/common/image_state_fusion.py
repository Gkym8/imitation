"""Residual image-to-state attention restored from f682be8."""

import torch
from torch import nn


class ImageStateFusion(nn.Module):
    """Read each frame's own state after spatial mean pooling/projection."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.state_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.state_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.state_norm = nn.LayerNorm(hidden_dim, eps=1.0e-6)

    def forward(self, visual_tokens, state_features, state_mask=None):
        if state_features is None:
            raise ValueError("image/state fusion requires each image's own state")
        if (
            state_features.ndim != visual_tokens.ndim
            or state_features.shape[:-2] != visual_tokens.shape[:-2]
            or state_features.shape[-1] != self.hidden_dim
            or state_features.shape[-2] <= 0
        ):
            raise ValueError("image/state fusion state must match image batch dimensions and hidden width")
        if state_mask is not None:
            if state_mask.shape != state_features.shape[:-1]:
                raise ValueError("image/state fusion state_mask must match state tokens")
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
