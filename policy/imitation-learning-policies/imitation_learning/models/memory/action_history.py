"""Recurrent Transformer memory for executed action chunks."""

import torch
from torch import nn


class ActionHistoryEncoder(nn.Module):
    def __init__(
        self,
        chunk_length: int,
        action_dim: int,
        hidden_dim: int,
        head_num: int,
        num_layers: int = 3,
    ):
        super().__init__()
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("action history layer count must be a positive integer")
        self.chunk_length = chunk_length
        self.initial_cls = nn.Parameter(torch.empty(1, hidden_dim))
        self.step_embedding = nn.Parameter(torch.empty(chunk_length, hidden_dim))
        nn.init.normal_(self.initial_cls, std=0.02)
        nn.init.normal_(self.step_embedding, std=0.02)

        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, head_num, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.sequence_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=head_num,
                dim_feedforward=2 * hidden_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers - 1)
        )

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return self.initial_cls.expand(batch_size, -1)

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Update the CLS from the available prefix of one executed chunk."""
        num_steps = actions.shape[1]
        action_tokens = self.action_projection(actions.to(self.action_projection.weight.dtype))
        action_tokens = action_tokens + self.step_embedding[:num_steps]
        tokens = torch.cat((state[:, None], action_tokens), dim=1)
        for layer in self.sequence_layers:
            tokens = layer(tokens)
        normalized = self.attention_norm(tokens)
        # The final layer computes only the CLS output.
        attended, _ = self.attention(
            normalized[:, :1], normalized, normalized, need_weights=False,
        )
        cls = tokens[:, 0] + attended[:, 0]
        return self.output_norm(cls + self.feedforward(cls))
