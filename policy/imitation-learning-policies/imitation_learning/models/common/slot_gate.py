import math

import torch
from torch import nn
from torch.nn import functional as F


class ObservationConditionedSlotGate(nn.Module):
    """Score each history slot per head using the current observation.

    Only the scoring summaries are pooled. The caller retains the complete
    visual tokens for attention. All slots share the same MLP, and the output
    is a log gate with shape [batch, slot, head].
    """

    def __init__(
        self,
        feature_dim: int,
        head_num: int,
        hidden_dim: int = 128,
        bias_init: float = 0.0,
    ):
        super().__init__()
        if feature_dim <= 0 or head_num <= 0 or hidden_dim <= 0:
            raise ValueError("slot gate dimensions must be positive")
        if not math.isfinite(bias_init):
            raise ValueError("slot gate bias_init must be finite")
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.bias_init = float(bias_init)
        self.slot_norm = nn.LayerNorm(feature_dim)
        self.observation_norm = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, head_num),
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Small random weights avoid an exactly constant gate while keeping
        # initial scores near sigmoid(bias_init), away from saturation.
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.mlp[-1].bias, self.bias_init)

    def forward(
        self,
        history_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        current_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history_tokens.ndim != 4 or history_tokens.shape[-1] != self.feature_dim:
            raise ValueError(
                "history_tokens must have shape [batch, slot, token, feature_dim]"
            )
        batch_size, slot_num, token_num, _ = history_tokens.shape
        if (
            current_tokens.ndim != 3
            or current_tokens.shape[0] != batch_size
            or current_tokens.shape[-1] != self.feature_dim
        ):
            raise ValueError(
                "current_tokens must have shape [batch, token, feature_dim]"
            )
        if token_num == 0 or current_tokens.shape[1] == 0:
            raise ValueError("slot gate requires non-empty token groups")
        if history_mask is not None:
            if history_mask.shape != (batch_size, slot_num):
                raise ValueError("history_mask must match [batch, slot]")
            history_mask = history_mask.to(device=history_tokens.device, dtype=torch.bool)
            history_tokens = history_tokens.masked_fill(
                ~history_mask[:, :, None, None], 0
            )
        slot_summary = self.slot_norm(history_tokens.mean(dim=-2))

        if current_mask is None:
            current_summary = current_tokens.mean(dim=-2)
        else:
            if current_mask.shape != current_tokens.shape[:2]:
                raise ValueError("current_mask must match [batch, token]")
            current_mask = current_mask.to(device=current_tokens.device, dtype=torch.bool)
            current_summary = current_tokens.masked_fill(
                ~current_mask[:, :, None], 0
            ).sum(dim=-2) / current_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        current_summary = self.observation_norm(current_summary)
        current_summary = current_summary[:, None, :].expand(-1, slot_num, -1)
        logits = self.mlp(torch.cat([slot_summary, current_summary], dim=-1))
        # Float32 log-sigmoid remains stable with mixed precision and very
        # small gates. Do not clamp sigmoid outputs and cut off gradients.
        log_gate = F.logsigmoid(logits.float())
        if history_mask is not None:
            log_gate = log_gate.masked_fill(~history_mask[:, :, None], 0)
        return log_gate
