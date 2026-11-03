from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureFeaturePredictor(nn.Module):
    """Predict future observation features from context, observation and actions.

    The module deliberately does not prescribe how memory is retrieved or how the
    current observation is encoded. Both inputs may contain any number and layout
    of tokens, as long as their final dimension is ``input_feature_dim``. All
    dimensions between batch and feature are flattened into one token dimension.

    ``hidden_dim`` is kept as the input feature dimension for backward
    compatibility. ``predictor_hidden_dim`` optionally widens the internal
    Transformer without changing the decision-memory width.

    Valid input examples include ``[B, D]``, ``[B, N, D]`` and
    ``[B, slots, tokens_per_slot, D]``. This keeps the predictor independent from
    the memory implementation that will be connected later.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_feature_dim: int,
        action_dim: int,
        head_num: int = 8,
        layer_num: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        future_token_num: int = 2,
        max_action_length: int = 16,
        use_action_condition: bool = True,
        predictor_hidden_dim: int | None = None,
        feedforward_dim: int | None = None,
        activation: str = "gelu",
        norm_first: bool = True,
        final_norm: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if predictor_hidden_dim is None:
            predictor_hidden_dim = hidden_dim
        if predictor_hidden_dim <= 0:
            raise ValueError(
                "predictor_hidden_dim must be positive, got "
                f"{predictor_hidden_dim}"
            )
        if output_feature_dim <= 0:
            raise ValueError(
                f"output_feature_dim must be positive, got {output_feature_dim}"
            )
        if head_num <= 0 or predictor_hidden_dim % head_num != 0:
            raise ValueError(
                "head_num must be positive and divide predictor_hidden_dim, "
                f"got predictor_hidden_dim={predictor_hidden_dim}, "
                f"head_num={head_num}"
            )
        if layer_num <= 0:
            raise ValueError(f"layer_num must be positive, got {layer_num}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if feedforward_dim is None:
            feedforward_dim = int(predictor_hidden_dim * mlp_ratio)
        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive")
        if activation not in {"relu", "gelu"}:
            raise ValueError("activation must be relu or gelu")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if future_token_num <= 0:
            raise ValueError(
                f"future_token_num must be positive, got {future_token_num}"
            )
        if max_action_length <= 0:
            raise ValueError(
                f"max_action_length must be positive, got {max_action_length}"
            )
        if use_action_condition and action_dim <= 0:
            raise ValueError(
                "action_dim must be positive when action conditioning is enabled, "
                f"got {action_dim}"
            )

        self.input_feature_dim = hidden_dim
        self.hidden_dim = predictor_hidden_dim
        self.output_feature_dim = output_feature_dim
        self.action_dim = action_dim
        self.future_token_num = future_token_num
        self.max_action_length = max_action_length
        self.use_action_condition = use_action_condition

        if self.input_feature_dim == self.hidden_dim:
            self.context_projector = nn.Identity()
        else:
            self.context_projector = nn.Sequential(
                nn.LayerNorm(self.input_feature_dim),
                nn.Linear(self.input_feature_dim, self.hidden_dim),
            )

        self.current_observation_type_embedding = nn.Parameter(
            torch.empty(1, 1, self.hidden_dim)
        )
        self.memory_context_type_embedding = nn.Parameter(
            torch.empty(1, 1, self.hidden_dim)
        )
        self.future_tokens = nn.Parameter(
            torch.empty(1, future_token_num, self.hidden_dim)
        )

        if use_action_condition:
            self.action_type_embedding: nn.Parameter | None = nn.Parameter(
                torch.empty(1, 1, self.hidden_dim)
            )
            self.action_projector: nn.Module | None = nn.Linear(
                action_dim, self.hidden_dim
            )
            self.action_position_embedding: nn.Parameter | None = nn.Parameter(
                torch.empty(1, max_action_length, self.hidden_dim)
            )
        else:
            self.register_parameter("action_type_embedding", None)
            self.action_projector = None
            self.register_parameter("action_position_embedding", None)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=head_num,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=layer_num,
            norm=nn.LayerNorm(self.hidden_dim) if final_norm else None,
            enable_nested_tensor=False,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, output_feature_dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.current_observation_type_embedding, std=0.02)
        nn.init.normal_(self.memory_context_type_embedding, std=0.02)
        nn.init.normal_(self.future_tokens, std=0.02)
        if self.action_type_embedding is not None:
            nn.init.normal_(self.action_type_embedding, std=0.02)
        if self.action_position_embedding is not None:
            nn.init.normal_(self.action_position_embedding, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _flatten_tokens(
        self,
        tokens: torch.Tensor,
        name: str,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if tokens.ndim < 2:
            raise ValueError(
                f"{name} must have at least batch and feature dimensions, "
                f"got shape {tuple(tokens.shape)}"
            )
        if tokens.shape[-1] != self.input_feature_dim:
            raise ValueError(
                f"{name}'s last dimension must be input_feature_dim="
                f"{self.input_feature_dim}, "
                f"got shape {tuple(tokens.shape)}"
            )
        if batch_size is not None and tokens.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch size must be {batch_size}, got {tokens.shape[0]}"
            )
        return tokens.reshape(tokens.shape[0], -1, self.input_feature_dim)

    @staticmethod
    def _flatten_valid_mask(
        mask: torch.Tensor | None,
        original_tokens: torch.Tensor,
        flattened_tokens: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        expected_shape = original_tokens.shape[:-1]
        if mask is None:
            return torch.ones(
                flattened_tokens.shape[:2],
                dtype=torch.bool,
                device=flattened_tokens.device,
            )
        if tuple(mask.shape) != tuple(expected_shape):
            raise ValueError(
                f"{name} must match token dimensions {tuple(expected_shape)}, "
                f"got {tuple(mask.shape)}"
            )
        return mask.to(device=flattened_tokens.device, dtype=torch.bool).reshape(
            flattened_tokens.shape[:2]
        )

    def _encode_actions(
        self,
        action_chunk: torch.Tensor,
        action_mask: torch.Tensor | None,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.action_projector is None
            or self.action_position_embedding is None
            or self.action_type_embedding is None
        ):
            raise RuntimeError("Action conditioning is disabled for this predictor")

        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(1)
        if action_chunk.ndim != 3:
            raise ValueError(
                "action_chunk must have shape [batch, action_length, action_dim] "
                f"or [batch, action_dim], got {tuple(action_chunk.shape)}"
            )
        if action_chunk.shape[0] != batch_size:
            raise ValueError(
                f"action_chunk batch size must be {batch_size}, "
                f"got {action_chunk.shape[0]}"
            )
        if action_chunk.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_chunk's last dimension must be action_dim={self.action_dim}, "
                f"got {action_chunk.shape[-1]}"
            )

        action_length = action_chunk.shape[1]
        if action_length > self.max_action_length:
            raise ValueError(
                f"action length {action_length} exceeds configured "
                f"max_action_length={self.max_action_length}"
            )

        action_tokens = self.action_projector(action_chunk)
        action_tokens = (
            action_tokens
            + self.action_position_embedding[:, :action_length]
            + self.action_type_embedding
        )

        if action_mask is None:
            valid_mask = torch.ones(
                (batch_size, action_length),
                dtype=torch.bool,
                device=action_chunk.device,
            )
        else:
            if tuple(action_mask.shape) != (batch_size, action_length):
                raise ValueError(
                    "action_mask must have shape "
                    f"{(batch_size, action_length)}, got {tuple(action_mask.shape)}"
                )
            valid_mask = action_mask.to(device=action_chunk.device, dtype=torch.bool)

        return action_tokens, valid_mask

    def forward(
        self,
        current_observation_tokens: torch.Tensor,
        memory_context_tokens: torch.Tensor | None = None,
        action_chunk: torch.Tensor | None = None,
        current_observation_mask: torch.Tensor | None = None,
        memory_context_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return predicted future features.

        A mask value of ``True`` means the corresponding input token is valid.
        The return shape is always
        ``[batch, future_token_num, output_feature_dim]``.
        """

        current_tokens = self._flatten_tokens(
            current_observation_tokens, "current_observation_tokens"
        )
        batch_size = current_tokens.shape[0]
        current_valid_mask = self._flatten_valid_mask(
            current_observation_mask,
            current_observation_tokens,
            current_tokens,
            "current_observation_mask",
        )
        current_tokens = self.context_projector(current_tokens)
        current_tokens = (
            current_tokens + self.current_observation_type_embedding
        )

        token_groups = [current_tokens]
        valid_mask_groups = [current_valid_mask]

        if memory_context_tokens is not None:
            memory_tokens = self._flatten_tokens(
                memory_context_tokens,
                "memory_context_tokens",
                batch_size=batch_size,
            )
            memory_valid_mask = self._flatten_valid_mask(
                memory_context_mask,
                memory_context_tokens,
                memory_tokens,
                "memory_context_mask",
            )
            memory_tokens = self.context_projector(memory_tokens)
            token_groups.append(
                memory_tokens + self.memory_context_type_embedding
            )
            valid_mask_groups.append(memory_valid_mask)
        elif memory_context_mask is not None:
            raise ValueError(
                "memory_context_mask was provided without memory_context_tokens"
            )

        if self.use_action_condition:
            if action_chunk is None:
                raise ValueError(
                    "action_chunk is required when action conditioning is enabled"
                )
            action_tokens, action_valid_mask = self._encode_actions(
                action_chunk, action_mask, batch_size
            )
            token_groups.append(action_tokens)
            valid_mask_groups.append(action_valid_mask)
        else:
            if action_chunk is not None or action_mask is not None:
                raise ValueError(
                    "action input was provided while action conditioning is disabled"
                )

        future_tokens = self.future_tokens.expand(batch_size, -1, -1)
        future_valid_mask = torch.ones(
            (batch_size, self.future_token_num),
            dtype=torch.bool,
            device=future_tokens.device,
        )
        token_groups.append(future_tokens)
        valid_mask_groups.append(future_valid_mask)

        tokens = torch.cat(token_groups, dim=1)
        valid_mask = torch.cat(valid_mask_groups, dim=1)
        encoded_tokens = self.transformer(
            tokens,
            src_key_padding_mask=~valid_mask,
        )
        encoded_future_tokens = encoded_tokens[:, -self.future_token_num :]
        return self.output_head(encoded_future_tokens)

    @staticmethod
    def mse_prediction_loss(
        predicted_features: torch.Tensor,
        target_features: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        detach_target: bool = True,
    ) -> torch.Tensor:
        """Mean squared error on raw features, averaged over valid tokens/features.

        ``target_mask`` matches all dimensions except feature width. Select valid
        tokens before computing errors so padding cannot contaminate gradients.
        Accumulate in float32; no valid targets gives a differentiable zero.
        """
        if predicted_features.shape != target_features.shape:
            raise ValueError("predicted_features and target_features must have the same shape")
        if predicted_features.ndim < 2:
            raise ValueError("future features must have at least batch and feature dimensions")
        if detach_target:
            target_features = target_features.detach()
        if target_mask is not None:
            expected_shape = predicted_features.shape[:-1]
            if tuple(target_mask.shape) != tuple(expected_shape):
                raise ValueError(f"target_mask must have shape {tuple(expected_shape)}, got {tuple(target_mask.shape)}")
            valid = target_mask.to(device=predicted_features.device, dtype=torch.bool)
            predicted_features = predicted_features[valid]
            target_features = target_features[valid]
        if predicted_features.numel() == 0:
            return predicted_features.float().sum() * 0.0
        return F.mse_loss(predicted_features.float(), target_features.float())

    @staticmethod
    def cosine_prediction_error(
        predicted_features: torch.Tensor,
        target_features: torch.Tensor,
        detach_target: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Return cosine distance without reducing token/batch dimensions."""

        if predicted_features.shape != target_features.shape:
            raise ValueError(
                "predicted_features and target_features must have the same shape, "
                f"got {tuple(predicted_features.shape)} and "
                f"{tuple(target_features.shape)}"
            )
        if predicted_features.ndim < 2:
            raise ValueError(
                "future features must have at least batch and feature dimensions"
            )

        if detach_target:
            target_features = target_features.detach()

        predicted_features = F.normalize(predicted_features, dim=-1, eps=eps)
        target_features = F.normalize(target_features, dim=-1, eps=eps)
        return 1.0 - (predicted_features * target_features).sum(dim=-1)

    @staticmethod
    def cosine_prediction_loss(
        predicted_features: torch.Tensor,
        target_features: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        detach_target: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Cosine-distance loss for future feature supervision.

        ``target_mask`` uses ``True`` for valid future targets. When a batch has
        no valid target, a differentiable zero is returned.
        """

        cosine_distance = FutureFeaturePredictor.cosine_prediction_error(
            predicted_features=predicted_features,
            target_features=target_features,
            detach_target=detach_target,
            eps=eps,
        )

        if target_mask is None:
            return cosine_distance.mean()

        if tuple(target_mask.shape) != tuple(cosine_distance.shape):
            raise ValueError(
                f"target_mask must have shape {tuple(cosine_distance.shape)}, "
                f"got {tuple(target_mask.shape)}"
            )
        valid_mask = target_mask.to(
            device=cosine_distance.device, dtype=torch.bool
        )
        valid_count = valid_mask.sum()
        if valid_count == 0:
            return predicted_features.sum() * 0.0
        return cosine_distance.masked_select(valid_mask).mean()
