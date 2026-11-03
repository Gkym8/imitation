"""Independent expert-trained dynamics model; inputs are in physical units."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import nn

from imitation_learning.models.future_prediction.future_feature_predictor import (
    FutureFeaturePredictor,
)


CHECKPOINT_VERSION = 1


class StateActionFuturePredictor(nn.Module):
    """Predict ordered future camera patches from the current state and actions.

    No history or policy latent is accepted. ``actions`` and ``proprio`` are
    physical (unnormalized) values. Normalization stays in the autograd graph,
    including when a loaded, frozen model is used to score generated actions.
    Camera patches are concatenated in ``camera_names`` order, without CLS.
    """

    def __init__(
        self,
        feature_dim: int,
        patch_token_num: int,
        camera_names: list[str],
        proprio_dim: int,
        action_dim: int,
        execution_chunk_length: int = 10,
        hidden_dim: int = 512,
        head_num: int = 8,
        layer_num: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        feedforward_dim: int | None = None,
        activation: str = "gelu",
        norm_first: bool = True,
        final_norm: bool = True,
    ) -> None:
        super().__init__()
        if min(feature_dim, patch_token_num, proprio_dim, action_dim,
               execution_chunk_length) <= 0:
            raise ValueError("Feature, patch, state, action and horizon sizes must be positive")
        if not camera_names or len(set(camera_names)) != len(camera_names):
            raise ValueError("camera_names must be nonempty and unique")
        self.model_config = dict(
            feature_dim=feature_dim, patch_token_num=patch_token_num,
            camera_names=list(camera_names), proprio_dim=proprio_dim,
            action_dim=action_dim, execution_chunk_length=execution_chunk_length,
            hidden_dim=hidden_dim, head_num=head_num, layer_num=layer_num,
            mlp_ratio=mlp_ratio, dropout=dropout,
            feedforward_dim=feedforward_dim, activation=activation,
            norm_first=norm_first, final_norm=final_norm,
        )
        self.token_num = patch_token_num * len(camera_names)
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.execution_chunk_length = execution_chunk_length
        self.proprio_projector = nn.Linear(proprio_dim, feature_dim)
        self.proprio_type = nn.Parameter(torch.empty(1, 1, feature_dim))
        self.image_position = nn.Parameter(torch.empty(1, self.token_num, feature_dim))
        nn.init.normal_(self.proprio_type, std=0.02)
        nn.init.normal_(self.image_position, std=0.02)
        self.predictor = FutureFeaturePredictor(
            hidden_dim=feature_dim, predictor_hidden_dim=hidden_dim,
            output_feature_dim=feature_dim, action_dim=action_dim,
            head_num=head_num, layer_num=layer_num, mlp_ratio=mlp_ratio,
            dropout=dropout, future_token_num=self.token_num,
            max_action_length=execution_chunk_length, use_action_condition=True,
            feedforward_dim=feedforward_dim, activation=activation,
            norm_first=norm_first, final_norm=final_norm,
        )
        for field, dim in (("action", action_dim), ("proprio", proprio_dim)):
            self.register_buffer(f"{field}_offset", torch.zeros(dim))
            self.register_buffer(f"{field}_scale", torch.ones(dim))

    @torch.no_grad()
    def set_normalizer(self, statistics: dict[str, list[float]]) -> None:
        expected = {f"{field}_{stat}" for field in ("action", "proprio")
                    for stat in ("offset", "scale")}
        if set(statistics) != expected:
            raise ValueError(f"Normalizer must contain exactly {sorted(expected)}")
        for name, values in statistics.items():
            buffer = getattr(self, name)
            value = torch.as_tensor(values, dtype=buffer.dtype, device=buffer.device)
            if value.shape != buffer.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid normalizer field {name}")
            if name.endswith("scale") and not (value > 0).all():
                raise ValueError(f"Normalizer scale must be positive: {name}")
            buffer.copy_(value)

    def normalizer_dict(self) -> dict[str, list[float]]:
        return {name: getattr(self, name).detach().cpu().tolist()
                for name in ("action_offset", "action_scale", "proprio_offset", "proprio_scale")}

    def forward(
        self, current_features: torch.Tensor, proprio: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        batch = current_features.shape[0]
        if current_features.shape != (batch, self.token_num, self.feature_dim):
            raise ValueError("current_features must be [B, cameras * patches, feature_dim]")
        if proprio.shape != (batch, self.proprio_dim):
            raise ValueError(f"proprio must be [B, {self.proprio_dim}]")
        if actions.shape != (batch, self.execution_chunk_length, self.action_dim):
            raise ValueError("actions must contain exactly the configured execution window")
        normalized_proprio = (proprio - self.proprio_offset) / self.proprio_scale
        normalized_actions = (actions - self.action_offset) / self.action_scale
        proprio_token = self.proprio_projector(normalized_proprio).unsqueeze(1)
        context = torch.cat((current_features + self.image_position,
                             proprio_token + self.proprio_type), dim=1)
        return self.predictor(current_observation_tokens=context,
                              action_chunk=normalized_actions)


def save_future_predictor_checkpoint(
    path: str | Path, model: StateActionFuturePredictor,
    metadata: dict[str, Any], training_state: dict[str, Any] | None = None,
) -> None:
    """Atomically save the independent model, its contract and optional trainer state."""
    payload = dict(
        format_version=CHECKPOINT_VERSION, model_config=model.model_config,
        model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
        normalizer=model.normalizer_dict(), metadata=metadata,
        training_state=training_state or {},
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary_name = handle.name
            torch.save(payload, handle)
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_future_predictor_checkpoint(
    path: str | Path, device: str | torch.device = "cpu", freeze: bool = True,
    expected_feature_contract: dict[str, Any] | None = None,
    expected_control_interface: str | None = None,
) -> tuple[StateActionFuturePredictor, dict[str, Any]]:
    """Load without disabling input gradients; return model and full checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported future predictor checkpoint format")
    metadata = checkpoint["metadata"]
    if expected_feature_contract is not None:
        if metadata.get("feature_contract") != expected_feature_contract:
            raise ValueError("Future predictor feature contract does not match")
    if expected_control_interface is not None:
        if metadata.get("control_interface") != expected_control_interface:
            raise ValueError("Future predictor control interface does not match")
    model = StateActionFuturePredictor(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    # Validate metadata and positive scales as well as the tensor state itself.
    saved_normalizer = model.normalizer_dict()
    if checkpoint["normalizer"] != saved_normalizer:
        raise ValueError("Normalizer metadata disagrees with checkpoint tensors")
    model.set_normalizer(saved_normalizer)
    contract = metadata["feature_contract"]
    for key in ("feature_dim", "patch_token_num", "camera_names"):
        if contract[key] != model.model_config[key]:
            raise ValueError(f"Checkpoint feature contract disagrees with model: {key}")
    model.to(device)
    model.eval()
    if freeze:
        model.requires_grad_(False)
    return model, checkpoint
