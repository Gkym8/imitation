from dataclasses import dataclass
from typing import Callable, Literal

import torch


@dataclass
class MemorySlot:
    slot_id: int
    frame_index: int
    latent: torch.Tensor
    raw_features: torch.Tensor | None = None
    state: Literal["active", "shadow"] = "active"
    expire_at: int | None = None


class GateMemoryBank:
    """Chronological slots, with eviction scored by the current observation.

    A full Active bank demotes an existing slot before inserting the new one.
    Shadows remain readable for the next ``shadow_horizon`` valid anchors.
    The original first slot and the newest Active slots can be protected.
    ``keep_recent`` counts the incoming slot: protect max(K-1, 0) existing
    Active predecessors before demotion, then insert the new slot.
    Scores are transient outputs of the current read, never persistent weights.
    """

    def __init__(
        self, capacity: int, keep_first: bool = True, keep_recent: int = 2,
        shadow_horizon: int = 0,
    ):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("memory capacity must be a positive integer")
        if not isinstance(keep_first, bool):
            raise ValueError("keep_first must be boolean")
        if isinstance(keep_recent, bool) or not isinstance(keep_recent, int) or keep_recent < 0:
            raise ValueError("keep_recent must be a non-negative integer")
        if int(keep_first) + max(keep_recent - 1, 0) >= capacity:
            raise ValueError("memory protection must leave at least one eviction candidate")
        if isinstance(shadow_horizon, bool) or not isinstance(shadow_horizon, int) or shadow_horizon < 0:
            raise ValueError("shadow_horizon must be a non-negative integer")
        self.capacity = capacity
        self.keep_first = keep_first
        self.keep_recent = keep_recent
        self.shadow_horizon = shadow_horizon
        self.slots: list[MemorySlot] = []
        self.next_slot_id = 0
        self.next_anchor_index = 0

    def __len__(self):
        return len(self.slots)

    @property
    def active_count(self) -> int:
        return sum(slot.state == "active" for slot in self.slots)

    @property
    def shadow_count(self) -> int:
        return len(self.slots) - self.active_count

    @property
    def storage_capacity(self) -> int:
        return self.capacity + self.shadow_horizon

    def append(
        self,
        latent: torch.Tensor,
        frame_index: int,
        gate_scores: torch.Tensor,
        raw_features: torch.Tensor | None = None,
    ) -> int | None:
        """Finish one valid read, expire Shadows, demote, and append.

        Scores align with all slots from the read, before any removals. A slot
        demoted after anchor t is read as Shadow at t+1 through t+K inclusive,
        then removed after the read at t+K. K=0 deletes it immediately.
        """
        evicted_id = None
        selected = None
        if self.active_count == self.capacity:
            if gate_scores.shape != (len(self.slots),):
                raise ValueError("gate_scores must have one value per retained slot")
            if not torch.isfinite(gate_scores).all():
                raise ValueError("eviction gate scores must be finite")
            active_indices = [i for i, slot in enumerate(self.slots) if slot.state == "active"]
            # The incoming slot is always inserted after this read and counts
            # toward keep_recent. K=2 protects the newest existing Active slot
            # plus the incoming slot, rather than two old slots plus the new one.
            existing_recent_count = max(self.keep_recent - 1, 0)
            protected_recent = (
                set(active_indices[-existing_recent_count:])
                if existing_recent_count else set()
            )
            candidates = [
                i for i in active_indices
                if not (self.keep_first and self.slots[i].slot_id == 0)
                and i not in protected_recent
            ]
            if not candidates:
                raise RuntimeError("memory protection left no eviction candidate")
            candidate_scores = gate_scores.detach()[candidates]
            index = candidates[int(candidate_scores.argmin().item())]
            selected = self.slots[index]
            evicted_id = selected.slot_id
        # Expiry happens after this anchor's read; incoming lookahead/padding
        # never calls append and therefore cannot age or expire any Shadow.
        self.slots = [
            slot for slot in self.slots
            if slot.expire_at is None or slot.expire_at > self.next_anchor_index
        ]
        if selected is not None:
            if self.shadow_horizon:
                selected.state = "shadow"
                selected.expire_at = self.next_anchor_index + self.shadow_horizon
            else:
                self.slots = [slot for slot in self.slots if slot is not selected]
        self.slots.append(MemorySlot(self.next_slot_id, int(frame_index), latent, raw_features))
        self.next_slot_id += 1
        self.next_anchor_index += 1
        return evicted_id

    def detach(self):
        for slot in self.slots:
            slot.latent = slot.latent.detach()
            if slot.raw_features is not None:
                slot.raw_features = slot.raw_features.detach()

    def reproject(self, projector: Callable[[torch.Tensor], torch.Tensor]):
        """Rebuild historical latents using the current chunk's parameters."""
        if self.slots and all(slot.raw_features is not None for slot in self.slots):
            projected = projector(torch.stack([slot.raw_features for slot in self.slots]))
            for slot, latent in zip(self.slots, projected):
                slot.latent = latent


def gate_selectivity_penalty(
    log_gates: torch.Tensor,
    history_mask: torch.Tensor,
    capacity: int,
    min_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample hinge penalty, eligible only for a full, nontrivial bank."""
    if log_gates.ndim != 3 or history_mask.shape != log_gates.shape[:2]:
        raise ValueError("gate selectivity expects [batch, slot, head] and [batch, slot]")
    gates = log_gates.float().exp()
    valid = history_mask.to(device=gates.device, dtype=torch.bool)
    count = valid.sum(dim=1)
    weights = valid[:, :, None].to(gates.dtype)
    denominator = count[:, None, None].clamp_min(1)
    mean = (gates * weights).sum(dim=1, keepdim=True) / denominator
    variance = ((gates - mean).square() * weights).sum(dim=1) / count[:, None].clamp_min(1)
    safe_std = variance.clamp_min(1.0e-12).sqrt()
    std = torch.where(variance > 0, safe_std, torch.zeros_like(safe_std))
    penalty = (min_std - std).clamp_min(0).square().mean(dim=-1)
    eligible = (count == capacity) & (count >= 2)
    return penalty, eligible
