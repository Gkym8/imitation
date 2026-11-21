"""Consolidate a chronological visual memory bank by merging adjacent slots."""

from typing import Callable

import torch

from imitation_learning.models.memory.gate_memory import GateMemoryBank, MemorySlot
from imitation_learning.models.memory.redundancy_eviction import token_mean_cosine


def merge_most_similar_adjacent_slots(slots: list[MemorySlot]) -> None:
    with torch.no_grad():
        latents = torch.stack([slot.latent for slot in slots])
        pair = int(token_mean_cosine(latents[:-1], latents[1:]).argmax().item())

    left, right = slots[pair:pair + 2]
    # Preserve the task gradient inside the current training chunk.
    slots[pair:pair + 2] = [MemorySlot(
        left.slot_id, left.frame_index, 0.5 * (left.latent + right.latent),
    )]


class FusionMemoryBank(GateMemoryBank):
    def __init__(self, capacity: int):
        super().__init__(capacity, keep_first=False, keep_recent=0, shadow_horizon=0)

    def append(
        self, latent: torch.Tensor, frame_index: int, gate_scores: None = None,
        raw_features: torch.Tensor | None = None,
        raw_state: dict[str, torch.Tensor] | None = None,
        *, record_eviction_scores: bool = False,
    ) -> None:
        self.slots.append(MemorySlot(
            self.next_slot_id, int(frame_index), latent, raw_features, raw_state=raw_state,
        ))
        self.next_slot_id += 1
        self.next_anchor_index += 1
        if len(self.slots) <= self.capacity:
            return
        merge_most_similar_adjacent_slots(self.slots)

    def reproject(self, projector: Callable[..., torch.Tensor]):
        # A merged slot stores only its fused latent, with no source frame to re-encode.
        self._reproject_slots(
            [slot for slot in self.slots if slot.raw_features is not None], projector,
        )
