"""Gate eviction with cached redundancy between adjacent Active memories."""

import math
from numbers import Real
from typing import Callable

import torch
import torch.nn.functional as F

from imitation_learning.models.memory.gate_memory import GateMemoryBank


@torch.no_grad()
def token_mean_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Average corresponding-token cosines, without pooling token features."""
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError("similarity expects matching [..., token, feature] shapes")
    with torch.autocast(device_type=left.device.type, enabled=False):
        return F.cosine_similarity(left.float(), right.float(), dim=-1, eps=1e-8).mean(dim=-1)


class RedundancyGateMemoryBank(GateMemoryBank):
    """Replace only eviction scores; inherit protection and Shadow lifetime.

    Similarity uses stored visual slot latents, including every corresponding
    token. Cache entries are shared by both endpoints, contain no gradients,
    and survive changes in gates but not historical feature reprojection.
    """

    def __init__(
        self, capacity: int, keep_first: bool = True, keep_recent: int = 2,
        shadow_horizon: int = 0, redundancy_alpha: float = 0.0,
    ):
        super().__init__(capacity, keep_first, keep_recent, shadow_horizon)
        if (
            isinstance(redundancy_alpha, bool)
            or not isinstance(redundancy_alpha, Real)
            or not math.isfinite(redundancy_alpha)
            or redundancy_alpha < 0
        ):
            raise ValueError("redundancy_alpha must be a finite non-negative number")
        self.redundancy_alpha = float(redundancy_alpha)
        self._similarity_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _active_pairs(self):
        active = [slot for slot in self.slots if slot.state == "active"]
        return list(zip(active, active[1:]))

    def _prune_similarity_cache(self):
        adjacent_ids = {(left.slot_id, right.slot_id) for left, right in self._active_pairs()}
        self._similarity_cache = {
            key: similarity for key, similarity in self._similarity_cache.items()
            if key in adjacent_ids
        }

    @torch.no_grad()
    def _eviction_scores(self, gate_scores: torch.Tensor) -> torch.Tensor:
        if gate_scores.shape != (len(self.slots),):
            raise ValueError("gate_scores must have one value per retained slot")
        if not torch.isfinite(gate_scores).all():
            raise ValueError("eviction gate scores must be finite")
        active_indices = [i for i, slot in enumerate(self.slots) if slot.state == "active"]
        pairs = self._active_pairs()
        missing = [
            (left, right) for left, right in pairs
            if (left.slot_id, right.slot_id) not in self._similarity_cache
        ]
        # Batch all new edges once. Both endpoints reuse the same scalar.
        if missing:
            similarities = token_mean_cosine(
                torch.stack([left.latent.detach() for left, _ in missing]),
                torch.stack([right.latent.detach() for _, right in missing]),
            )
            for (left, right), similarity in zip(missing, similarities.unbind()):
                self._similarity_cache[left.slot_id, right.slot_id] = similarity

        scores = gate_scores.detach().float().clone()
        gates = scores[active_indices]
        redundancy = torch.zeros_like(gates)
        neighbor_count = torch.zeros_like(gates)
        if pairs:
            similarity = torch.stack([
                self._similarity_cache[left.slot_id, right.slot_id]
                for left, right in pairs
            ])
            redundancy[:-1] += similarity * gates[1:]
            redundancy[1:] += similarity * gates[:-1]
            neighbor_count[:-1] += 1
            neighbor_count[1:] += 1
        # With first/recent protection every eligible slot has two neighbors.
        # Actual-neighbor averaging also supports configs that unprotect ends.
        redundancy /= neighbor_count.clamp_min(1)
        scores[active_indices] = gates + self.redundancy_alpha * (1 - redundancy)
        return scores

    def append(
        self, latent: torch.Tensor, frame_index: int, gate_scores: torch.Tensor,
        raw_features: torch.Tensor | None = None,
        raw_state: dict[str, torch.Tensor] | None = None,
    ) -> int | None:
        # Do not decorate append with no_grad: the stored latent must retain
        # its task-loss gradient path inside the current training chunk.
        if self.active_count == self.capacity and self.redundancy_alpha > 0:
            gate_scores = self._eviction_scores(gate_scores)
        evicted_id = super().append(latent, frame_index, gate_scores, raw_features, raw_state)
        self._prune_similarity_cache()
        return evicted_id

    def reproject(self, projector: Callable[..., torch.Tensor]):
        super().reproject(projector)
        # Reprojection may change every latent, even when slot IDs are stable.
        self._similarity_cache.clear()
