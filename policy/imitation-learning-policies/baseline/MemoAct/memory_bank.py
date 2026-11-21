from collections.abc import Callable

import torch
from torch import nn

from baseline.Fusion.memory_bank import merge_most_similar_adjacent_slots
from imitation_learning.models.memory.gate_memory import GateMemoryBank, MemorySlot


class MemoActCompressor(nn.Module):
    source_count = 3

    def __init__(self, token_num: int, hidden_dim: int, head_num: int):
        super().__init__()
        self.summary_tokens = nn.Parameter(torch.empty(token_num, hidden_dim))
        self.time_embedding = nn.Parameter(torch.empty(self.source_count, 1, hidden_dim))
        nn.init.normal_(self.summary_tokens, std=0.02)
        nn.init.normal_(self.time_embedding, std=0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                hidden_dim, head_num, dim_feedforward=4 * hidden_dim,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(2)
        ])
        # Each slot can attend within itself and to earlier slots. The summary
        # occupies the last slot so it can read all three source observations.
        slot_positions = torch.arange(self.source_count + 1).repeat_interleave(token_num)
        self.register_buffer(
            "causal_mask", slot_positions[None, :] > slot_positions[:, None],
            persistent=False,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        sources = observations + self.time_embedding.to(observations)
        summary = self.summary_tokens.to(observations)
        sequence = torch.cat((sources.flatten(0, 1), summary), dim=0).unsqueeze(0)
        for layer in self.layers:
            sequence = layer(sequence, src_mask=self.causal_mask)
        return sequence[0, -summary.shape[0]:]


class MemoActMemoryBank(GateMemoryBank):
    def __init__(
        self, short_term_capacity: int, long_term_capacity: int,
        compressor: MemoActCompressor,
    ):
        if type(short_term_capacity) is not int or short_term_capacity < compressor.source_count:
            raise ValueError("MemoAct short-term capacity must be an integer of at least 3")
        if type(long_term_capacity) is not int or long_term_capacity < 1:
            raise ValueError("MemoAct long-term capacity must be a positive integer")
        super().__init__(short_term_capacity + long_term_capacity, keep_first=False, keep_recent=0)
        self.short_term_capacity = short_term_capacity
        self.long_term_capacity = long_term_capacity
        self.compressor = compressor
        self.long_term_count = 0
        self.pending_sources: list[MemorySlot] = []

    def append(
        self, latent: torch.Tensor, frame_index: int, gate_scores: None = None,
        raw_features: torch.Tensor | None = None,
        raw_state: dict[str, torch.Tensor] | None = None,
        *, record_eviction_scores: bool = False,
    ) -> None:
        self.prepare_read()
        if len(self.slots) - self.long_term_count == self.short_term_capacity:
            start = self.long_term_count
            end = start + self.compressor.source_count
            self.pending_sources = self.slots[start:end]
            del self.slots[start:end]
        self.slots.append(MemorySlot(
            self.next_slot_id, int(frame_index), latent, raw_features, raw_state=raw_state,
        ))
        self.next_slot_id += 1
        self.next_anchor_index += 1

    def prepare_read(self) -> None:
        if not self.pending_sources:
            return
        # Materialize at the first read so gradients reach the compressor even
        # when the write was at the end of a previous streaming chunk.
        source = self.pending_sources[0]
        latent = self.compressor(torch.stack([slot.latent for slot in self.pending_sources]))
        long_term = self.slots[:self.long_term_count]
        long_term.append(MemorySlot(source.slot_id, source.frame_index, latent))
        if len(long_term) > self.long_term_capacity:
            merge_most_similar_adjacent_slots(long_term)
        self.slots[:self.long_term_count] = long_term
        self.long_term_count = len(long_term)
        self.pending_sources = []

    def detach(self):
        super().detach()
        for slot in self.pending_sources:
            slot.latent = slot.latent.detach()
            if slot.raw_features is not None:
                slot.raw_features = slot.raw_features.detach()
            if slot.raw_state is not None:
                slot.raw_state = {name: value.detach() for name, value in slot.raw_state.items()}

    def reproject(self, projector: Callable[..., torch.Tensor]):
        observations = self.slots[self.long_term_count:] + self.pending_sources
        self._reproject_slots(
            [slot for slot in observations if slot.raw_features is not None], projector,
        )
