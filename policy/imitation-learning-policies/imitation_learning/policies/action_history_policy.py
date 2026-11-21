"""Opt-in recurrent action history for the memory diffusion policy."""

from functools import partial

import torch
from torch import nn

from imitation_learning.models.denoising_networks.memory_transformer import MemoryTransformer
from imitation_learning.models.memory.action_history import ActionHistoryEncoder
from imitation_learning.policies.history_denoising_policy import HistoryDenoisingPolicy


class ActionHistoryMemoryTransformer(MemoryTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_history_type_embedding = nn.Parameter(torch.empty(self.hidden_dim))
        nn.init.normal_(self.action_history_type_embedding, std=0.02)
        for block in self.blocks:
            block.input_token_num += 1

    def _insert_action_history(self, inputs, history):
        x = inputs["x"]
        history = (history + self.action_history_type_embedding).to(x)
        start = x.shape[1] - self.action_token_num
        # Insert after positional encoding to keep the original token positions.
        inputs["x"] = torch.cat((x[:, :start], history[:, None], x[:, start:]), dim=1)

    def _project_to_latent_space(self, data_dict):
        history = data_dict.pop("action_history_cls", None)
        inputs = super()._project_to_latent_space(data_dict)
        if history is not None:
            self._insert_action_history(inputs, history)
        return inputs

    def prepare_parallel_forward(self, data_dict):
        history = data_dict["action_history_cls"].flatten(0, 1)
        inputs = super().prepare_parallel_forward(data_dict)
        self._insert_action_history(inputs, history)
        return inputs


class ActionHistoryPolicy(HistoryDenoisingPolicy):
    def __init__(
        self, *, action_history: bool = False, action_history_num_layers: int = 1, **kwargs,
    ):
        if not isinstance(action_history, bool):
            raise ValueError("action_history must be boolean")
        if action_history:
            memory_bank_type = kwargs.get("memory_bank_type", "gate")
            if memory_bank_type != "gate":
                raise ValueError(f"{memory_bank_type} requires action_history=False")
            factory = kwargs["denoising_network_partial"]
            if factory.func is not MemoryTransformer:
                raise ValueError("action_history requires MemoryTransformer")
            kwargs["denoising_network_partial"] = partial(
                ActionHistoryMemoryTransformer, *factory.args, **factory.keywords,
            )
        super().__init__(**kwargs)
        self.action_history = action_history
        if action_history:
            if self.skip_memory:
                raise ValueError("action_history requires the history policy")
            # Configs stored by old checkpoints omit the layer count and use one layer.
            self.action_history_encoder = ActionHistoryEncoder(
                self.execution_chunk_length, self.action_decoder.latent_dim,
                self.denoising_network.hidden_dim, self.denoising_network.head_num,
                num_layers=action_history_num_layers,
            )
        self._training_action_history = None
        self._online_action_history = {}
        self._pending_actions = {}

    def _training_history_tokens(self, batch, supervised, previous):
        names = self.action_decoder.data_entry_names
        rows, anchors = batch[names[0]].shape[:2]
        if not 0 < supervised <= anchors:
            raise ValueError("invalid supervised anchor count")
        frames = batch["traj_idx"].reshape(rows, anchors)
        valid = ~batch["entire_traj_is_padding"].bool()
        with torch.no_grad():
            chunks = self.action_decoder.encode({name: batch[name].flatten(0, 1) for name in names})
            chunks = chunks.unflatten(0, (rows, anchors))[:, :, :self.execution_chunk_length]
        encoder = self.action_history_encoder
        if previous is None:
            state = encoder.initial_state(rows)
            pending = torch.zeros_like(chunks[:, 0])
            pending_end = torch.full_like(frames[:, 0], -1)
        else:
            state, pending, pending_end = previous

        tokens = []
        for anchor in range(supervised):
            use = valid[:, anchor]
            confirmed = use & (frames[:, anchor] == pending_end)
            if previous is not None or anchor > 0:
                updated = encoder(state, pending)
                state = torch.where(confirmed[:, None], updated, state)
            tokens.append(state)
            pending = torch.where(use[:, None, None], chunks[:, anchor], pending)
            pending_end = torch.where(use, frames[:, anchor] + self.execution_chunk_length, pending_end)
        # Only supervised anchors may advance the state or stage a chunk.
        tokens.extend([state] * (anchors - supervised))
        return torch.stack(tokens, dim=1), (state.detach(), pending, pending_end)

    def compute_loss(self, normalized_batch):
        if not self.action_history:
            return super().compute_loss(normalized_batch)
        streaming = "_streaming_supervised_length" in normalized_batch
        supervised = (
            int(normalized_batch["_streaming_supervised_length"]) if streaming
            else normalized_batch["entire_traj_is_padding"].shape[1]
        )
        reset = not streaming or bool(normalized_batch["_streaming_reset"])
        final = not streaming or bool(normalized_batch["_streaming_final"])
        previous = None if reset else self._training_action_history
        if not reset and previous is None:
            raise RuntimeError("missing streaming action history")
        tokens, next_state = self._training_history_tokens(normalized_batch, supervised, previous)
        loss = super().compute_loss({**normalized_batch, "action_history_cls": tokens})
        self._training_action_history = None if final else next_state
        return loss

    def _encode_input_multi_traj(self, normalized_batch):
        inputs, targets = super()._encode_input_multi_traj(normalized_batch)
        if self.action_history:
            inputs["action_history_cls"] = normalized_batch["action_history_cls"]
        return inputs, targets

    def _encode_input_add_noise(self, normalized_batch, mode):
        inputs, targets = super()._encode_input_add_noise(normalized_batch, mode)
        if self.action_history:
            initial = self.action_history_encoder.initial_state(1)[0]
            episodes = normalized_batch["episode_idx"].reshape(-1).tolist()
            inputs["action_history_cls"] = torch.stack([
                self._online_action_history.get(episode, initial) for episode in episodes
            ])
        return inputs, targets

    def predict_single_traj(self, normalized_batch):
        output = super().predict_single_traj(normalized_batch)
        if self.action_history:
            episodes = normalized_batch["episode_idx"].reshape(-1).tolist()
            padding = normalized_batch.get("entire_traj_is_padding")
            valid = [True] * len(episodes) if padding is None else (~padding.bool().reshape(-1)).tolist()
            chunks = self.action_decoder.encode(output)[:, :self.execution_chunk_length]
            for episode, chunk, use in zip(episodes, chunks, valid):
                if use:
                    self._pending_actions[episode] = chunk.detach().clone()
        return output

    @torch.inference_mode()
    def record_executed_actions(self, episode: int, executed_steps: int):
        if not 0 <= executed_steps <= self.execution_chunk_length:
            raise ValueError(f"invalid executed action count: {executed_steps}")
        chunk = self._pending_actions.get(episode)
        if chunk is None:
            raise RuntimeError("no pending action chunk to confirm")
        if executed_steps:
            encoder = self.action_history_encoder
            state = self._online_action_history.get(episode, encoder.initial_state(1)[0])
            self._online_action_history[episode] = encoder(
                state[None], chunk[None, :executed_steps],
            )[0]
        del self._pending_actions[episode]

    def reset(self):
        super().reset()
        self._training_action_history = None
        self._online_action_history = {}
        self._pending_actions = {}
