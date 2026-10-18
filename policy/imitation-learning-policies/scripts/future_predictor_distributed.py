"""Single-node DDP support with a global, task-balanced training batch."""

from contextlib import contextmanager
from datetime import timedelta
import math
import os

import torch
import torch.distributed as dist
from torch.utils.data import Sampler, WeightedRandomSampler


class DistributedWeightedSampler(Sampler):
    """Shard one shared weighted draw, padding at most world_size - 1 items.

    Every rank receives the same number of samples and optimizer steps. The
    interleaved shards reconstruct the single-process sample sequence.
    """

    def __init__(self, weights, num_samples, rank, world_size, seed):
        if num_samples <= 0 or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed sampler size or rank")
        self.weights, self.num_samples = weights, num_samples
        self.rank, self.world_size, self.seed = rank, world_size, seed

    def __len__(self):
        return math.ceil(self.num_samples / self.world_size)

    def __iter__(self):
        draw = list(WeightedRandomSampler(
            self.weights, self.num_samples, replacement=True,
            generator=torch.Generator().manual_seed(self.seed)))
        size = len(self) * self.world_size
        if len(draw) < size:
            draw += (draw * math.ceil((size - len(draw)) / len(draw)))[:size - len(draw)]
        return iter(draw[self.rank:size:self.world_size])


class DistributedTraining:
    def __init__(self, device, rank=0, world_size=1):
        self.device, self.rank, self.world_size = device, rank, world_size

    @property
    def is_main(self):
        return self.rank == 0

    def broadcast(self, value):
        if self.world_size == 1:
            return value
        values = [value if self.is_main else None]
        dist.broadcast_object_list(values, src=0, device=self.device)
        return values[0]

    def on_main(self, function):
        """Propagate rank-zero failures before peers enter another collective."""
        if self.world_size == 1:
            return function()
        result = None
        if self.is_main:
            try:
                result = (True, function())
            except Exception as error:
                result = (False, f"{type(error).__name__}: {error}")
        success, value = self.broadcast(result)
        if not success:
            raise RuntimeError(f"Main process failed: {value}")
        return value

    def mean_loss(self, loss):
        value = loss.detach().clone()
        if self.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value /= self.world_size
        return value.item()

    def rng_states(self):
        state = dict(torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(self.device)
                     if self.device.type == "cuda" else None)
        if self.world_size == 1:
            return [state]
        states = [None] * self.world_size
        dist.all_gather_object(states, state)
        return states

    def restore_rng(self, state, seed):
        states = state.get("rank_rng_states")
        if states is not None and len(states) == self.world_size:
            state = states[self.rank]
        elif self.rank != 0:
            # Moving an old single-GPU run to DDP retains its optimizer/steps,
            # with a separate dropout stream on each newly introduced rank.
            torch.manual_seed(seed + self.rank + state["global_step"])
            return
        torch.set_rng_state(state["torch_rng"])
        if self.device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], self.device)


@contextmanager
def distributed_training(device_setting):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0")) if world_size > 1 else 0
    use_cuda = (torch.cuda.is_available() if device_setting == "auto"
                else str(device_setting).startswith("cuda"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
        if use_cuda:
            torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl" if use_cuda else "gloo",
                                timeout=timedelta(hours=1))
    else:
        device = torch.device(("cuda:0" if use_cuda else "cpu")
                              if device_setting == "auto" else device_setting)
    try:
        yield DistributedTraining(device, rank, world_size)
    finally:
        if world_size > 1:
            dist.destroy_process_group()
