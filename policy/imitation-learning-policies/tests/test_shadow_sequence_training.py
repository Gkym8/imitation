"""Distributed regression for saved Shadow Gates across chunk boundaries."""
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from imitation_learning.trainers.policy_trainer import PolicyTrainer
import test_shadow_memory as shadow_fixture


class _DistributedAccelerator:
    """Use real CPU DDP collectives with the trainer's small Accelerator API."""
    num_processes = 2
    sync_gradients = True

    def __init__(self, rank):
        self.is_main_process = rank == 0
        self.backward_calls = 0

    def unwrap_model(self, model):
        return model.module

    def gather(self, tensor):
        rows = [torch.empty_like(tensor) for _ in range(self.num_processes)]
        dist.all_gather(rows, tensor)
        return torch.cat(rows)

    def reduce(self, tensor, reduction):
        assert reduction == "sum"
        result = tensor.clone()
        dist.all_reduce(result)
        return result

    def accumulate(self, model):
        return nullcontext()

    def autocast(self):
        return nullcontext()

    def no_sync(self, model):
        return model.no_sync()

    def backward(self, loss, **kwargs):
        self.backward_calls += 1
        loss.backward(**kwargs)

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def _shadow_ddp_worker(rank, init_path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_path,
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    case = shadow_fixture.ShadowMemoryPolicyTest()
    case.setUpClass()
    case.setUp()
    try:
        cfg = case.make_config()
        cfg.workspace.model.denoising_network_partial.max_history_len = 2
        cfg.workspace.model.denoising_network_partial.shadow_horizon = 6
        cfg.workspace.model.memory_eviction.keep_recent = 0
        policy = case.fixture.instantiate_policy(cfg)
        trainer = PolicyTrainer.__new__(PolicyTrainer)
        trainer.model = DistributedDataParallel(policy, find_unused_parameters=True)
        trainer.accelerator = _DistributedAccelerator(rank)
        trainer.optimizer = torch.optim.SGD(policy.parameters(), lr=0.001)
        trainer.lr_scheduler = Mock()
        trainer.ema_model = Mock()
        trainer.epoch = 0
        trainer.clip_grad_norms = [1.0]
        trainer.streaming_chunk_length = 4
        trainer.streaming_chunk_accumulate = 1
        trainer.future_prediction_loss_weight = 0.05
        trainer.gate_selectivity_loss_weight = 0.01
        trainer.critical_action_loss_weights = [0.0]
        batch = case.fixture.make_batch(anchors=13)
        if rank == 1:
            # This rank has no local supervision in the last two chunks and
            # never creates a Shadow; the other rank must still backpropagate.
            batch["entire_traj_is_padding"][:, 2:] = True
        for _ in range(2):
            loss, logs = trainer.backward_streaming_batch(batch)
            assert torch.isfinite(loss)
            assert logs["info/grad_norm"] > 0
            assert policy._streaming_training_state is None
            assert trainer._last_streaming_step_count == 1
            weight = policy.denoising_network.memory_retriever.slot_gate.mlp[0].weight.detach()
            peers = [torch.empty_like(weight), torch.empty_like(weight)]
            dist.all_gather(peers, weight)
            torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)
        assert trainer.accelerator.backward_calls == 8
    finally:
        case.tearDown()
        dist.destroy_process_group()


class ShadowSequenceDistributedTest(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU DDP unavailable")
    def test_two_ranks_with_unequal_lengths_and_multiple_sequence_updates(self):
        with tempfile.TemporaryDirectory(prefix="shadow-ddp-") as directory:
            mp.spawn(_shadow_ddp_worker, args=(str(Path(directory) / "init"),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
