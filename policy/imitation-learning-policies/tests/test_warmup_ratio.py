import math
from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from accelerate import Accelerator
from accelerate.scheduler import AcceleratedScheduler
from diffusers.optimization import get_scheduler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from imitation_learning.trainers.base_trainer import BaseTrainer


class _Streams(Dataset):
    def __init__(self, size=490, streaming=True):
        self.size = size
        self.anchor_sampling_mode = "streaming" if streaming else "contiguous"
        self.traj_num = 35

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.tensor([float(index)])


def _trainer(**overrides):
    kwargs = dict(
        seed=0, optimizer_partial=partial(torch.optim.SGD, lr=0.1),
        lr_scheduler_name="cosine", lr_warmup_steps=None, lr_warmup_ratio=0.05,
        num_epochs=301, val_every=0, sample_every=0, checkpoint_every=1,
        output_dir="/tmp/warmup-test", tqdm_interval_sec=1.0, mixed_precision="",
        gradient_accumulation_steps=1, checkpoint_manager=Mock(),
        finetune_keywords=[], regular_keywords=[], skip_saving_keywords=[],
        keep_saving_keywords=[], debug=True, loss_temperatures=[],
        clip_grad_norms=[1.0], finetune_lr_ratio=0.1,
    )
    kwargs.update(overrides)
    trainer = BaseTrainer(**kwargs)
    trainer.streaming_chunk_length = 4
    trainer.streaming_chunk_accumulate = 1
    trainer.model = SimpleNamespace(requires_full_sequence_backward=True)
    trainer.train_dataloader = DataLoader(_Streams(), batch_size=8)
    trainer.accelerator = SimpleNamespace(num_processes=4, process_index=0,
                                         split_batches=False, even_batches=True)
    return trainer


class WarmupRatioTest(unittest.TestCase):
    def test_four_gpu_budget_matches_sixteen_updates_and_five_percent(self):
        trainer = _trainer()
        updates, ticks = trainer._scheduler_step_counts()
        self.assertEqual((updates, ticks), (16, 4))
        total = updates * trainer.num_epochs
        warmup = trainer._resolved_warmup_steps(total * ticks)
        self.assertEqual(total, 4816)
        self.assertEqual(warmup, 964)
        self.assertEqual(math.ceil(warmup / ticks), 241)

    def test_batch_gpu_and_epoch_changes_recalculate(self):
        trainer = _trainer(num_epochs=100)
        trainer.train_dataloader = DataLoader(_Streams(1000), batch_size=4)
        trainer.accelerator.num_processes = 2
        updates, ticks = trainer._scheduler_step_counts()
        self.assertEqual((updates, ticks), (125, 2))
        self.assertEqual(trainer._resolved_warmup_steps(updates * 100 * ticks), 1250)

    def test_gradient_accumulation_uses_optimizer_updates(self):
        trainer = _trainer(gradient_accumulation_steps=2)
        self.assertEqual(trainer._scheduler_step_counts(), (8, 4))

    def test_split_batches_and_drop_last_follow_real_sharding(self):
        trainer = _trainer()
        trainer.accelerator.split_batches = True
        self.assertEqual(trainer._scheduler_step_counts(), (62, 1))
        trainer.accelerator.split_batches = False
        trainer.train_dataloader = DataLoader(_Streams(), batch_size=8, drop_last=True)
        self.assertEqual(trainer._scheduler_step_counts(), (15, 4))

    def test_chunked_and_nonstreaming_budgets(self):
        trainer = _trainer()
        trainer.model.requires_full_sequence_backward = False
        trainer.streaming_chunk_accumulate = 2
        self.assertEqual(trainer._scheduler_step_counts(), (80, 4))
        trainer.train_dataloader = DataLoader(_Streams(streaming=False), batch_size=8)
        self.assertEqual(trainer._scheduler_step_counts(), (16, 4))

    def test_zero_and_full_ratio(self):
        self.assertEqual(_trainer(lr_warmup_ratio=0)._resolved_warmup_steps(101), 0)
        self.assertEqual(_trainer(lr_warmup_ratio=1)._resolved_warmup_steps(101), 101)
        self.assertEqual(_trainer(lr_warmup_ratio=.01)._resolved_warmup_steps(101), 2)

    def test_explicit_steps_override_ratio_and_old_configs_work(self):
        self.assertEqual(_trainer(lr_warmup_steps=1000)._resolved_warmup_steps(20000), 1000)
        self.assertEqual(_trainer(lr_warmup_steps=2000, lr_warmup_ratio=None)._resolved_warmup_steps(20000), 2000)
        self.assertEqual(_trainer(lr_warmup_steps=0)._resolved_warmup_steps(20000), 0)

    def test_invalid_settings_are_rejected(self):
        for value in (True, -.1, 1.01, math.nan, math.inf, "0.05"):
            with self.subTest(ratio=value), self.assertRaisesRegex(ValueError, "lr_warmup_ratio"):
                _trainer(lr_warmup_ratio=value)
        for value in (True, -1, 1.5, "100"):
            with self.subTest(steps=value), self.assertRaisesRegex(ValueError, "lr_warmup_steps"):
                _trainer(lr_warmup_steps=value)
        with self.assertRaisesRegex(ValueError, "set lr_warmup_ratio"):
            _trainer(lr_warmup_ratio=None)

    def test_empty_training_loader_is_rejected(self):
        trainer = _trainer()
        trainer.train_dataloader = DataLoader(_Streams(0), batch_size=8)
        with self.assertRaisesRegex(ValueError, "at least one"):
            trainer._scheduler_step_counts()

    def test_accelerate_curve_reaches_peak_at_241_and_zero_at_4816_updates(self):
        trainer = _trainer()
        updates, ticks = trainer._scheduler_step_counts()
        total = updates * trainer.num_epochs * ticks
        parameter = nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = get_scheduler("cosine", optimizer,
                                  num_warmup_steps=trainer._resolved_warmup_steps(total),
                                  num_training_steps=total)
        accelerated = AcceleratedScheduler(scheduler, [SimpleNamespace(step_was_skipped=False)], split_batches=False)
        accelerated.gradient_state = SimpleNamespace(sync_gradients=True, adjust_scheduler=True)
        with patch("accelerate.scheduler.AcceleratorState", return_value=SimpleNamespace(num_processes=4)):
            for step in range(1, 4817):
                optimizer.step()
                accelerated.step()
                if step == 240:
                    self.assertLess(optimizer.param_groups[0]["lr"], 0.1)
                if step == 241:
                    self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)
            self.assertEqual(scheduler.last_epoch, total)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0)

    def test_real_trainer_initialization_resolves_ratio_and_restores_scheduler(self):
        trainer = _trainer(num_epochs=10)
        model = nn.Linear(1, 1)
        model.requires_full_sequence_backward = True
        loader = DataLoader(_Streams(80), batch_size=8)
        # Exercise init_trainer/get_scheduler using a real CPU Accelerator.
        with patch("imitation_learning.trainers.base_trainer.Accelerator",
                   side_effect=lambda **kwargs: Accelerator(cpu=True, **kwargs)):
            trainer.init_trainer(loader, loader, model, Mock(), "{}")
        self.assertEqual(trainer.optimizer_steps_per_epoch, 10)
        self.assertEqual(trainer.total_optimizer_steps, 100)
        self.assertEqual(trainer.resolved_lr_warmup_steps, 5)
        for _ in range(3):
            trainer.optimizer.step()
            trainer.lr_scheduler.step()
        state = trainer.lr_scheduler.state_dict()
        other = _trainer(num_epochs=10)
        other_model = nn.Linear(1, 1)
        other_model.requires_full_sequence_backward = True
        with patch("imitation_learning.trainers.base_trainer.Accelerator",
                   side_effect=lambda **kwargs: Accelerator(cpu=True, **kwargs)):
            other.init_trainer(loader, loader, other_model, Mock(), "{}", lr_scheduler_state_dict=state)
        self.assertEqual(other.lr_scheduler.last_epoch, trainer.lr_scheduler.last_epoch)


if __name__ == "__main__":
    unittest.main()
