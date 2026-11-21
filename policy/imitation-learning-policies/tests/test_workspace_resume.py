import copy
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from imitation_learning.models.ema_model import EMAModel
from imitation_learning.trainers.base_trainer import BaseTrainer, get_scheduler
from imitation_learning.workspaces.base_workspace import BaseWorkspace


def make_ema(model):
    return EMAModel(model, inv_gamma=1.0, power=0.67, min_value=0.0, max_value=0.9999)


class ResumeTrainer:
    def init_trainer(self, **kwargs):
        self.optimizer = torch.optim.AdamW(kwargs['model'].parameters(), lr=0.1)
        self.optimizer.load_state_dict(copy.deepcopy(kwargs['optimizer_state_dict']))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1.0)
        self.scheduler.load_state_dict(kwargs['lr_scheduler_state_dict'])


class WorkspaceResumeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.model = torch.nn.Linear(2, 1)
        self.ema = make_ema(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1.0)
        for _ in range(3):
            self.update(self.model, self.optimizer, self.ema)
            self.scheduler.step()
        normalizer = torch.nn.Linear(1, 1)
        self.ckpt = copy.deepcopy({
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema.averaged_model.state_dict(),
            'normalizer_state_dict': normalizer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': 250, 'global_step': 108405,
        })
        self.workspace = BaseWorkspace.__new__(BaseWorkspace)
        self.workspace.model = torch.nn.Linear(2, 1)
        self.workspace.ema_model = make_ema(self.workspace.model)
        self.workspace.train_dataset = SimpleNamespace(normalizer=normalizer)
        self.workspace.val_dataset = SimpleNamespace(normalizer=copy.deepcopy(normalizer))
        self.workspace.trainer = ResumeTrainer()
        self.workspace.cfg_str_unresolved = ''
        self.workspace.logging_cfg = {}
        self.workspace.train_dataloader = []
        self.workspace.val_dataloader = []

    @staticmethod
    def update(model, optimizer, ema):
        optimizer.zero_grad()
        model(torch.ones(1, 2)).square().sum().backward()
        optimizer.step()
        ema.step(model)

    def test_legacy_workspace_matches_next_uninterrupted_update(self):
        self.workspace._load_workspace(self.ckpt)
        self.assertEqual(self.workspace.trainer.epoch, 251)
        self.assertEqual(self.workspace.trainer.global_step, 108406)
        self.assertEqual(int(self.workspace.ema_model.optimization_step), 3)
        for key, expected in self.ema.averaged_model.state_dict().items():
            torch.testing.assert_close(self.workspace.ema_model.averaged_model.state_dict()[key], expected)
        self.assertEqual(self.workspace.trainer.scheduler.state_dict(), self.scheduler.state_dict())
        self.update(self.model, self.optimizer, self.ema)
        self.update(self.workspace.model, self.workspace.trainer.optimizer, self.workspace.ema_model)
        for key, expected in self.ema.state_dict().items():
            torch.testing.assert_close(self.workspace.ema_model.state_dict()[key], expected)
        for key, expected in self.model.state_dict().items():
            torch.testing.assert_close(self.workspace.model.state_dict()[key], expected)

    def test_explicit_saved_ema_counter_takes_precedence(self):
        self.ckpt['ema_optimization_step'] = 17
        self.workspace._load_workspace(self.ckpt)
        self.assertEqual(int(self.workspace.ema_model.optimization_step), 17)

    def test_wrapped_ema_state_is_supported(self):
        self.ema.optimization_step.fill_(23)
        self.ckpt['ema_model_state_dict'] = copy.deepcopy(self.ema.state_dict())
        self.workspace._load_workspace(self.ckpt)
        for key, expected in self.ema.state_dict().items():
            torch.testing.assert_close(self.workspace.ema_model.state_dict()[key], expected)

    def test_trainer_restores_optimizer_lr_after_scheduler_initialization(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW([
            {'params': [], 'lr': 0.01}, {'params': list(model.parameters()), 'lr': 0.1},
        ], lr=0.1)
        scheduler = get_scheduler('cosine', optimizer=optimizer, num_warmup_steps=2, num_training_steps=40)
        for _ in range(3):
            optimizer.zero_grad()
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            scheduler.step()
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        saved_scheduler = copy.deepcopy(scheduler.state_dict())
        trainer = BaseTrainer.__new__(BaseTrainer)
        for name, value in dict(
            mixed_precision='', gradient_accumulation_steps=1, debug=True,
            optimizer_partial=partial(torch.optim.AdamW, lr=0.1),
            finetune_keywords=[], regular_keywords=[], finetune_lr_ratio=0.1,
            lr_scheduler_name='cosine', lr_warmup_steps=2, num_epochs=10,
        ).items():
            setattr(trainer, name, value)
        with patch('imitation_learning.trainers.base_trainer.Accelerator', return_value=SimpleNamespace(is_main_process=False)):
            trainer.init_trainer(
                train_dataloader=list(range(4)), val_dataloader=[], model=model,
                ema_model=make_ema(model), cfg_str_unresolved='', logging_cfg=None,
                optimizer_state_dict=saved_optimizer, lr_scheduler_state_dict=saved_scheduler,
            )
        self.assertEqual(
            [group['lr'] for group in trainer.optimizer.param_groups],
            [group['lr'] for group in saved_optimizer['param_groups']],
        )
        self.assertEqual(trainer.lr_scheduler.state_dict(), saved_scheduler)


if __name__ == '__main__':
    unittest.main()
