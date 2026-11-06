import unittest
from unittest.mock import patch

import torch
from torch import nn

from imitation_learning.models.ema_model import EMAModel


class SmallPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2, bias=False)
        self.head = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.tensor([[0.1, -3.1415927], [1.4142135, 0.33333334]]))
            self.head.weight.fill_(2.0)
        self.backbone.requires_grad_(False)


class FrozenParameterEMATest(unittest.TestCase):
    def make_ema(self, model):
        return EMAModel(model, inv_gamma=1.0, power=0.67, min_value=0.0, max_value=0.9999)

    def test_frozen_weights_remain_bitwise_equal_after_many_steps(self):
        model = SmallPolicy()
        ema = self.make_ema(model)
        original = model.backbone.weight.detach().clone()
        for _ in range(4096):
            ema.step(model)
        self.assertTrue(torch.equal(model.backbone.weight, original))
        self.assertTrue(torch.equal(ema.averaged_model.backbone.weight, original))

    def test_trainable_weights_keep_averaging_and_newly_frozen_weights_sync(self):
        model = SmallPolicy()
        ema = self.make_ema(model)
        with patch.object(ema, '_get_decay', return_value=0.5), torch.no_grad():
            model.head.weight.fill_(4.0)
            ema.step(model)
            self.assertTrue(torch.equal(ema.averaged_model.head.weight, torch.full_like(model.head.weight, 3.0)))
            model.head.weight.fill_(8.0)
            ema.step(model)
            self.assertTrue(torch.equal(ema.averaged_model.head.weight, torch.full_like(model.head.weight, 5.5)))
            model.head.requires_grad_(False)
            model.head.weight.fill_(10.0)
            ema.step(model)
            self.assertTrue(torch.equal(ema.averaged_model.head.weight, model.head.weight))

    def test_existing_frozen_ema_drift_is_replaced_by_source_weights(self):
        model = SmallPolicy()
        ema = self.make_ema(model)
        # Reproduce a previously saved EMA copy differing from the frozen
        # training backbone. A later EMA step must also repair resumed copies.
        with torch.no_grad():
            ema.averaged_model.backbone.weight.add_(0.25)
        with patch.object(ema, '_get_decay', return_value=0.5):
            ema.step(model)
        self.assertTrue(torch.equal(ema.averaged_model.backbone.weight, model.backbone.weight))


if __name__ == '__main__':
    unittest.main()
