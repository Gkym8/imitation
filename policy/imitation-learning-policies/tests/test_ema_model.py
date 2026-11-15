import unittest
from unittest.mock import patch

import torch
from torch import nn

from imitation_learning.models.ema_model import EMAModel


class LegacyEMATest(unittest.TestCase):
    def test_frozen_and_trainable_parameters_both_use_original_ema_formula(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        model[0].requires_grad_(False)
        ema = EMAModel(model, inv_gamma=1.0, power=0.67, min_value=0.0, max_value=0.9999)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.25)
        for decay in (0.0, 0.5, 0.999):
            previous = [p.detach().clone() for p in ema.averaged_model.parameters()]
            with patch.object(ema, '_get_decay', return_value=decay):
                ema.step(model)
            for old, source, actual in zip(previous, model.parameters(), ema.averaged_model.parameters()):
                expected = old.mul_(decay).add_(source, alpha=1 - decay)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(int(ema.optimization_step), 3)
        self.assertTrue(all(not p.requires_grad for p in ema.averaged_model.parameters()))

    def test_frozen_saved_offset_is_averaged_rather_than_replaced(self):
        model = nn.Linear(2, 2, bias=False).requires_grad_(False)
        ema = EMAModel(model, inv_gamma=1.0, power=0.67, min_value=0.0, max_value=0.9999)
        with torch.no_grad():
            ema.averaged_model.weight.add_(0.25)
        expected = ema.averaged_model.weight.detach().clone().mul_(0.5).add_(model.weight, alpha=0.5)
        with patch.object(ema, '_get_decay', return_value=0.5):
            ema.step(model)
        torch.testing.assert_close(ema.averaged_model.weight, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(ema.averaged_model.weight, model.weight))


if __name__ == '__main__':
    unittest.main()
