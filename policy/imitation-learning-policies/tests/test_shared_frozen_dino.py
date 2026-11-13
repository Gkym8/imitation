from types import SimpleNamespace
import unittest

import torch
from torch import nn

from imitation_learning.models.encoders.image_encoders import SharedModelManager
from imitation_learning.models.ema_model import EMAModel
import test_future_prediction_mode as fixture


MODEL_NAME = 'facebook/dinov2-with-registers-base-test'


class MatrixDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type='dinov2_with_registers')
        self.projection = nn.Linear(3, 8)
        self.dropout = nn.Dropout(0.5)
        self.matrix_dtype = None
        self.grad_enabled = None

    def forward(self, x):
        y = self.projection(x.mean(dim=(-2, -1)))
        self.matrix_dtype = y.dtype
        self.grad_enabled = torch.is_grad_enabled()
        return SimpleNamespace(last_hidden_state=self.dropout(y).unsqueeze(1))


class FrozenDinoPrecisionTest(unittest.TestCase):
    def setUp(self):
        SharedModelManager.reset()
        self.manager = SharedModelManager(cache_size=10)

    def tearDown(self):
        SharedModelManager.reset()

    def test_frozen_backbone_stays_eval_and_follows_caller_autocast(self):
        model = MatrixDino()
        self.manager.register_vit_model(MODEL_NAME, True, model=model)
        self.manager.train()
        self.assertFalse(model.training)
        self.assertEqual(model.projection.weight.dtype, torch.float32)
        image = torch.randn(2, 3, 12, 12)
        full_precision = self.manager.forward_patches(MODEL_NAME, image)
        self.assertEqual(full_precision.dtype, torch.float32)
        self.assertFalse(full_precision.requires_grad)
        self.manager.clear_cache()
        head = nn.Linear(8, 4)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            with torch.no_grad():
                expected = model(image).last_hidden_state
            actual = self.manager.forward_patches(MODEL_NAME, image)
            prediction = head(actual)
            loss = prediction.float().square().mean()
        loss.backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(model.matrix_dtype, torch.bfloat16)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(model.projection.weight.dtype, torch.float32)
        self.assertFalse(model.grad_enabled)
        self.assertFalse(actual.requires_grad)
        self.assertEqual(prediction.dtype, torch.bfloat16)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertGreater(float(head.weight.grad.abs().sum()), 0)

    def test_trainable_backbone_retains_gradients_and_mixed_precision(self):
        model = MatrixDino()
        model.dropout.p = 0
        self.manager.register_vit_model(MODEL_NAME, False, model=model)
        self.manager.train()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            feature = self.manager.forward_patches(MODEL_NAME, torch.randn(2, 3, 12, 12))
        feature.float().square().mean().backward()
        self.assertTrue(model.training)
        self.assertEqual(model.matrix_dtype, torch.bfloat16)
        self.assertGreater(float(model.projection.weight.grad.abs().sum()), 0)

    def test_legacy_duplicate_target_is_shared_and_strictly_loadable(self):
        model = MatrixDino()
        self.manager.register_vit_model(MODEL_NAME, True, model=model)
        key = self.manager.register_vit_model(MODEL_NAME, True, model=MatrixDino(), registry_name='future_target_dinov2')
        self.assertEqual(key, MODEL_NAME)
        self.assertEqual(len(self.manager.vit_models), 1)
        legacy = {k: v.detach().clone() for k,v in self.manager.state_dict().items()}
        for k,v in list(legacy.items()):
            legacy[k.replace(MODEL_NAME, 'future_target_dinov2')] = v.clone()
        self.manager.load_state_dict(legacy, strict=True)
        self.assertFalse(any('future_target_dinov2' in k for k in self.manager.state_dict()))
        image = torch.randn(2, 3, 12, 12)
        a = self.manager.forward_patches(MODEL_NAME, image)
        b = self.manager.forward_patches('future_target_dinov2', image)
        self.assertIs(a, b)
        legacy['vit_models.future_target_dinov2.projection.weight'] += 1
        with self.assertRaisesRegex(RuntimeError, 'differs'):
            self.manager.load_state_dict(legacy, strict=True)

    def test_trainable_current_and_frozen_target_remain_separate(self):
        self.manager.register_vit_model(MODEL_NAME, False, model=MatrixDino())
        key = self.manager.register_vit_model(MODEL_NAME, True, model=MatrixDino(), registry_name='future_target_dinov2')
        self.assertEqual(key, 'future_target_dinov2')
        self.assertEqual(len(self.manager.vit_models), 2)
        self.manager.train()
        self.assertTrue(self.manager.vit_models[MODEL_NAME].training)
        self.assertFalse(self.manager.vit_models[key].training)


class SharedFrozenDinoPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FuturePredictionModeTest.setUpClass()

    def tearDown(self):
        SharedModelManager.reset()

    def test_current_history_and_future_share_one_forward_for_both_mergers(self):
        helper = fixture.FuturePredictionModeTest()
        for learned_query_merge in (False, True):
            with self.subTest(learned_query_merge=learned_query_merge):
                cfg = helper.make_config('pooled', cached=False, merge_size=8)
                cfg.visual_encoder.learned_query_merge = learned_query_merge
                policy = helper.instantiate_policy(cfg)
                manager = policy.shared_model_manager
                self.assertEqual(list(manager.vit_models), [fixture.MODEL_NAME])
                for encoder in (policy.global_cond_encoder, policy.history_img_feature_encoder, policy.future_img_feature_encoder):
                    self.assertEqual(encoder.encoder_dict['head_camera'].shared_model_key, fixture.MODEL_NAME)
                calls = []
                handle = manager.vit_models[fixture.MODEL_NAME].register_forward_hook(lambda *args: calls.append(1))
                policy.train()
                self.assertFalse(manager.vit_models[fixture.MODEL_NAME].training)
                losses = policy.compute_loss(helper.make_batch(cached=False, anchors=3))
                handle.remove()
                self.assertEqual(len(calls), 1)
                (losses['action'] + 0.05 * losses['future_prediction']).backward()
                if learned_query_merge:
                    self.assertGreater(float(policy.denoising_network.global_cond_projector.mlp[0].weight.grad.abs().sum()), 0)
                self.assertTrue(all(p.grad is None for p in manager.vit_models[fixture.MODEL_NAME].parameters()))
                del policy

    def test_nested_legacy_ema_checkpoint_loads_without_duplicate_backbone(self):
        helper = fixture.FuturePredictionModeTest()
        cfg = helper.make_config('pooled', cached=False)
        # Exercise the saved registry key from pre-sharing online checkpoints.
        cfg.workspace.model.future_img_feature_encoder.image_encoder_partial.shared_model_key = 'future_target_dinov2'
        policy = helper.instantiate_policy(cfg)
        ema = EMAModel(policy, inv_gamma=1.0, power=0.67, min_value=0.0, max_value=0.9999)
        self.assertEqual(len(ema.averaged_model.shared_model_manager.vit_models), 1)
        self.assertIsNot(policy.shared_model_manager, ema.averaged_model.shared_model_manager)
        state = {k:v.detach().clone() for k,v in ema.state_dict().items()}
        marker = 'vit_models.' + fixture.MODEL_NAME + '.'
        for key,value in list(state.items()):
            if marker in key:
                state[key.replace(marker, 'vit_models.future_target_dinov2.')] = value.clone()
        ema.load_state_dict(state, strict=True)
        self.assertFalse(any('future_target_dinov2' in key for key in ema.state_dict()))
        ema.step(policy)


if __name__ == '__main__':
    unittest.main()
