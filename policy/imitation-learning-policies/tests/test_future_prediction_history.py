from __future__ import annotations

import gc
import unittest

import torch

from imitation_learning.models.encoders.image_encoders import SharedModelManager
import test_future_prediction_mode as fixture


class FutureHistoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FuturePredictionModeTest.setUpClass()

    def tearDown(self):
        SharedModelManager.reset()

    def make_policy(self, mode='pooled', cached=True):
        helper = fixture.FuturePredictionModeTest()
        return helper.instantiate_policy(helper.make_config(mode, cached)), helper

    @staticmethod
    def chunk(batch, start, supervised, *, reset, final=False):
        result = {key: value[:, start:start + supervised + 1] for key, value in batch.items()}
        result.update(_streaming_supervised_length=supervised,
                      _streaming_reset=reset, _streaming_final=final)
        return result

    def test_first_observation_is_action_only_then_carried_history_enables_prediction(self):
        for mode in ('pooled', 'tokens'):
            for cached in (False, True):
                with self.subTest(mode=mode, cached=cached):
                    policy, helper = self.make_policy(mode, cached)
                    batch = helper.make_batch(cached, anchors=4)
                    # The first sampled observation need not be episode frame 0.
                    batch['traj_idx'] += 30
                    calls = []
                    hook = policy.future_feature_predictor.register_forward_pre_hook(
                        lambda module, args, kwargs: calls.append(kwargs['current_observation_tokens'].shape[0]),
                        with_kwargs=True,
                    )
                    losses = policy.compute_loss(self.chunk(batch, 0, 1, reset=True))
                    self.assertEqual(calls, [])
                    self.assertEqual(float(losses['future_prediction']), 0.0)
                    self.assertGreater(float(losses['action']), 0.0)
                    self.assertEqual(len(policy._streaming_training_state[0]), 1)
                    self.assertEqual(policy._streaming_training_state[0].slots[0].frame_index, 30)
                    total = losses['action'] + .05 * losses['future_prediction'] + .01 * losses['gate_selectivity']
                    total.backward()
                    self.assertTrue(all(p.grad is None for p in policy.future_feature_predictor.parameters()))
                    self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                        for p in policy.denoising_network.parameters()))
                    policy.zero_grad(set_to_none=True)

                    # A chunk boundary is not an episode/stream reset.
                    losses = policy.compute_loss(self.chunk(batch, 1, 1, reset=False))
                    self.assertEqual(calls, [1])
                    losses['future_prediction'].backward()
                    self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                        for p in policy.future_feature_predictor.parameters()))
                    policy.zero_grad(set_to_none=True)

                    # The next stream again starts with an empty bank.
                    losses = policy.compute_loss(self.chunk(batch, 0, 1, reset=True, final=True))
                    self.assertEqual(calls, [1])
                    self.assertEqual(float(losses['future_prediction']), 0.0)
                    self.assertIsNone(policy._streaming_training_state)
                    hook.remove()
                    del policy, helper, losses, total, batch
                    gc.collect()

    def test_mixed_padding_excludes_each_rows_first_valid_observation(self):
        for mode in ('pooled', 'tokens'):
            with self.subTest(mode=mode):
                policy, helper = self.make_policy(mode)
                batch = {key: value.repeat(2, *([1] * (value.ndim - 1)))
                         for key, value in helper.make_batch(anchors=4).items()}
                batch['entire_traj_is_padding'][1, 0] = True
                batch['traj_idx'][0] += 30
                batch['traj_idx'][1] += 70
                _, targets = policy._encode_input_multi_traj(batch)
                predictions = []
                hook = policy.future_feature_predictor.register_forward_hook(
                    lambda module, args, output: predictions.append(output.detach()))
                losses = policy.compute_loss(self.chunk(batch, 0, 3, reset=True, final=True))
                hook.remove()
                self.assertEqual(len(predictions), 1)
                self.assertEqual(predictions[0].shape[0], 3)
                # Row 0: anchors 1,2; row 1: anchor 2. Neither first valid
                # observation (row 0 anchor 0 / row 1 anchor 1) predicts future.
                eligible = torch.tensor([[False, True, True], [False, False, True]])
                expected_targets = targets['future_features'][:, :3][eligible]
                expected = (
                    (predictions[0].float() - expected_targets.float()).square().mean()
                    if mode == 'tokens' else
                    (1 - torch.nn.functional.cosine_similarity(predictions[0], expected_targets, dim=-1)).mean()
                )
                torch.testing.assert_close(losses['future_prediction'].detach(), expected)
                (losses['action'] + .05 * losses['future_prediction']).backward()
                del policy, helper, losses, batch
                gc.collect()

    def test_existing_history_without_a_future_target_skips_prediction(self):
        for mode in ('pooled', 'tokens'):
            with self.subTest(mode=mode):
                policy, helper = self.make_policy(mode)
                batch = helper.make_batch(anchors=2)
                first = policy.compute_loss(self.chunk(batch, 0, 1, reset=True))
                (first['action'] + first['future_prediction']).backward()
                policy.zero_grad(set_to_none=True)
                calls = []
                hook = policy.future_feature_predictor.register_forward_hook(
                    lambda module, args, output: calls.append(output))
                # Last valid observation has past memory but no next image.
                losses = policy.compute_loss(self.chunk(batch, 1, 1, reset=False, final=True))
                hook.remove()
                self.assertEqual(calls, [])
                self.assertEqual(float(losses['future_prediction']), 0.0)
                self.assertTrue(torch.isfinite(losses['action']))
                (losses['action'] + losses['future_prediction']).backward()
                self.assertTrue(all(p.grad is None for p in policy.future_feature_predictor.parameters()))
                del policy, helper, losses, first, batch
                gc.collect()


if __name__ == '__main__':
    unittest.main()
