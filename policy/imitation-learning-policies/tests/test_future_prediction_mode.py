from pathlib import Path
from types import SimpleNamespace
import gc
import unittest

import hydra
from omegaconf import OmegaConf
import torch
from torch import nn

from imitation_learning.models.encoders.image_encoders import SharedModelManager
from imitation_learning.policies.history_denoising_policy import HistoryDenoisingPolicy
from imitation_learning.utils.config_utils import compose_hydra_config
from robot_utils.config_utils import register_resolvers


CONFIG_DIR = Path(__file__).resolve().parents[1] / "imitation_learning/configs"
MODEL_NAME = "facebook/dinov2-with-registers-base"


class DummyDinoV2(nn.Module):
    """Exercise the actual encoders without downloading backbone weights."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="dinov2_with_registers",
            hidden_size=8,
            patch_size=14,
            num_register_tokens=4,
        )
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        positions = torch.arange(261, dtype=pixel_values.dtype)[None, :, None]
        channels = torch.arange(8, dtype=pixel_values.dtype)[None, None, :]
        content = pixel_values.mean(dim=(1, 2, 3)).reshape(batch_size, 1, 1)
        return SimpleNamespace(last_hidden_state=content + (positions + channels).sin())


class FuturePredictionModeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not OmegaConf.has_resolver("eval"):
            register_resolvers()

    def tearDown(self):
        SharedModelManager.reset()

    def make_config(self, mode="pooled", cached=True, merge_size=4):
        with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose_hydra_config("rmbench", "diffusion_memory_transformer")
        cfg.device = "cpu"
        cfg.visual_encoder.spatial_merge_size = merge_size
        cfg.workspace.train_dataset.use_cached_image_features = cached
        model = cfg.workspace.model
        model.future_prediction_mode = mode
        for name in (
            "global_cond_encoder", "local_cond_encoder",
            "history_img_feature_encoder", "future_img_feature_encoder",
        ):
            model[name].feature_dim = 8
        model.denoising_network_partial.hidden_dim = 32
        model.denoising_network_partial.head_num = 4
        model.denoising_network_partial.layer_num = 1
        model.denoising_network_partial.max_history_len = 2
        # Existing feature-mode tests exercise immediate eviction. Shadow
        # lifecycle and both feature modes are covered in test_shadow_memory.
        model.denoising_network_partial.shadow_horizon = 0
        model.memory_eviction.keep_recent = 0
        model.future_feature_predictor.predictor_hidden_dim = 32
        model.future_feature_predictor.head_num = 4
        model.future_feature_predictor.layer_num = 1
        model.noise_scheduler.inference_step_num = 1
        return cfg

    def instantiate_policy(self, cfg):
        # Policy destructors reset the singleton; collect any previous graph
        # cycles before registering the next test's backbone.
        gc.collect()
        SharedModelManager.reset()
        manager = SharedModelManager(cache_size=10)
        manager.register_vit_model(MODEL_NAME, frozen=True, model=DummyDinoV2())
        if not cfg.workspace.train_dataset.use_cached_image_features:
            manager.register_vit_model(
                MODEL_NAME, frozen=True, model=DummyDinoV2(),
                registry_name="future_target_dinov2",
            )
        return hydra.utils.instantiate(cfg.workspace.model)

    def make_batch(self, cached=True, anchors=3):
        torch.manual_seed(2)
        batch = {
            "agent_pos": torch.randn(1, anchors, 1, 14),
            "action": torch.randn(1, anchors, 16, 14),
            "entire_traj_is_padding": torch.zeros(1, anchors, dtype=torch.bool),
            "traj_idx": (10 * torch.arange(anchors)).reshape(1, anchors, 1),
        }
        if cached:
            batch["head_camera_feature"] = torch.randn(1, anchors, 1, 257, 8)
        else:
            batch["head_camera"] = torch.rand(1, anchors, 1, 3, 224, 224)
        return batch

    def test_mode_switches_context_and_target_and_backpropagates(self):
        for mode in ("pooled", "tokens"):
            for cached in (True, False):
                with self.subTest(mode=mode, cached=cached):
                    cfg = self.make_config(mode, cached)
                    expected_tokens = 1 if mode == "pooled" else 16
                    expected_context = 1 if mode == "pooled" else 16
                    policy = self.instantiate_policy(cfg)
                    self.assertEqual(policy.future_feature_predictor.future_token_num, expected_tokens)
                    self.assertEqual(policy.future_img_feature_encoder.token_num, expected_tokens)
                    self.assertEqual(policy.history_img_feature_encoder.token_num, 16)

                    batch = self.make_batch(cached)
                    _, target = policy._encode_input_multi_traj(batch)
                    key = "head_camera_feature" if cached else "head_camera"
                    encoded = policy.future_img_feature_encoder({key: batch[key]})
                    self.assertEqual(target["future_features"].shape, (1, 3, expected_tokens, 8))
                    torch.testing.assert_close(target["future_features"][:, :-1], encoded[:, 1:])
                    self.assertTrue(target["future_feature_mask"][:, :-1].all())
                    self.assertFalse(target["future_feature_mask"][:, -1].any())

                    # Future predictions use the selected context shape.
                    contexts = []
                    handle = policy.future_feature_predictor.register_forward_pre_hook(
                        lambda module, args, kwargs: contexts.append(
                            kwargs["current_observation_tokens"].shape[-2]
                        ),
                        with_kwargs=True,
                    )
                    predictions = []
                    prediction_handle = policy.future_feature_predictor.register_forward_hook(
                        lambda module, args, output: predictions.append(output.detach()),
                    )
                    losses = policy.compute_loss(batch)
                    handle.remove()
                    prediction_handle.remove()
                    self.assertTrue(contexts)
                    self.assertEqual(set(contexts), {expected_context})
                    prediction = predictions[-1].reshape_as(target["future_features"])
                    mask = target["future_feature_mask"]
                    expected_prediction_loss = (
                        (prediction[mask].float() - target["future_features"][mask].float()).square().mean()
                        if mode == "tokens"
                        else (1.0 - torch.nn.functional.cosine_similarity(
                            prediction[mask], target["future_features"][mask], dim=-1,
                        )).mean()
                    )
                    torch.testing.assert_close(losses["future_prediction"].detach(), expected_prediction_loss)
                    total_loss = losses["action"] + 0.05 * losses["future_prediction"]
                    self.assertTrue(torch.isfinite(total_loss))
                    total_loss.backward()
                    self.assertTrue(any(
                        p.grad is not None and bool(p.grad.abs().sum() > 0)
                        for p in policy.denoising_network.memory_retriever.parameters()
                    ))

                    # Online memory needs only current observations and gates;
                    # future prediction is exclusively a training objective.
                    policy.eval()
                    policy.reset()
                    future_calls = []
                    future_hook = policy.future_feature_predictor.register_forward_hook(
                        lambda module, args, output: future_calls.append(output)
                    )
                    for anchor in range(2):
                        observation = {
                            key: batch[key][:, anchor],
                            "agent_pos": batch["agent_pos"][:, anchor],
                            "episode_idx": torch.tensor([0]),
                        }
                        with torch.inference_mode():
                            result = policy.predict_action(observation)
                        self.assertEqual(result["action"].shape, (1, 16, 14))
                    future_hook.remove()
                    self.assertEqual(future_calls, [])
                    del policy

    def test_streaming_chunks_use_both_prediction_modes(self):
        for mode in ("pooled", "tokens"):
            with self.subTest(mode=mode):
                policy = self.instantiate_policy(self.make_config(mode))
                batch = self.make_batch(anchors=4)
                expected_context = 1 if mode == "pooled" else 16
                contexts = []
                handle = policy.future_feature_predictor.register_forward_pre_hook(
                    lambda module, args, kwargs: contexts.append(
                        kwargs["current_observation_tokens"].shape[-2]
                    ),
                    with_kwargs=True,
                )
                for start, length, final in ((0, 2, False), (2, 1, True)):
                    chunk = {key: value[:, start:start + length + 1] for key, value in batch.items()}
                    chunk["_streaming_supervised_length"] = length
                    chunk["_streaming_reset"] = start == 0
                    chunk["_streaming_final"] = final
                    losses = policy.compute_loss(chunk)
                    total_loss = losses["action"] + 0.05 * losses["future_prediction"]
                    self.assertTrue(torch.isfinite(total_loss))
                    total_loss.backward()
                    policy.zero_grad(set_to_none=True)
                handle.remove()
                self.assertEqual(set(contexts), {expected_context})
                del policy

    def test_patch_target_count_follows_spatial_merge(self):
        cfg = self.make_config("tokens", merge_size=1)
        policy = self.instantiate_policy(cfg)
        self.assertEqual(policy.future_feature_predictor.future_token_num, 256)
        self.assertEqual(policy.future_img_feature_encoder.token_num, 256)

    def test_legacy_config_keeps_original_shapes_and_state_dict(self):
        cfg = self.make_config()
        pooled_policy = self.instantiate_policy(cfg)
        expected_state = pooled_policy.state_dict()
        del pooled_policy
        cfg.workspace.model.future_prediction_context_pooling = "mean"
        cfg.workspace.model.future_img_feature_encoder.image_encoder_partial.feature_aggregation = "map"
        del cfg.workspace.model.future_prediction_mode
        legacy_policy = self.instantiate_policy(cfg)
        legacy_policy.load_state_dict(expected_state, strict=True)
        self.assertEqual(legacy_policy.future_prediction_context_pooling, "mean")
        self.assertEqual(legacy_policy.future_feature_predictor.future_token_num, 1)

    def test_mode_rejects_unknown_value_and_mismatched_target(self):
        with self.assertRaisesRegex(ValueError, "future_prediction_mode"):
            HistoryDenoisingPolicy(
                skip_memory=False,
                history_img_feature_encoder=None,
                future_prediction_mode="unknown",
            )
        cfg = self.make_config("tokens")
        cfg.workspace.model.future_img_feature_encoder.image_encoder_partial.feature_aggregation = "map"
        with self.assertRaisesRegex(hydra.errors.InstantiationException, "requires future image"):
            self.instantiate_policy(cfg)


if __name__ == "__main__":
    unittest.main()
