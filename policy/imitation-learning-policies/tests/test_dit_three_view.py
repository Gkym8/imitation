import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf, open_dict
import torch
import zarr

from imitation_learning.datasets.rmbench_dataset import (
    CAMERA_NAMES, RMBenchMultiTrajDataset, _prepare_kwargs,
)
from imitation_learning.models.encoders.image_encoders import SharedModelManager
from rmbench_model import RMBenchMemoryPolicy, _restore_dit_encoder_value_types
import test_future_prediction_mode as future_fixture
import test_rmbench_cached_features as cache_fixture


class DiTThreeViewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        future_fixture.FuturePredictionModeTest.setUpClass()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def config(self, cached=True, cameras=None, learned=True):
        cfg = future_fixture.FuturePredictionModeTest().make_config(
            mode="tokens", cached=cached, merge_size=8,
        )
        cfg.workspace.model.dit_three_view = True
        cfg.workspace.train_dataset.image_keys = cameras or ["head_camera"]
        cfg.workspace.model.denoising_network_partial.layer_num = 2
        cfg.workspace.model.denoising_network_partial.use_learned_history_query = learned
        return cfg

    def policy(self, cfg):
        return future_fixture.FuturePredictionModeTest().instantiate_policy(cfg)

    def batch(self, cached=True, anchors=3):
        batch = future_fixture.FuturePredictionModeTest().make_batch(cached, anchors)
        for camera in CAMERA_NAMES[1:]:
            if cached:
                batch[f"{camera}_feature"] = torch.randn(1, anchors, 1, 257, 8)
            else:
                batch[camera] = torch.rand(1, anchors, 1, 3, 224, 224)
        return batch

    def test_only_dit_expands_and_other_modules_follow_selected_views(self):
        for cached in (True, False):
            for cameras in (["head_camera"], ["right_camera", "left_camera"], list(CAMERA_NAMES)):
                for learned in (True, False):
                    with self.subTest(cached=cached, cameras=cameras, learned=learned):
                        cfg = self.config(cached, cameras, learned)
                        policy = self.policy(cfg)
                        batch = self.batch(cached)
                        network = policy.denoising_network
                        for encoder in (policy.global_cond_encoder,
                                        policy.history_img_feature_encoder,
                                        policy.future_img_feature_encoder):
                            self.assertEqual(list(encoder.data_entry_names), cameras)
                        self.assertTrue(cfg.workspace.train_dataset.dit_three_view)
                        self.assertEqual(list(policy.dit_img_feature_encoder.data_entry_names), list(CAMERA_NAMES))
                        selected_tokens = 4 * len(cameras)
                        if learned:
                            self.assertEqual(network.memory_retriever.history_query_tokens.shape[1], selected_tokens)
                        seen = []
                        handles = [block.register_forward_pre_hook(
                            lambda module, args, kwargs: seen.append((kwargs["global_cond"].shape,
                                                                     kwargs["decision_memory"].shape)),
                            with_kwargs=True,
                        ) for block in network.blocks]
                        try:
                            data, target = policy._encode_input_multi_traj(batch)
                            self.assertEqual(data["dit_cond"].shape[-2], 12)
                            self.assertEqual(data["global_cond"].shape[-2], selected_tokens)
                            self.assertEqual(data["history_img_features"].shape[-2], selected_tokens)
                            self.assertEqual(target["future_features"].shape[-2], selected_tokens)
                            losses = policy.compute_loss(batch)
                            self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                            self.assertEqual(len(seen), 2)
                            for current, memory in seen:
                                self.assertEqual(current[-2:], (12, 32))
                                self.assertEqual(memory[-2:], (selected_tokens, 32))
                        finally:
                            for handle in handles:
                                handle.remove()
                        del policy

    def test_extra_views_change_actions_but_not_memories_or_future_supervision(self):
        for cached in (True, False):
            with self.subTest(cached=cached):
                policy = self.policy(self.config(cached))
                batch = self.batch(cached)
                changed = dict(batch)
                for camera in CAMERA_NAMES[1:]:
                    key = f"{camera}_feature" if cached else camera
                    changed[key] = batch[key] + torch.randn_like(batch[key]) * 2
                memories = []
                handle = policy.denoising_network.blocks[0].register_forward_pre_hook(
                    lambda module, args, kwargs: memories.append(kwargs["decision_memory"].detach().clone()),
                    with_kwargs=True,
                )
                try:
                    policy.reset_rng()
                    torch.manual_seed(71)
                    original = policy.compute_loss(batch)
                    policy.reset_rng()
                    torch.manual_seed(71)
                    modified = policy.compute_loss(changed)
                finally:
                    handle.remove()
                torch.testing.assert_close(memories[0], memories[1], rtol=0, atol=0)
                for name in ("future_prediction", "gate_selectivity"):
                    torch.testing.assert_close(original[name], modified[name], rtol=0, atol=0)
                self.assertGreater(float((original["action"] - modified["action"]).abs()), 1e-6)
                del policy, original, modified

    def test_action_gradients_reach_extra_views_but_future_gradients_do_not(self):
        policy = self.policy(self.config())
        batch = self.batch()
        extra = [batch[f"{camera}_feature"].requires_grad_() for camera in CAMERA_NAMES[1:]]
        losses = policy.compute_loss(batch)
        future_grads = torch.autograd.grad(losses["future_prediction"], extra,
                                           allow_unused=True, retain_graph=True)
        self.assertTrue(all(gradient is None for gradient in future_grads))
        action_grads = torch.autograd.grad(losses["action"], extra)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in action_grads))
        self.assertTrue(all(float(gradient.abs().sum()) > 0 for gradient in action_grads))

    def test_streaming_updates_online_reuse_and_checkpoint_adapter_inputs(self):
        for cached in (True, False):
            with self.subTest(cached=cached):
                cfg = self.config(cached)
                cfg.workspace.model.noise_scheduler.inference_step_num = 3
                cfg.workspace.model.denoising_network_partial.query_tokens = 8
                cfg.workspace.model.denoising_network_partial.history_time_encoding = "absolute"
                cfg.workspace.model.denoising_network_partial.shadow_horizon = 3
                policy = self.policy(cfg)
                optimizer = torch.optim.SGD(policy.parameters(), lr=0.001)
                batch = self.batch(cached, anchors=7)
                for start in (0, 3):
                    chunk = {key: value[:, start:start + 4] for key, value in batch.items()}
                    chunk.update(_streaming_supervised_length=3, _streaming_reset=start == 0,
                                 _streaming_final=start == 3)
                    optimizer.zero_grad(set_to_none=True)
                    losses = policy.compute_loss(chunk)
                    sum(losses.values()).backward()
                    self.assertGreater(float(policy.denoising_network.dit_cond_pos_embedding.grad.abs().sum()), 0)
                    optimizer.step()
                policy.eval()
                calls = []
                handle = policy.denoising_network.memory_retriever.register_forward_hook(
                    lambda module, args, output: calls.append(output)
                )
                try:
                    with torch.no_grad():
                        for anchor in range(3):
                            observation = {key: value[:, anchor] for key, value in batch.items()
                                           if key not in {"action", "traj_idx", "entire_traj_is_padding"}}
                            observation["episode_idx"] = torch.tensor([0])
                            output = policy.predict_action(observation)
                            self.assertEqual(output["action"].shape, (1, 16, 14))
                            self.assertTrue(torch.isfinite(output["action"]).all())
                    self.assertEqual(len(calls), 3)
                finally:
                    handle.remove()
                normalizer_state = {
                    f"normalizers.{name}.{parameter}": value
                    for name in ("agent_pos", "action")
                    for parameter, value in (("scale", torch.ones(14)), ("offset", torch.zeros(14)))
                }
                checkpoint = dict(cfg_str_unresolved=OmegaConf.to_yaml(cfg),
                                  model_state_dict=policy.state_dict(), normalizer_state_dict=normalizer_state)
                with patch("rmbench_model.torch_load", return_value=checkpoint):
                    adapter = RMBenchMemoryPolicy("/tmp/test-dit-three-view.ckpt", device="cpu")
                self.assertEqual(set(adapter.input_lengths), {*CAMERA_NAMES, "agent_pos"})
                self.assertTrue(set(CAMERA_NAMES).issubset(adapter.transforms.data_meta))
                self.assertEqual(adapter.image_keys, ["head_camera"])
                np_actions = adapter.get_action({
                    **{camera: np.full((240, 320, 3), 127, dtype=np.uint8) for camera in CAMERA_NAMES},
                    "agent_pos": np.zeros(14, dtype=np.float32),
                })
                self.assertEqual(np_actions.shape, (10, 14))
                self.assertTrue(np.isfinite(np_actions).all())
                del adapter, policy

    def test_disabled_and_legacy_configs_have_identical_state_and_behavior(self):
        cfg = self.config()
        cfg.workspace.model.dit_three_view = False
        policy = self.policy(cfg)
        self.assertIsNone(policy.dit_img_feature_encoder)
        self.assertFalse(any("dit_" in key for key in policy.state_dict()))
        batch = self.batch()
        policy.reset_rng()
        torch.manual_seed(71)
        expected = {name: value.detach().clone() for name, value in policy.compute_loss(batch).items()}
        state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        with open_dict(cfg.workspace.model):
            del cfg.workspace.model.dit_three_view
            del cfg.workspace.model.dit_img_feature_encoder_partial
        del policy
        restored = self.policy(cfg)
        restored.load_state_dict(state, strict=True)
        restored.reset_rng()
        torch.manual_seed(71)
        actual = restored.compute_loss(batch)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_embedded_checkpoint_config_preserves_types_and_loads_legacy_alias(self):
        for legacy_alias in (False, True):
            with self.subTest(legacy_alias=legacy_alias):
                cfg = self.config(cached=False)
                cfg.run_name = "test-dit-checkpoint-types"
                cfg.project_name = "test"
                cfg.logger_project_name = "test"
                cfg.train_server_name = "localhost"
                with open_dict(cfg.workspace):
                    # Match training: Hydra resolves interpolations inside the
                    # YAML string while passing it into the workspace.
                    cfg.workspace.cfg_str_unresolved = OmegaConf.to_yaml(cfg)
                saved_config = cfg.workspace.cfg_str_unresolved
                restored_cfg = OmegaConf.create(saved_config)
                image_cfg = restored_cfg.workspace.model.dit_img_feature_encoder_partial.image_encoder_partial
                if legacy_alias:
                    # Replay the scalar types in the failed real checkpoint.
                    # These were emitted by its whole-dictionary alias.
                    for name in ("apply_image_norm", "resize_size", "individual_forward",
                                 "pretrained", "frozen", "spatial_merge_size"):
                        image_cfg[name] = str(image_cfg[name])
                    saved_config = OmegaConf.to_yaml(restored_cfg)
                self.assertIsInstance(image_cfg.spatial_merge_size, str if legacy_alias else int)
                policy = self.policy(cfg)
                normalizer_state = {
                    f"normalizers.{name}.{parameter}": value
                    for name in ("agent_pos", "action")
                    for parameter, value in (("scale", torch.ones(14)), ("offset", torch.zeros(14)))
                }
                checkpoint = dict(cfg_str_unresolved=saved_config,
                                  model_state_dict=policy.state_dict(), normalizer_state_dict=normalizer_state)
                with patch("rmbench_model.torch_load", return_value=checkpoint):
                    adapter = RMBenchMemoryPolicy("/tmp/test-dit-checkpoint-types.ckpt", device="cpu")
                for encoder in adapter.policy.dit_img_feature_encoder.encoder_dict.values():
                    self.assertEqual(encoder.spatial_merge_size, 8)
                    self.assertIs(encoder.apply_image_norm, True)
                    self.assertIs(encoder.individual_forward, False)
                    self.assertEqual(encoder.resize_size, (224, 224))
                actions = adapter.get_action({
                    **{camera: np.full((240, 320, 3), 127, dtype=np.uint8) for camera in CAMERA_NAMES},
                    "agent_pos": np.zeros(14, dtype=np.float32),
                })
                self.assertTrue(np.isfinite(actions).all())
                del adapter, policy

    def test_checkpoint_type_repair_preserves_explicit_differences_and_old_configs(self):
        cfg = self.config()
        image_cfg = cfg.workspace.model.dit_img_feature_encoder_partial.image_encoder_partial
        image_cfg.spatial_merge_size = 4
        image_cfg.individual_forward = True
        _restore_dit_encoder_value_types(cfg)
        self.assertEqual(image_cfg.spatial_merge_size, 4)
        self.assertIs(image_cfg.individual_forward, True)
        del cfg.workspace.model.dit_img_feature_encoder_partial
        _restore_dit_encoder_value_types(cfg)

    def test_dataset_loads_extra_patch_caches_without_changing_auto_selection(self):
        source, output = cache_fixture.make_meta()
        raw = _prepare_kwargs(["head_camera"], source, output, {}, extra_image_keys=list(CAMERA_NAMES))
        self.assertEqual(raw["image_keys"], list(CAMERA_NAMES))
        self.assertTrue(set(CAMERA_NAMES).issubset(raw["output_data_meta"]))
        with tempfile.TemporaryDirectory() as directory:
            root = zarr.open_group(str(Path(directory) / "test.zarr"), mode="w")
            episode = root.create_group("episode_0")
            for index, camera in enumerate(CAMERA_NAMES):
                array = episode.create_dataset(f"{camera}_patch_feature",
                                               data=np.full((20, 5, 8), index + 1, dtype=np.float32))
                array.attrs["complete"] = True
            for name in ("agent_pos", "action"):
                episode.create_dataset(name, data=np.zeros((20, 2), dtype=np.float32))
            root.attrs["episode_frame_nums"] = {"0": 20}
            root.attrs["dinov2_patch_feature_cache"] = dict(
                model_name=cache_fixture.MODEL_NAME, input_size=224, feature_kind="patches",
                feature_dim=8, patch_token_num=4, feature_dtype="float32", camera_names=list(CAMERA_NAMES),
            )
            dataset = RMBenchMultiTrajDataset(
                root_dir=directory, name="test", compressed_dir=directory, robot_num=2,
                include_episode_num=-1, include_episode_indices=[], used_episode_ratio=1.0,
                random_split_dataset=False, index_pool_size_per_episode=2,
                history_padding_length=0, future_padding_length=0, seed=0,
                source_data_meta=source, output_data_meta=output, dataloader_cfg={"batch_size": 1},
                starting_percentile_max=1.0, starting_percentile_min=0.0,
                apply_image_augmentation_in_cpu=True, use_relative_pose=False,
                use_relative_gripper_width=False, normalizer_sample_num=-1, normalizer_dir="",
                repeat_dataset_num=1, down_sample_steps=1, statistics_data_path="",
                image_keys=["head_camera"], dit_three_view=True, use_cached_image_features=True,
                cached_image_feature_dim=8, cached_image_feature_model_name=cache_fixture.MODEL_NAME,
                cached_image_feature_patch_token_num=4, traj_num=3, traj_interval_min=2,
                traj_interval_max=2, split_dataloader_cfg=None, episode_starting_idx_max=None,
            )
            self.assertEqual(dataset.cached_image_feature_kind, "patches")
            sample = dataset[0]
            for index, camera in enumerate(CAMERA_NAMES):
                values = sample[f"{camera}_feature"]
                self.assertEqual(values.shape, (3, 1, 5, 8))
                np.testing.assert_allclose(values, index + 1)


if __name__ == "__main__":
    unittest.main()
