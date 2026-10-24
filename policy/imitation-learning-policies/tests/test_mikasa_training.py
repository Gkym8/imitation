from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import copy
import gc
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import default_collate
import zarr

from imitation_learning.models.encoders.image_encoders import SharedModelManager
from imitation_learning.datasets.mikasa_dataset import MikasaSingleTrajDataset
from scripts import train_mikasa as training
import test_future_prediction_mode as future_fixture

MODEL_NAME = future_fixture.MODEL_NAME


class MikasaTrainingTest(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    @contextmanager
    def dataset_config(self, *, cached=True, sample_range=None, frames=43):
        with tempfile.TemporaryDirectory(prefix="mikasa-training-") as directory:
            cfg = training.load_config("mikasa_remember_color_3")
            cfg.seed = 0
            cfg.workspace.train_dataset.root_dir = directory
            cfg.workspace.train_dataset.compressed_dir = directory
            cfg.workspace.train_dataset.normalizer_dir = directory
            cfg.workspace.train_dataset.used_episode_ratio = 1.0
            cfg.workspace.train_dataset.random_split_dataset = False
            cfg.workspace.train_dataset.use_cached_image_features = cached
            cfg.workspace.train_dataset.cached_image_feature_dim = 8
            cfg.workspace.train_dataset.sample_range = sample_range
            location = Path(directory) / "RememberColor3-v0/episode_data.zarr"
            root = zarr.open_group(str(location), mode="w")
            root.attrs["episode_frame_nums"] = {"0": frames}
            root.attrs["dinov2_patch_feature_cache"] = {
                "model_name": MODEL_NAME, "input_size": 224,
                "feature_kind": "patches", "feature_dim": 8,
                "feature_dtype": "float32", "patch_token_num": 256,
                "camera_names": ["third_person_camera"],
            }
            episode = root.create_group("episode_0")
            # Deliberately omit the wrist array: unused views must not be read.
            episode.create_dataset("third_person_camera", data=np.zeros((frames, 128, 128, 3), dtype=np.uint8))
            values = np.arange(frames, dtype=np.float32)[:, None]
            state = np.broadcast_to(values, (frames, 8)).copy()
            episode.create_dataset("robot0_8d", data=state)
            episode.create_dataset("action0_8d", data=state)
            features = np.broadcast_to(values[:, :, None], (frames, 257, 8)).copy()
            feature_array = episode.create_dataset("third_person_camera_patch_feature", data=features)
            feature_array.attrs["complete"] = True
            yield cfg, location

    def test_all_mikasa_task_configs_use_shared_method_and_only_fixed_camera(self):
        tasks = sorted((training.POLICY_ROOT / "imitation_learning/configs/task").glob("mikasa_*.yaml"))
        self.assertGreaterEqual(len(tasks), 9)
        for path in tasks:
            with self.subTest(task=path.stem):
                cfg = training.load_config(path.stem, ["workspace.train_dataset.sample_range=null"])
                training.validate_streaming_config(cfg)
                dataset = cfg.workspace.train_dataset
                self.assertEqual(dataset.anchor_sampling_mode, "streaming")
                self.assertIsNone(dataset.sample_range)
                self.assertEqual(list(dataset.image_keys), ["third_person_camera"])
                self.assertEqual(list(dataset.proprio_keys), ["robot0_8d"])
                self.assertEqual(dataset.starting_percentile_max, 1.0)
                self.assertEqual(dataset.future_padding_length, cfg.workspace.model.action_length - 1)
                self.assertEqual(cfg.visual_encoder.spatial_merge_size, 8)
                self.assertEqual(
                    dataset.use_cached_image_features,
                    training.load_config("mikasa").workspace.train_dataset.use_cached_image_features,
                )
                self.assertNotIn("history_action_num_per_chunk", cfg.workspace.model)

    def test_mikasa_uses_shared_policy_classes_and_causal_streaming(self):
        with hydra.initialize_config_dir(
            config_dir=str(training.POLICY_ROOT / "imitation_learning/configs"), version_base=None
        ), redirect_stdout(io.StringIO()):
            reference = training.compose_hydra_config("rmbench", "diffusion_memory_transformer")
        for task in ("mikasa_remember_color_3", "mikasa_remember_color_5", "mikasa_remember_color_9",
                     "mikasa_shell_game_touch", "mikasa_rotate_strict_pos", "mikasa_rotate_lenient_pos"):
            with self.subTest(task=task):
                actual = training.load_config(task)
                # Experiment knobs (e.g. history capacity and Shadow) may be
                # customized in task YAMLs; implementations remain shared.
                for subtree in ("visual_encoder", "workspace.model", "workspace.trainer",
                                "workspace.model.denoising_network_partial",
                                "workspace.model.global_cond_encoder", "workspace.model.local_cond_encoder",
                                "workspace.model.history_img_feature_encoder",
                                "workspace.model.future_img_feature_encoder",
                                "workspace.model.future_feature_predictor"):
                    self.assertEqual(
                        OmegaConf.select(actual, subtree + "._target_"),
                        OmegaConf.select(reference, subtree + "._target_"),
                        subtree,
                    )
                training.validate_streaming_config(actual)
                self.assertEqual(actual.workspace.train_dataset.anchor_sampling_mode, "streaming")

    def test_legacy_json_augmentation_metadata_does_not_override_current_transforms(self):
        for cached in (True, False):
            with self.subTest(cached=cached), self.dataset_config(cached=cached, sample_range=[0, 0]) as (cfg, location):
                path = training.ensure_normalizer(cfg)
                saved = json.loads(path.read_text())
                saved["third_person_camera"] = {
                    "type": "identity", "scale": 1.0, "offset": 0.0,
                    "meta": {"augmentation": [{"name": "InvalidLegacyColorJitter", "hue": 0.5}]},
                }
                path.write_text(json.dumps(saved))
                dataset = hydra.utils.instantiate(cfg.workspace.train_dataset)
                dataset[0]
                if cached:
                    self.assertEqual(dataset.output_data_meta["third_person_camera_feature"].augmentation, [])
                else:
                    self.assertEqual(
                        [item["name"] for item in dataset.output_data_meta["third_person_camera"].augmentation],
                        ["Resize"],
                    )
                    self.assertEqual(dataset.normalizer.data_meta["third_person_camera"].shape, (3, 224, 224))

    def test_cache_and_raw_dataset_exclude_wrist_and_keep_eight_action_channels(self):
        for cached in (True, False):
            with self.subTest(cached=cached), self.dataset_config(cached=cached, sample_range=[0, 0]) as (cfg, _):
                metadata = copy.deepcopy(OmegaConf.to_container(cfg.workspace.train_dataset.output_data_meta))
                dataset = hydra.utils.instantiate(cfg.workspace.train_dataset)
                sample = dataset[0]
                self.assertNotIn("robot0_wrist_camera", dataset.source_data_meta)
                self.assertNotIn("robot0_wrist_camera", sample)
                self.assertNotIn("robot0_wrist_camera_feature", sample)
                self.assertEqual(sample["action0_8d"].shape, (5, 10, 8))
                self.assertEqual(sample["robot0_8d"].shape, (5, 1, 8))
                key = "third_person_camera_feature" if cached else "third_person_camera"
                self.assertEqual(sample[key].shape[0], 5)
                self.assertEqual(OmegaConf.to_container(cfg.workspace.train_dataset.output_data_meta), metadata)
                self.assertTrue(sample["streaming_sequence"])
                self.assertEqual(int(sample["entire_traj_is_padding"].sum()), 0)

    def test_start_range_is_absolute_and_has_no_earlier_prefix(self):
        with self.dataset_config(sample_range=[30, 40], frames=83) as (cfg, _):
            dataset = hydra.utils.instantiate(cfg.workspace.train_dataset)
            self.assertEqual(dataset.index_pool, [(0, start) for start in range(30, 41)])
            for index, (_, start) in enumerate(dataset.index_pool):
                sample = dataset[index]
                actual = sample["third_person_camera_feature"][~sample["entire_traj_is_padding"], 0, 0, 0]
                expected = torch.tensor(list(range(start, 83, 10)), dtype=actual.dtype)
                torch.testing.assert_close(actual, expected)
                self.assertGreaterEqual(float(actual.min()), 30)

    def test_existing_complete_cache_avoids_backbone_loading(self):
        with self.dataset_config() as (cfg, location):
            with patch("scripts.extract_rmbench_dinov2_features.DinoV2FeatureExtractor", side_effect=AssertionError("cache should be reused")):
                training.ensure_feature_cache(cfg, location, 8, "cpu")

    def test_task_config_can_supply_data_root_without_a_command_line_override(self):
        with self.dataset_config() as (cfg, location):
            self.assertEqual(training.choose_data_root(cfg, None), location.parents[1])

    def test_missing_patch_cache_is_not_silently_replaced_by_old_pooled_features(self):
        with self.dataset_config() as (cfg, location):
            root = zarr.open_group(str(location), mode="a")
            del root.attrs["dinov2_patch_feature_cache"]
            del root["episode_0/third_person_camera_patch_feature"]
            root["episode_0"].create_dataset("third_person_camera_feature", shape=(43, 768), dtype="float32")
            with self.assertRaisesRegex(Exception, "complete DINOv2 feature-cache"):
                hydra.utils.instantiate(cfg.workspace.train_dataset)

    def test_missing_normalizer_is_fitted_and_reused_without_rgb_encoding(self):
        with self.dataset_config() as (cfg, _):
            path = training.ensure_normalizer(cfg)
            saved = json.loads(path.read_text())
            for field in ("robot0_8d", "action0_8d"):
                np.testing.assert_allclose(saved[field]["scale"], [21.0] * 8)
                self.assertEqual(saved[field]["offset"], [21.0] * 8)
            with patch("hydra.utils.instantiate", side_effect=AssertionError("reuse should not read data")):
                self.assertEqual(training.ensure_normalizer(cfg), path)

    def test_old_action_only_normalizer_backfills_only_state_and_keeps_original_backup(self):
        with self.dataset_config() as (cfg, location):
            path = location.parents[1] / "RememberColor3-v0_normalizer.json"
            original = {"action0_8d": {"scale": [17.0] * 8, "offset": [-2.0] * 8},
                        "third_person_camera": {"legacy_metadata": "preserve"}}
            original_text = json.dumps(original)
            path.write_text(original_text)
            fit = MikasaSingleTrajDataset.calc_stats_worker_single_process
            with patch.object(MikasaSingleTrajDataset, "calc_stats_worker_single_process",
                              autospec=True, side_effect=fit) as statistics:
                training.ensure_normalizer(cfg)
            self.assertEqual(statistics.call_args.args[-1], ["robot0_8d"])
            result = json.loads(path.read_text())
            self.assertEqual(result["action0_8d"], original["action0_8d"])
            self.assertEqual(result["third_person_camera"], original["third_person_camera"])
            self.assertIn("robot0_8d", result)
            backups = list(path.parent.glob(path.name + ".backup.*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), original_text)
            unchanged_time = path.stat().st_mtime_ns
            training.ensure_normalizer(cfg)
            self.assertEqual(path.stat().st_mtime_ns, unchanged_time)

    def test_absolute_joint_actions_keep_the_saved_transform_and_round_trip(self):
        with self.dataset_config(sample_range=[0, 0]) as (cfg, location):
            episode = zarr.open_group(str(location), mode="a")["episode_0"]
            absolute_targets = episode["robot0_8d"][:] + np.float32(0.075)
            episode["action0_8d"][:] = absolute_targets
            path = location.parents[1] / "RememberColor3-v0_normalizer.json"
            path.write_text(json.dumps({"action0_8d": {"scale": [17.0] * 8, "offset": [-2.0] * 8}}))
            training.ensure_normalizer(cfg)
            dataset = hydra.utils.instantiate(cfg.workspace.train_dataset)
            normalized = dataset[0]["action0_8d"][0]
            expected = torch.from_numpy(absolute_targets[:10])
            torch.testing.assert_close(normalized, (expected + 2.0) / 17.0)
            restored = dataset.normalizer.unnormalize({"action0_8d": normalized.clone()})["action0_8d"]
            torch.testing.assert_close(restored, expected)

    def test_streaming_interval_validation_and_runtime_overrides(self):
        cfg = training.load_config("mikasa_remember_color_3", [
            "workspace.train_dataset.sample_range=[30,40]",
            "workspace.model.denoising_network_partial.shadow_window=false",
            "workspace.model.execution_chunk_length=5",
        ])
        training.validate_streaming_config(cfg)
        self.assertEqual(list(cfg.workspace.train_dataset.sample_range), [30, 40])
        self.assertEqual(cfg.workspace.train_dataset.traj_interval_min, 5)
        self.assertFalse(cfg.workspace.model.denoising_network_partial.shadow_window)
        cfg.workspace.train_dataset.traj_interval_max = 6
        with self.assertRaisesRegex(ValueError, "interval must match"):
            training.validate_streaming_config(cfg)

    def test_dry_run_selects_current_code_and_does_not_prepare_or_launch(self):
        with self.dataset_config() as (_, location):
            output = io.StringIO()
            with patch.object(training, "ensure_feature_cache", side_effect=AssertionError("dry run")), \
                 patch.object(training, "ensure_normalizer", side_effect=AssertionError("dry run")), \
                 patch.object(training.subprocess, "run", side_effect=AssertionError("dry run")), redirect_stdout(output):
                training.main(["mikasa_remember_color_3", "2,3", "test_run", "--data-root", str(location.parents[1]),
                               "--dry-run", "--override", "workspace.train_dataset.sample_range=[0,0]"])
            command = output.getvalue()
            self.assertIn("CUDA_VISIBLE_DEVICES=2,3", command)
            self.assertIn("--num_processes 2", command)
            self.assertIn(str(training.POLICY_ROOT / "scripts/train_policy.py"), command)
            self.assertIn("+workspace.train_dataset.sample_range=[0,0]", command)
            self.assertNotIn("Using dataset config", command)

    def test_both_future_modes_train_and_online_inference_never_predicts_future(self):
        fixture = future_fixture.FuturePredictionModeTest()
        for mode in ("pooled", "tokens"):
            for cached in (True, False):
                with self.subTest(mode=mode, cached=cached), self.dataset_config(cached=cached, sample_range=[0, 0]) as (cfg, _):
                    cfg.device = "cpu"
                    model = cfg.workspace.model
                    model.future_prediction_mode = mode
                    for name in ("global_cond_encoder", "local_cond_encoder", "history_img_feature_encoder", "future_img_feature_encoder"):
                        model[name].feature_dim = 8
                    network = model.denoising_network_partial
                    network.hidden_dim, network.head_num, network.layer_num = 32, 4, 1
                    network.max_history_len, network.shadow_horizon = 2, 2
                    network.shadow_window = True
                    model.memory_eviction.keep_recent = 0
                    model.future_feature_predictor.predictor_hidden_dim = 32
                    model.future_feature_predictor.head_num = 4
                    model.future_feature_predictor.layer_num = 1
                    model.noise_scheduler.inference_step_num = 1
                    dataset = hydra.utils.instantiate(cfg.workspace.train_dataset)
                    batch = default_collate([dataset[0]])
                    policy = fixture.instantiate_policy(cfg)
                    self.assertEqual(policy.global_cond_encoder.token_num, 4)
                    self.assertEqual(policy.history_img_feature_encoder.token_num, 4)
                    self.assertEqual(policy.future_img_feature_encoder.token_num, 1 if mode == "pooled" else 4)
                    losses = policy.compute_loss(batch)
                    total = losses["action"] + .01 * losses["future_prediction"] + .01 * losses["gate_selectivity"]
                    self.assertTrue(torch.isfinite(total))
                    total.backward()
                    self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                        for p in policy.denoising_network.memory_retriever.slot_gate.parameters()))
                    policy.eval()
                    policy.reset()
                    future_calls = []
                    hook = policy.future_feature_predictor.register_forward_hook(lambda *args: future_calls.append(True))
                    image_key = "third_person_camera_feature" if cached else "third_person_camera"
                    for anchor in range(4):
                        observation = {key: batch[key][:, anchor] for key in (image_key, "robot0_8d")}
                        observation["episode_idx"] = torch.tensor([0])
                        with torch.inference_mode():
                            prediction = policy.predict_action(observation)
                        self.assertEqual(prediction["action0_8d"].shape, (1, 10, 8))
                    hook.remove()
                    self.assertEqual(future_calls, [])
                    self.assertEqual(policy.online_memory_banks[0].active_count, 2)
                    self.assertGreater(policy.online_memory_banks[0].shadow_count, 0)
                    del policy, losses, total
                    gc.collect()


if __name__ == "__main__":
    unittest.main()
