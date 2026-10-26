import gc
import inspect
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

from imitation_learning.datasets.rmbench_dataset import RMBenchMultiTrajDataset
from imitation_learning.models.denoising_networks.memory_transformer import (
    MemoryTransformer, ObservationDrivenMemoryRetriever,
)
from imitation_learning.models.encoders.image_encoders import SharedModelManager
from imitation_learning.policies.history_denoising_policy import HistoryDenoisingPolicy
from rmbench_model import RMBenchMemoryPolicy
import test_future_prediction_mode as fixture_module


class ObservationQueryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_module.FuturePredictionModeTest.setUpClass()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def retriever(self, tokens=3, queries=8, **kwargs):
        return ObservationDrivenMemoryRetriever(
            hidden_dim=16, head_num=4,
            history_attention_type="token_wise", ssmax_scaling_param=None,
            binary_gating=False, straight_through="", retrieval_mode="perception_patches",
            use_conditioned_slot_gate=True, perception_token_num=tokens,
            query_tokens=queries, **kwargs,
        )

    def config(self, mode="pooled", cached=True, cameras=None, task="rmbench"):
        cfg = fixture_module.FuturePredictionModeTest().make_config(mode, cached, merge_size=8, task=task)
        cfg.workspace.model.denoising_network_partial.query_tokens = 8
        cfg.workspace.model.denoising_network_partial.shadow_horizon = 2
        cfg.workspace.model.denoising_network_partial.layer_num = 2
        if cameras is not None:
            cfg.workspace.train_dataset.image_keys = cameras
        return cfg

    def batch(self, cfg, anchors=6):
        fixture = fixture_module.FuturePredictionModeTest()
        cached = cfg.workspace.train_dataset.use_cached_image_features
        batch = fixture.make_batch(cached, anchors)
        suffix = "_feature" if cached else ""
        image = batch.pop("head_camera" + suffix)
        for i, camera in enumerate(cfg.workspace.train_dataset.image_keys):
            batch[camera + suffix] = image + i * .1
        if cfg.dataset_type == "mikasa":
            batch["robot0_8d"] = batch.pop("agent_pos")[..., :8]
            batch["action0_8d"] = batch.pop("action")[..., :10, :8]
        return batch

    def policy(self, cfg):
        return fixture_module.FuturePredictionModeTest().instantiate_policy(cfg)

    @staticmethod
    def observation(batch, anchor):
        obs = {key: value[:, anchor] for key, value in batch.items()
               if key not in {"action", "action0_8d", "traj_idx", "entire_traj_is_padding"}}
        obs["episode_idx"] = torch.arange(batch["traj_idx"].shape[0])
        return obs

    def test_three_tokens_project_to_eight_and_fuse_without_realignment(self):
        retriever = self.retriever()
        image = torch.randn(2, 3, 16, requires_grad=True)
        state = torch.randn(2, 1, 16, requires_grad=True)
        history = torch.randn(2, 5, 3, 16, requires_grad=True)
        captured = []
        handle = retriever.perception_gate.register_forward_pre_hook(
            lambda module, args: captured.append(args))
        mask = torch.tensor([[True] * 5, [False] * 5])
        output = retriever(global_cond=image, local_cond=state,
                           history_latents=history, history_mask=mask)
        handle.remove()
        current, retrieved = captured[0]
        base = retriever.condition_current_tokens(image, state)
        expected = retriever.current_query_projector(base.flatten(1)).reshape(2, 8, 16)
        torch.testing.assert_close(current, expected)
        self.assertEqual(output.shape, (2, 8, 16))
        self.assertEqual(retrieved.shape, current.shape)
        torch.testing.assert_close(output[1], expected[1])
        torch.testing.assert_close(retriever(global_cond=image, local_cond=state), expected)
        torch.testing.assert_close(retriever(global_cond=image, local_cond=state,
                                             history_latents=history, memory_gate_val=torch.zeros(2)), expected)
        self.assertFalse(any("history_query" in name for name, _ in retriever.named_parameters()))
        self.assertFalse(hasattr(retriever, "query_tokens"))
        output.square().mean().backward()
        for value in (image.grad, state.grad, history.grad, retriever.current_query_projector.weight.grad):
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreater(float(value.abs().sum()), 0)
        self.assertEqual(float(history.grad[1].abs().sum()), 0)
        self.assertGreater(float(retriever.perception_gate.proj.weight.grad.abs().sum()), 0)

    def test_projection_count_masking_and_validation(self):
        for count in (1, 3, 8, 16):
            retriever = self.retriever(queries=count)
            image = torch.randn(2, 3, 16, requires_grad=True)
            mask = torch.tensor([[True, True, False], [False, False, False]])
            clean = retriever(global_cond=image, global_cond_mask=mask)
            dirty = image.masked_fill(~mask[..., None], float("nan"))
            torch.testing.assert_close(retriever(global_cond=dirty, global_cond_mask=mask), clean)
            self.assertEqual(clean.shape, (2, count, 16))
            clean.sum().backward()
            self.assertEqual(float(image.grad[~mask].abs().sum()), 0)
        retriever = self.retriever(queries=None)
        image = torch.randn(2, 3, 16)
        self.assertIsNone(retriever.current_query_projector)
        torch.testing.assert_close(retriever(global_cond=image), image)
        for value in (0, -1, True, 2.5, "8"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "query_tokens"):
                self.retriever(queries=value)

    def test_dit_and_future_receive_same_full_eight_tokens_for_all_views_and_targets(self):
        layouts = (("rmbench", ["head_camera"]),
                   ("rmbench", ["head_camera", "left_camera", "right_camera"]),
                   ("mikasa_remember_color_3", ["third_person_camera", "robot0_wrist_camera"]))
        for task, cameras in layouts:
            for mode in ("pooled", "tokens"):
                for cached in (False, True):
                    with self.subTest(task=task, cameras=cameras, mode=mode, cached=cached):
                        cfg = self.config(mode, cached, cameras, task)
                        policy = self.policy(cfg)
                        network = policy.denoising_network
                        batch = self.batch(cfg)
                        memories, dit, future = [], [], []
                        handles = [
                            network.memory_retriever.register_forward_hook(
                                lambda module, args, output: memories.append(output)),
                            policy.future_feature_predictor.register_forward_pre_hook(
                                lambda module, args, kwargs: future.append(kwargs), with_kwargs=True),
                        ]
                        for block in network.blocks:
                            handles.append(block.register_forward_pre_hook(
                                lambda module, args, kwargs: dit.append(kwargs), with_kwargs=True))
                        try:
                            losses = policy.compute_loss(batch)
                            h = torch.cat(memories)
                            self.assertEqual(h.shape, (6, 8, 32))
                            for inputs in dit:
                                self.assertEqual(inputs["global_cond"].shape[1], 4 * len(cameras))
                                torch.testing.assert_close(inputs["decision_memory"], h)
                            self.assertEqual(len(future), 1)
                            torch.testing.assert_close(future[0]["current_observation_tokens"], h[1:-1])
                            self.assertEqual(future[0]["action_chunk"].shape[1], 10)
                            self.assertEqual(policy.future_feature_predictor.future_token_num,
                                             len(cameras) * (1 if mode == "pooled" else 4))
                            projection = network.memory_retriever.current_query_projector.weight
                            for key in ("action", "future_prediction"):
                                gradient, = torch.autograd.grad(losses[key], projection, retain_graph=True)
                                self.assertTrue(torch.isfinite(gradient).all())
                                self.assertGreater(float(gradient.abs().sum()), 0)
                            sum(losses.values()).backward()
                            memories.clear()
                            policy.eval()
                            with patch.object(network, "effective_shadow_horizon", 0), torch.no_grad():
                                policy.compute_loss(batch)
                            expected = torch.cat(memories).detach().clone()
                            memories.clear()
                            future.clear()
                            policy.reset()
                            with torch.inference_mode():
                                for anchor in range(6):
                                    actions = policy.predict_action(self.observation(batch, anchor))
                                    self.assertTrue(all(torch.isfinite(value).all() for value in actions.values()))
                            torch.testing.assert_close(torch.cat(memories), expected, rtol=2e-5, atol=2e-5)
                            self.assertEqual(future, [])
                            self.assertEqual(policy.online_memory_banks[0].slots[-1].latent.shape[0], 4 * len(cameras))
                        finally:
                            for handle in handles:
                                handle.remove()
                        del policy

    def test_streaming_backward_reprojects_slots_and_keeps_eight_query_tokens(self):
        cfg = self.config("pooled")
        cfg.workspace.model.denoising_network_partial.history_time_encoding = "absolute"
        policy = self.policy(cfg)
        batch = {key: value.repeat(2, *([1] * (value.ndim - 1)))
                 for key, value in self.batch(cfg, anchors=7).items()}
        batch["entire_traj_is_padding"][1, 2:4] = True
        optimizer = torch.optim.SGD(policy.parameters(), lr=.001)
        batch.update(_streaming_sequence_chunk_length=3, _streaming_sequence_length=6)
        optimizer.zero_grad(set_to_none=True)
        losses = policy.compute_loss(batch)
        sum(losses.values()).backward()
        projection = policy.denoising_network.memory_retriever.current_query_projector
        self.assertGreater(float(projection.weight.grad.abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        optimizer.step()
        self.assertIsNone(policy._streaming_training_state)

    def test_checkpoint_adapter_roundtrip_uses_only_configured_cameras(self):
        cfg = self.config("pooled", cached=False)
        cfg.workspace.model.noise_scheduler.inference_step_num = 3
        policy = self.policy(cfg)
        normalizer = {f"normalizers.{name}.{parameter}": value
                      for name in ("agent_pos", "action")
                      for parameter, value in (("scale", torch.ones(14)), ("offset", torch.zeros(14)))}
        payload = dict(cfg_str_unresolved=OmegaConf.to_yaml(cfg),
                       model_state_dict=policy.state_dict(), normalizer_state_dict=normalizer)
        with patch("rmbench_model.torch_load", return_value=payload):
            adapter = RMBenchMemoryPolicy("/tmp/test-observation-query.ckpt", device="cpu")
        self.assertEqual(set(adapter.input_lengths), {"head_camera", "agent_pos"})
        self.assertEqual(adapter.policy.denoising_network.decision_memory_token_num, 8)
        for _ in range(3):
            actions = adapter.get_action({"head_camera": np.full((240, 320, 3), 127, dtype=np.uint8),
                                          "agent_pos": np.zeros(14, dtype=np.float32)})
            self.assertEqual(actions.shape, (10, 14))
            self.assertTrue(np.isfinite(actions).all())

    def test_configuration_defaults_select_projection_and_removed_paths_are_absent(self):
        retired = {"use_learned_history_query", "direct_bank", "dit_three_view"}
        for cls in (MemoryTransformer, ObservationDrivenMemoryRetriever, HistoryDenoisingPolicy, RMBenchMultiTrajDataset):
            self.assertFalse(retired.intersection(inspect.signature(cls).parameters))
        fixture = fixture_module.FuturePredictionModeTest()
        for task in ("rmbench", "observe_and_pickup", "swap_T", "mikasa_remember_color_3"):
            cfg = fixture.make_config(task=task)
            # The shared fixture explicitly requests null; task composition
            # itself must configure a positive query count and no retired switches.
            from imitation_learning.utils.config_utils import compose_hydra_config
            import hydra
            with hydra.initialize_config_dir(config_dir=str(fixture_module.CONFIG_DIR), version_base=None):
                actual = compose_hydra_config(task, "diffusion_memory_transformer")
            query_tokens = actual.workspace.model.denoising_network_partial.query_tokens
            self.assertIsInstance(query_tokens, int)
            self.assertGreater(query_tokens, 0)
            text = OmegaConf.to_yaml(actual)
            for name in retired | {"future_prediction_context_pooling"}:
                self.assertNotIn(name + ":", text)


if __name__ == "__main__":
    unittest.main()
