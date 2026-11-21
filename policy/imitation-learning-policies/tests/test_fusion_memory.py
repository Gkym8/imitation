import unittest
from unittest.mock import patch

import hydra
import torch
from omegaconf import OmegaConf

from baseline.Fusion.config import configure_fusion
from baseline.Fusion.memory_bank import FusionMemoryBank
from imitation_learning.models.encoders.image_encoders import SharedModelManager
from imitation_learning.utils.config_utils import compose_training_config
import policy_fixture as fixture


class FusionMemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.PolicyFixture.setUpClass()

    def tearDown(self):
        SharedModelManager.reset()

    def test_selects_adjacent_pair_including_incoming_slot(self):
        bank = FusionMemoryBank(3)
        zero_degree = torch.tensor([[1., 0.], [1., 0.]])
        right_angle = torch.tensor([[0., 1.], [0., 1.]])
        latent = [zero_degree, right_angle, zero_degree.clone(), 2 * zero_degree]
        for k, item in enumerate(latent):
            bank.append(item.requires_grad_(), 10 * k, raw_features=item.detach())
        self.assertEqual([slot.frame_index for slot in bank.slots], [0, 10, 20])
        self.assertEqual(bank.next_anchor_index, 4)
        self.assertEqual(bank.shadow_count, 0)
        merged = bank.slots[-1]
        torch.testing.assert_close(merged.latent, 1.5 * zero_degree)
        merged.latent.sum().backward()
        for item in latent[-2:]:
            torch.testing.assert_close(item.grad, torch.full_like(item, 0.5))
        bank.detach()
        bank.reproject(lambda features: features * 10)
        torch.testing.assert_close(bank.slots[0].latent, zero_degree * 10)
        torch.testing.assert_close(merged.latent, 1.5 * zero_degree)
        self.assertIsNone(merged.raw_features)

    def test_train_entry_disables_task_overrides(self):
        launch = OmegaConf.create({
            "task_name": "rearrange_blocks",
            "policy_name": "diffusion_memory_transformer",
            "workspace": {"model": {"memory_bank_type": "fusion"}},
        })
        with hydra.initialize_config_dir(config_dir=str(fixture.CONFIG_DIR), version_base=None):
            cfg = compose_training_config(launch)
        saved = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=False))
        model = saved.workspace.model
        self.assertEqual(model.memory_bank_type, "fusion")
        self.assertEqual(model.execution_chunk_length, 10)
        self.assertEqual(model.denoising_network_partial.max_history_len, 8)
        self.assertFalse(model.action_history)
        self.assertFalse(model.denoising_network_partial.use_conditioned_slot_gate)
        self.assertEqual(model.denoising_network_partial.shadow_horizon, 0)

    def test_streaming_and_online_read_fusion_with_relative_positions(self):
        helper = fixture.PolicyFixture()
        cfg = helper.make_config(merge_size=[4, 8])
        cfg.visual_encoder.state_query_merge = True
        cfg.workspace.model.memory_bank_type = "fusion"
        cfg.workspace.model.denoising_network_partial.max_history_len = 8
        cfg.workspace.model.denoising_network_partial.state_as_dit_token = True
        configure_fusion(cfg)
        policy = helper.instantiate_policy(cfg)
        batch = helper.make_batch(anchors=10)
        network = policy.denoising_network
        self.assertIsNone(network.memory_retriever.slot_gate)
        self.assertFalse(policy.action_history)
        self.assertFalse(any("future_" in key or "slot_gate" in key for key in policy.state_dict()))
        observations = []

        def record(inputs, **kwargs):
            for key in ("history_shadow_mask", "history_shadow_steps"):
                self.assertNotIn(key, inputs)
            values = inputs["history_latents"]
            keys = inputs["history_key_latents"]
            torch.testing.assert_close(keys, network.add_history_key_encoding(
                values, network.history_time_embedding,
            ))
            observations.append((inputs["history_mask"].sum().item(), values.detach().clone()))

        retrieve = network.retrieve_decision_memory

        def observe(inputs, **kwargs):
            record(inputs)
            return retrieve(inputs, **kwargs)

        with patch.object(network, "retrieve_decision_memory", side_effect=observe):
            for start, count in ((0, 4), (4, 4), (8, 1), (9, 1)):
                chunk = {
                    key: value[:, start:start + count + 1] for key, value in batch.items()
                }
                chunk.update(_streaming_supervised_length=count, _streaming_reset=start == 0,
                             _streaming_final=start == 9)
                loss = policy.compute_streaming_loss(chunk)
                self.assertEqual(set(loss), {"action"})
                loss["action"].backward()
                if start == 8:
                    bank = policy._streaming_training_state[0]
                    self.assertEqual(bank.next_anchor_index, 9)
                    self.assertEqual(len(bank), 8)
                    merged_index, merged = next(
                        (i, slot.latent.clone()) for i, slot in enumerate(bank.slots)
                        if slot.raw_features is None
                    )
            torch.testing.assert_close(observations[-1][1][0, merged_index], merged)
            self.assertEqual([count for count, _ in observations], list(range(9)) + [8])
            self.assertIsNone(policy._streaming_training_state)

            policy.eval()
            observations.clear()
            with torch.inference_mode():
                for anchor in range(10):
                    observation = {key: batch[key][:, anchor]
                                   for key in ("head_camera_feature", "agent_pos")}
                    observation["episode_idx"] = torch.tensor([0])
                    self.assertEqual(policy.predict_action(observation)["action"].shape, (1, 16, 14))
                    bank = policy.online_memory_banks[0]
                    self.assertEqual(len(bank), min(anchor + 1, 8))
                    self.assertEqual(bank.shadow_count, 0)
                    if anchor == 8:
                        fused_bank = torch.stack([slot.latent.clone() for slot in bank.slots])
            torch.testing.assert_close(observations[-1][1][0], fused_bank)
            self.assertEqual(policy.next_frame_index_dict[0], 100)


if __name__ == "__main__":
    unittest.main()
