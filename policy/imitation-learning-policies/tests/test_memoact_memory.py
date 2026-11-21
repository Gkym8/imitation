import unittest
from unittest.mock import patch

import hydra
import torch
from omegaconf import OmegaConf

from baseline.MemoAct.config import configure_memoact
from baseline.MemoAct.memory_bank import MemoActMemoryBank
from imitation_learning.models.encoders.image_encoders import SharedModelManager
from imitation_learning.utils.config_utils import compose_training_config
import policy_fixture as fixture


class MeanCompressor:
    source_count = 3

    def __call__(self, observations):
        return observations.mean(dim=0)


class MemoActMemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.PolicyFixture.setUpClass()

    def tearDown(self):
        SharedModelManager.reset()

    def test_only_long_term_entries_merge_at_capacity(self):
        bank = MemoActMemoryBank(6, 2, MeanCompressor())
        # The incoming summary is most similar to its predecessor. The short
        # term contains matching entries too, but none may be merged.
        directions = [(-1., 0.), (1., 2.), (1., 0.), (-1., 0.), (-1., 0.)]
        for anchor in range(13):
            bank.append(torch.tensor([directions[anchor // 3]]), 10 * anchor)
            bank.prepare_read()
            self.assertLessEqual(len(bank) - bank.long_term_count, 6)
            self.assertLessEqual(bank.long_term_count, 2)

        self.assertEqual(bank.long_term_count, 2)
        self.assertEqual([slot.frame_index for slot in bank.slots], [0, 30, 90, 100, 110, 120])
        torch.testing.assert_close(bank.slots[0].latent, torch.tensor([[-1., 0.]]))
        torch.testing.assert_close(bank.slots[1].latent, torch.tensor([[1., 1.]]))
        for anchor, slot in enumerate(bank.slots[2:], start=9):
            torch.testing.assert_close(slot.latent, torch.tensor([directions[anchor // 3]]))

    def test_train_entry_uses_six_plus_eight(self):
        launch = OmegaConf.create({
            "task_name": "rearrange_blocks",
            "policy_name": "diffusion_memory_transformer",
            "workspace": {"model": {"memory_bank_type": "memoact"}},
        })
        with hydra.initialize_config_dir(config_dir=str(fixture.CONFIG_DIR), version_base=None):
            cfg = compose_training_config(launch)
        saved = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=False))
        model = saved.workspace.model
        self.assertEqual(model.memory_bank_type, "memoact")
        self.assertEqual((model.memoact_short_term_capacity, model.memoact_long_term_capacity), (6, 8))
        self.assertEqual(model.denoising_network_partial.max_history_len, 14)
        self.assertEqual(model.execution_chunk_length, 10)
        self.assertIsNone(model.memory_eviction)
        self.assertFalse(model.action_history)
        self.assertFalse(model.denoising_network_partial.use_conditioned_slot_gate)
        self.assertEqual(model.denoising_network_partial.shadow_horizon, 0)

    def test_shared_reader_and_gradients_across_streaming_boundary(self):
        helper = fixture.PolicyFixture()
        cfg = helper.make_config(merge_size=[4, 8])
        cfg.visual_encoder.state_query_merge = True
        cfg.workspace.model.memory_bank_type = "memoact"
        configure_memoact(cfg)
        policy = helper.instantiate_policy(cfg)
        batch = helper.make_batch(anchors=8)
        network = policy.denoising_network
        self.assertIsNotNone(network.memory_retriever.perception_gate)
        history_sizes = []
        retrieve = network.retrieve_decision_memory

        def observe(inputs, **kwargs):
            self.assertNotIn("history_shadow_mask", inputs)
            values = inputs["history_latents"]
            torch.testing.assert_close(
                inputs["history_key_latents"], network.add_history_key_encoding(values, network.history_time_embedding),
            )
            history_sizes.append(int(inputs["history_mask"].sum()))
            return retrieve(inputs, **kwargs)

        params = list(policy.memoact_compressor.parameters())
        with patch.object(network, "retrieve_decision_memory", side_effect=observe):
            chunk = {key: value[:, :8] for key, value in batch.items()}
            chunk.update(_streaming_supervised_length=7, _streaming_reset=True, _streaming_final=False)
            loss = policy.compute_streaming_loss(chunk)
            self.assertEqual(set(loss), {"action"})
            loss["action"].backward()
            self.assertTrue(all(param.grad is not None for param in params))
            self.assertTrue(all(torch.count_nonzero(param.grad) == 0 for param in params))
            bank = policy._streaming_training_state[0]
            self.assertEqual([source.frame_index for source in bank.pending_sources], [0, 10, 20])
            self.assertTrue(all(not source.latent.requires_grad for source in bank.pending_sources))

            policy.zero_grad(set_to_none=True)
            chunk = {key: value[:, 7:] for key, value in batch.items()}
            chunk.update(_streaming_supervised_length=1, _streaming_reset=False, _streaming_final=True)
            policy.compute_streaming_loss(chunk)["action"].backward()
            self.assertGreater(policy.memoact_compressor.summary_tokens.grad.norm().item(), 0)
            self.assertGreater(policy.memoact_compressor.layers[0].linear1.weight.grad.norm().item(), 0)
            self.assertEqual(history_sizes, [0, 1, 2, 3, 4, 5, 6, 5])
            self.assertIsNone(policy._streaming_training_state)

            history_sizes.clear()
            policy.eval()
            with torch.inference_mode():
                for anchor in range(8):
                    observation = {
                        key: batch[key][:, anchor] for key in ("head_camera_feature", "agent_pos")
                    }
                    observation["episode_idx"] = torch.tensor([0])
                    self.assertEqual(policy.predict_action(observation)["action"].shape, (1, 16, 14))
            self.assertEqual(history_sizes, [0, 1, 2, 3, 4, 5, 6, 5])
            self.assertEqual(policy.online_memory_banks[0].long_term_count, 1)
            policy.reset()
            self.assertEqual(policy.online_memory_banks, {})


if __name__ == "__main__":
    unittest.main()
