import gc
import unittest
from unittest.mock import patch

import hydra
from omegaconf import open_dict
import torch

from imitation_learning.models.common.pos_embeddings import get_1d_sincos_pos_embed_from_grid
from imitation_learning.models.encoders.image_encoders import SharedModelManager
from imitation_learning.models.memory.gate_memory import GateMemoryBank
import test_future_prediction_mode as future_fixture


class HistoryTimeEncodingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        future_fixture.FuturePredictionModeTest.setUpClass()

    def setUp(self):
        self.fixture = future_fixture.FuturePredictionModeTest()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def config(self, mode="absolute", cached=True, future="tokens", learned=False):
        cfg = self.fixture.make_config(future, cached, merge_size=8)
        network = cfg.workspace.model.denoising_network_partial
        network.history_time_encoding = mode
        network.shadow_horizon = 1
        network.use_learned_history_query = learned
        cfg.workspace.model.memory_eviction.keep_first = False
        return cfg

    def expected(self, frames, dtype=torch.float32):
        frames = torch.as_tensor(frames)
        result = torch.from_numpy(get_1d_sincos_pos_embed_from_grid(32, frames.numpy()))
        return result.reshape(*frames.shape, 32).to(dtype=dtype)

    def test_absolute_encoding_uses_real_frames_and_is_invariant_to_bank_rank(self):
        policy = self.fixture.instantiate_policy(self.config())
        network = policy.denoising_network
        frames = torch.tensor([[0, 30, 80], [35, 90, 123456]])
        history = torch.randn(2, 3, 4, 32)
        values = history.clone()
        keys = network.add_history_key_encoding(history, network.history_time_embedding,
                                                history_frame_indices=frames)
        expected = history + self.expected(frames)[:, :, None, :]
        torch.testing.assert_close(keys, expected, rtol=0, atol=0)
        torch.testing.assert_close(history, values, rtol=0, atol=0)
        order = torch.tensor([2, 0, 1])
        permuted = network.add_history_key_encoding(history[:, order], network.history_time_embedding,
                                                    history_frame_indices=frames[:, order])
        torch.testing.assert_close(permuted, keys[:, order], rtol=0, atol=0)
        self.assertFalse(network.history_time_embedding.requires_grad)
        for dtype in (torch.float32, torch.bfloat16):
            embedding = network.absolute_frame_time_embedding(history.to(dtype), frames)
            self.assertEqual(embedding.dtype, dtype)
            torch.testing.assert_close(embedding, self.expected(frames, dtype), rtol=0, atol=0)

    def test_pack_keeps_frame_encoding_across_eviction_shadow_and_rank_changes(self):
        policy = self.fixture.instantiate_policy(self.config())
        bank = GateMemoryBank(2, keep_first=False, keep_recent=0, shadow_horizon=1)
        slot = torch.zeros(4, 32)
        reference = slot.unsqueeze(0)
        bank.append(slot, 30, torch.empty(0))
        bank.append(slot, 70, torch.ones(1))
        initial = policy._pack_history([bank], reference)
        self.assertEqual(initial["history_mask"].tolist(), [[False, True, True]])
        torch.testing.assert_close(initial["history_key_latents"][0, 1:, 0], self.expected([30, 70]))
        bank.append(slot, 100, torch.tensor([0.1, 0.9]))
        first = policy._pack_history([bank], reference)
        self.assertEqual([item.state for item in bank.slots], ["shadow", "active", "active"])
        torch.testing.assert_close(first["history_key_latents"][0, :, 0], self.expected([30, 70, 100]))
        bank.append(slot, 150, torch.tensor([0.9, 0.8, 0.1]))
        second = policy._pack_history([bank], reference)
        self.assertEqual([item.frame_index for item in bank.slots], [70, 100, 150])
        torch.testing.assert_close(second["history_key_latents"][0, :, 0], self.expected([70, 100, 150]))
        torch.testing.assert_close(first["history_key_latents"][0, 1], second["history_key_latents"][0, 0])
        self.assertEqual(float(second["history_latents"].abs().sum()), 0)

    def test_default_rank_mode_and_legacy_checkpoint_layout_are_preserved(self):
        cfg = self.fixture.make_config()
        self.assertEqual(cfg.workspace.model.denoising_network_partial.history_time_encoding, "relative")
        policy = self.fixture.instantiate_policy(cfg)
        network = policy.denoising_network
        history = torch.randn(1, 2, 4, 32)
        encoded = network.add_history_key_encoding(history, network.history_time_embedding)
        torch.testing.assert_close(encoded, history + network.history_time_embedding[:, :, None], rtol=0, atol=0)
        self.assertTrue(network.history_time_embedding.requires_grad)
        encoded.sum().backward()
        self.assertIsNotNone(network.history_time_embedding.grad)
        state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        del policy
        with open_dict(cfg.workspace.model.denoising_network_partial):
            del cfg.workspace.model.denoising_network_partial["history_time_encoding"]
        policy = self.fixture.instantiate_policy(cfg)
        policy.load_state_dict(state, strict=True)
        self.assertEqual(policy.denoising_network.history_time_encoding, "relative")
        del policy
        with open_dict(cfg.workspace.model.denoising_network_partial):
            cfg.workspace.model.denoising_network_partial.history_time_encoding = "absolute"
        policy = self.fixture.instantiate_policy(cfg)
        policy.load_state_dict(state, strict=True)
        self.assertEqual(set(policy.state_dict()), set(state))

    def test_single_and_window_projections_use_explicit_frame_metadata(self):
        policy = self.fixture.instantiate_policy(self.config())
        network = policy.denoising_network
        data = {
            "noisy_action": torch.randn(1, 16, 14), "step": torch.zeros(1, dtype=torch.long),
            "global_cond": torch.randn(1, 4, 8), "local_cond": torch.randn(1, 1, 8),
            "history_img_features": torch.randn(1, 3, 4, 8),
            "history_mask": torch.ones(1, 3, dtype=torch.bool),
            "history_frame_indices": torch.tensor([[30, 80, 1200]]),
        }
        inputs = network.prepare_single_memory_retrieval(data)
        expected = inputs["history_latents"] + self.expected([[30, 80, 1200]])[:, :, None]
        torch.testing.assert_close(inputs["history_key_latents"], expected)
        batch = self.fixture.make_batch(anchors=4)
        batch["traj_idx"] = torch.tensor([[[30], [40], [80], [140]]])
        data, _ = policy._encode_input_multi_traj(batch)
        data["history_frame_indices"] = batch["traj_idx"].squeeze(-1)
        for selected in (None, torch.tensor([[3, 1]])):
            projected_data = dict(data)
            if selected is not None:
                projected_data["training_traj_indices"] = selected
            projected = network.prepare_parallel_forward(projected_data)
            expected_frames = torch.tensor([[0, 0], [0, 30], [30, 40], [40, 80]])
            if selected is not None:
                expected_frames = expected_frames[selected[0]]
            expected = projected["history_latents"] + self.expected(expected_frames)[:, :, None]
            torch.testing.assert_close(projected["history_key_latents"], expected)

    def test_streaming_nonzero_start_keeps_frames_and_trains_both_query_modes(self):
        for cached, future, learned in ((True, "pooled", False), (True, "tokens", True),
                                         (False, "pooled", True), (False, "tokens", False)):
            with self.subTest(cached=cached, future=future, learned=learned):
                policy = self.fixture.instantiate_policy(self.config(cached=cached, future=future, learned=learned))
                network = policy.denoising_network
                optimizer = torch.optim.SGD(policy.parameters(), lr=0.001)
                batch = self.fixture.make_batch(cached, anchors=7)
                batch["traj_idx"] = torch.tensor([[[30], [40], [80], [110], [160], [200], [350]]])
                reads = []
                add_encoding = network.add_history_key_encoding

                def capture(history, table, history_frame_indices=None):
                    reads.append(history_frame_indices.clone())
                    return add_encoding(history, table, history_frame_indices)

                with patch.object(network, "add_history_key_encoding", side_effect=capture):
                    for start in (0, 3):
                        chunk = {key: value[:, start:start + 4] for key, value in batch.items()}
                        chunk.update(_streaming_supervised_length=3, _streaming_reset=start == 0,
                                     _streaming_final=start == 3)
                        optimizer.zero_grad(set_to_none=True)
                        losses = policy.compute_loss(chunk)
                        total = losses["action"] + .05 * losses["future_prediction"] + .01 * losses["gate_selectivity"]
                        self.assertTrue(torch.isfinite(total))
                        total.backward()
                        self.assertIsNone(network.history_time_embedding.grad)
                        self.assertGreater(sum(float(p.grad.abs().sum()) for p in network.memory_retriever.slot_gate.parameters()), 0)
                        if learned:
                            self.assertGreater(float(network.memory_retriever.history_query_tokens.grad.abs().sum()), 0)
                        optimizer.step()
                self.assertEqual(len(reads), 6)
                self.assertEqual(reads[1][0, -1].item(), 30)
                self.assertEqual(reads[3][0, -1].item(), 80)
                self.assertTrue(all(frame < 350 for read in reads for frame in read.flatten().tolist()))
                self.assertIsNone(policy._streaming_training_state)
                del policy

    def test_online_and_chunked_training_use_the_same_absolute_frames(self):
        policy = self.fixture.instantiate_policy(self.config(learned=True)).eval()
        batch = self.fixture.make_batch(anchors=7)
        reads = []
        network = policy.denoising_network
        add_encoding = network.add_history_key_encoding

        def capture(history, table, history_frame_indices=None):
            encoded = add_encoding(history, table, history_frame_indices)
            reads.append((history_frame_indices.detach().clone(), encoded.detach().clone()))
            return encoded

        with patch.object(network, "add_history_key_encoding", side_effect=capture):
            for start in (0, 3):
                chunk = {key: value[:, start:start + 4] for key, value in batch.items()}
                chunk.update(_streaming_supervised_length=3, _streaming_reset=start == 0,
                             _streaming_final=start == 3)
                with torch.no_grad():
                    policy.compute_loss(chunk)
            training_reads = list(reads)
            reads.clear()
            for anchor in range(6):
                with torch.no_grad():
                    output = policy.predict_action({
                        "head_camera_feature": batch["head_camera_feature"][:, anchor],
                        "agent_pos": batch["agent_pos"][:, anchor],
                        "episode_idx": torch.tensor([0]),
                    })
                self.assertTrue(torch.isfinite(output["action"]).all())
        self.assertEqual(len(reads), 6)
        for (train_frames, train_keys), (online_frames, online_keys) in zip(training_reads, reads):
            torch.testing.assert_close(online_frames, train_frames, rtol=0, atol=0)
            torch.testing.assert_close(online_keys, train_keys)
        self.assertEqual(policy.next_frame_index_dict[0], 60)
        policy.reset()
        self.assertEqual(policy.next_frame_index_dict, {})

    def test_invalid_configuration_or_missing_real_frame_metadata_fails_clearly(self):
        cfg = self.config(mode="relative_age")
        with self.assertRaisesRegex(hydra.errors.InstantiationException, "history_time_encoding"):
            self.fixture.instantiate_policy(cfg)
        policy = self.fixture.instantiate_policy(self.config())
        network = policy.denoising_network
        history = torch.zeros(1, 3, 4, 32)
        for frames in (None, torch.tensor([[0, 10]]), torch.tensor([[0, -1, 20]]),
                       torch.tensor([[0., 10.5, 20.]]), torch.tensor([[0., float("nan"), 20.]])):
            with self.subTest(frames=frames):
                with self.assertRaisesRegex(ValueError, "history_frame_indices"):
                    network.add_history_key_encoding(history, network.history_time_embedding,
                                                     history_frame_indices=frames)


if __name__ == "__main__":
    unittest.main()
