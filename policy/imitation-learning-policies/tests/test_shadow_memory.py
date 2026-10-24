import gc
import math
import unittest
from unittest.mock import patch

import hydra
from omegaconf import OmegaConf, open_dict
import torch

from imitation_learning.models.encoders.image_encoders import SharedModelManager
from imitation_learning.models.memory.gate_memory import GateMemoryBank, gate_selectivity_penalty
import test_conditioned_slot_gate as gate_fixture
import test_future_prediction_mode as future_fixture


class ShadowMemoryBankTest(unittest.TestCase):
    def append(self, bank, scores=None):
        i = bank.next_anchor_index
        if scores is None:
            scores = torch.ones(len(bank))
        return bank.append(torch.full((2, 3), float(i)), 10 * i, scores)

    def test_demotion_preserves_identity_and_expires_after_exact_future_reads(self):
        bank = GateMemoryBank(2, keep_first=False, keep_recent=0, shadow_horizon=2)
        self.append(bank)
        self.append(bank)
        original = bank.slots[0]
        self.assertEqual(self.append(bank, torch.tensor([0.1, 0.9])), 0)
        self.assertIs(bank.slots[0], original)
        self.assertEqual((original.state, original.expire_at), ("shadow", 4))
        self.assertEqual((bank.active_count, bank.shadow_count), (2, 1))
        # Anchor 3 reads slot 0 as Shadow. Its tiny score cannot reselect it.
        self.assertEqual(self.append(bank, torch.tensor([0.0, 0.1, 0.9])), 1)
        self.assertIn(original, bank.slots)
        # Slot 0 is still present for the read at anchor 4, then expires.
        self.assertEqual(bank.next_anchor_index, 4)
        self.assertIn(original, bank.slots)
        self.append(bank, torch.tensor([0.0, 0.0, 0.1, 0.9]))
        self.assertNotIn(0, [slot.slot_id for slot in bank.slots])
        self.assertEqual((bank.active_count, bank.shadow_count), (2, 2))

    def test_long_stream_has_bounded_storage_and_fixed_shadow_lifetime(self):
        horizon = 3
        bank = GateMemoryBank(4, keep_first=True, keep_recent=1, shadow_horizon=horizon)
        demoted_at, shadow_reads = {}, {}
        for anchor in range(30):
            for slot in bank.slots:
                if slot.state == "shadow":
                    shadow_reads.setdefault(slot.slot_id, []).append(anchor)
            demoted = self.append(bank)
            if demoted is not None:
                demoted_at[demoted] = anchor
            self.assertLessEqual(len(bank), bank.storage_capacity)
            self.assertLessEqual(bank.shadow_count, horizon)
            self.assertEqual(bank.active_count, min(anchor + 1, bank.capacity))
            self.assertEqual(bank.slots[0].slot_id, 0)
            self.assertEqual(bank.slots[0].state, "active")
            self.assertEqual([s.frame_index for s in bank.slots], sorted(s.frame_index for s in bank.slots))
        for slot_id, anchor in demoted_at.items():
            expected = list(range(anchor + 1, min(anchor + horizon + 1, 30)))
            self.assertEqual(shadow_reads.get(slot_id, []), expected)

    def test_protection_and_candidates_only_use_active_slots(self):
        bank = GateMemoryBank(4, keep_first=True, keep_recent=2, shadow_horizon=3)
        for _ in range(4):
            self.append(bank)
        self.assertEqual(self.append(bank, torch.tensor([0.0, 0.1, 0.9, 0.0])), 1)
        # Slot 1 is already Shadow. Slot 0 and newest Active slot 4 remain
        # protected, even though all three have lower scores than slot 2.
        self.assertEqual(self.append(bank, torch.tensor([0.0, 0.0, 0.2, 0.8, 0.0])), 2)
        self.assertEqual([s.slot_id for s in bank.slots if s.state == "shadow"], [1, 2])
        self.assertEqual(bank.active_count, 4)

    def test_zero_horizon_and_invalid_horizon(self):
        bank = GateMemoryBank(2, keep_first=False, keep_recent=0, shadow_horizon=0)
        for _ in range(5):
            self.append(bank)
        self.assertEqual([slot.slot_id for slot in bank.slots], [3, 4])
        self.assertEqual(bank.shadow_count, 0)
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "shadow_horizon"):
                GateMemoryBank(4, shadow_horizon=value)

    def test_shadow_age_starts_at_zero_and_advances_once_per_append(self):
        bank = GateMemoryBank(2, keep_first=False, keep_recent=0, shadow_horizon=3)
        for _ in range(3):
            self.append(bank)
        original = bank.slots[0]
        self.assertEqual(original.state, "shadow")
        for age in range(3):
            self.assertEqual(bank.shadow_age(original), age)
            self.assertEqual(bank.shadow_age(original), age)  # Reads cannot age it.
            self.append(bank)
        self.assertNotIn(original.slot_id, [slot.slot_id for slot in bank.slots])


class ShadowMemoryRetrievalTest(unittest.TestCase):
    def test_shadow_scaling_is_before_shared_softmax_and_gate_is_reused(self):
        retriever = gate_fixture.make_retriever()
        for block in retriever.perception_retrieval_blocks:
            with torch.no_grad():
                block.history_cross_attn.q.weight.zero_()
                block.history_cross_attn.q.bias.zero_()
                block.history_cross_attn.kv.weight.zero_()
                block.history_cross_attn.kv.bias.zero_()
        current = torch.randn(1, 4, 16)
        history = torch.randn(1, 1, 3, 16).expand(-1, 4, -1, -1)
        records = {"history_cross_attention_slot_weights": [], "slot_gate_training_log_gates": []}
        calls = []
        handle = retriever.slot_gate.register_forward_hook(lambda module, args, output: calls.append(output))
        output = retriever(
            global_cond=current, history_latents=history,
            history_mask=torch.ones(1, 4, dtype=torch.bool),
            history_shadow_mask=torch.tensor([[False, True, True, True]]),
            history_shadow_age=torch.tensor([[0, 0, 1, 2]]), record_data_dict=records,
        )
        handle.remove()
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(len(calls), 1)
        weights = records["history_cross_attention_slot_weights"][0]
        scales = torch.tensor([1.0, 0.01, 0.001, 0.0001])
        expected = scales / scales.sum()
        torch.testing.assert_close(weights, expected.expand_as(weights))
        raw, effective = retriever.latest_slot_gates, retriever.latest_effective_slot_gates
        torch.testing.assert_close(raw[:, 0], raw[:, 1])
        torch.testing.assert_close(effective[:, 0], raw[:, 0])
        torch.testing.assert_close(effective, raw * scales[None, :, None])
        torch.testing.assert_close(records["slot_gate_training_log_gates"][0], calls[0])

    def test_shadow_task_gradients_reach_raw_gate_and_slot_features(self):
        retriever = gate_fixture.make_retriever()
        history = torch.randn(1, 3, 4, 16, requires_grad=True)
        records = {"slot_gate_training_log_gates": []}
        output = retriever(
            global_cond=torch.randn(1, 4, 16), history_latents=history,
            history_mask=torch.tensor([[True, True, False]]),
            history_shadow_mask=torch.tensor([[False, True, False]]), record_data_dict=records,
            history_shadow_age=torch.tensor([[0, 2, 0]]),
        )
        raw_log_gates = records["slot_gate_training_log_gates"][0]
        gate_grad, feature_grad = torch.autograd.grad(output.square().sum(), (raw_log_gates, history))
        self.assertGreater(float(gate_grad[:, 1].abs().sum()), 0)
        self.assertGreater(float(feature_grad[:, 1].abs().sum()), 0)
        self.assertEqual(float(gate_grad[:, 2].abs().sum()), 0)
        self.assertTrue(torch.isfinite(gate_grad).all())
        self.assertIsNone(retriever.latest_effective_slot_gates.grad_fn)

    def test_shadow_masks_and_scale_are_validated(self):
        retriever = gate_fixture.make_retriever()
        inputs = dict(global_cond=torch.randn(1, 4, 16), history_latents=torch.randn(1, 2, 3, 16))
        with self.assertRaisesRegex(ValueError, "history_shadow_mask"):
            retriever(**inputs, history_shadow_mask=torch.ones(1, 3, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "valid history"):
            retriever(**inputs, history_mask=torch.tensor([[True, False]]),
                       history_shadow_mask=torch.tensor([[False, True]]))
        for ages in (torch.tensor([[0, -1]]), torch.tensor([[1, 0]]),
                     torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 2)):
            with self.subTest(ages=ages), self.assertRaisesRegex(ValueError, "history_shadow_age"):
                retriever(**inputs, history_shadow_mask=torch.tensor([[False, True]]),
                          history_shadow_age=ages)
        with self.assertRaisesRegex(ValueError, "requires history_shadow_mask"):
            retriever(**inputs, history_shadow_age=torch.zeros(1, 2, dtype=torch.long))


class ShadowMemoryPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        future_fixture.FuturePredictionModeTest.setUpClass()

    def setUp(self):
        self.fixture = future_fixture.FuturePredictionModeTest()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def make_config(self, mode="tokens", cached=True):
        cfg = self.fixture.make_config(mode, cached)
        network = cfg.workspace.model.denoising_network_partial
        network.max_history_len = 4
        network.shadow_horizon = 3
        network.shadow_gate_scale = 0.01
        cfg.workspace.model.memory_eviction.keep_recent = 2
        return cfg

    def chunk(self, batch, start, length, reset, final):
        chunk = {key: value[:, start:start + length + 1] for key, value in batch.items()}
        chunk.update(_streaming_supervised_length=length, _streaming_reset=reset, _streaming_final=final)
        return chunk

    def snapshot(self, bank):
        return tuple((slot.slot_id, slot.state, slot.expire_at, slot.frame_index) for slot in bank.slots)

    def test_training_chunks_and_online_share_shadow_lifecycle_and_decay(self):
        for mode in ("pooled", "tokens"):
            with self.subTest(mode=mode):
                policy = self.fixture.instantiate_policy(self.make_config(mode)).eval()
                batch = self.fixture.make_batch(anchors=13)
                reads, packed_reads = [], []
                pack_history = policy._pack_history

                def capture_history(banks, reference):
                    reads.append(tuple(self.snapshot(bank) for bank in banks))
                    packed = pack_history(banks, reference)
                    packed_reads.append(packed)
                    return packed

                with patch.object(policy, "_pack_history", side_effect=capture_history):
                    full_loss = policy.compute_loss(self.chunk(batch, 0, 12, True, False))["gate_selectivity"].detach()
                full_reads = list(reads)
                full_ages = [packed["history_shadow_age"].clone() for packed in packed_reads]
                reads.clear()
                packed_reads.clear()
                full_bank = self.snapshot(policy._streaming_training_state[0])
                chunk_losses = []
                with patch.object(policy, "_pack_history", side_effect=capture_history):
                    for start in range(0, 12, 3):
                        chunk_losses.append(policy.compute_loss(
                            self.chunk(batch, start, 3, start == 0, start == 9),
                        )["gate_selectivity"].detach())
                self.assertEqual(reads, full_reads)
                for full_age, packed in zip(full_ages, packed_reads):
                    torch.testing.assert_close(full_age, packed["history_shadow_age"])
                self.assertTrue(any(bool((age == 2).any()) for age in full_ages))
                self.assertEqual(len(reads), 12)
                # Each read sees only preceding valid anchors, even at a
                # chunk boundary; lookahead/current slots are never inserted early.
                for anchor, (snapshot,) in enumerate(reads):
                    ids = [row[0] for row in snapshot]
                    self.assertTrue(all(slot_id < anchor for slot_id in ids))
                    if anchor:
                        self.assertIn(anchor - 1, ids)
                reads.clear()
                packed_reads.clear()
                self.assertIsNone(policy._streaming_training_state)
                self.assertEqual(policy.latest_training_slot_ids[0], tuple(row[0] for row in full_bank))
                self.assertEqual(policy.latest_training_slot_states[0], tuple(row[1] for row in full_bank))
                torch.testing.assert_close(full_loss, torch.stack(chunk_losses).mean())
                calls = []
                handle = policy.denoising_network.memory_retriever.slot_gate.register_forward_hook(
                    lambda module, args, output: calls.append(output.detach()),
                )
                with patch.object(policy.future_feature_predictor, "forward", side_effect=AssertionError("inference predictor")), \
                     patch.object(policy.future_img_feature_encoder, "forward", side_effect=AssertionError("inference target encoder")), \
                     patch.object(policy, "_pack_history", side_effect=capture_history):
                    for anchor in range(12):
                        with torch.inference_mode():
                            output = policy.predict_action({
                                "head_camera_feature": batch["head_camera_feature"][:, anchor],
                                "agent_pos": batch["agent_pos"][:, anchor], "episode_idx": torch.tensor([0]),
                            })
                        self.assertTrue(torch.isfinite(output["action"]).all())
                handle.remove()
                self.assertEqual(reads, full_reads)
                for full_age, packed in zip(full_ages, packed_reads):
                    self.assertEqual(packed["history_mask"].shape[1], 7)
                    torch.testing.assert_close(full_age, packed["history_shadow_age"])
                bank = policy.online_memory_banks[0]
                self.assertEqual(self.snapshot(bank), full_bank)
                self.assertEqual(len(calls), 12)
                self.assertEqual((bank.active_count, bank.shadow_count), (4, 3))
                self.assertEqual((bank.storage_capacity, bank.shadow_horizon), (7, 3))
                snapshot = policy.latest_online_memory_query_dict[0]
                self.assertEqual(snapshot["active_slot_count"], 4)
                self.assertEqual(snapshot["shadow_slot_count"], 3)
                raw = torch.tensor(snapshot["slot_gates"])
                effective = torch.tensor(snapshot["slot_effective_gates"])
                scales = torch.tensor([
                    0.01 / 10 ** (snapshot["query_anchor_index"] + 3 - expiry - 1)
                    if state == "shadow" else 1.0
                    for state, expiry in zip(snapshot["slot_states"], snapshot["slot_expire_at"])
                ])
                torch.testing.assert_close(effective, raw * scales[:, None])
                self.assertTrue(any(expiry == snapshot["query_anchor_index"] for expiry in snapshot["slot_expire_at"]))
                policy.reset()
                self.assertEqual(policy.online_memory_banks, {})
                self.assertIsNone(policy.denoising_network.memory_retriever.latest_effective_slot_gates)
                del capture_history, pack_history, policy

    def test_regularizer_uses_only_active_raw_gates(self):
        policy = self.fixture.instantiate_policy(self.make_config())
        captured, raw_outputs = [], []
        def regularizer(log_gates, active_mask, capacity, min_std):
            captured.append((log_gates.detach(), active_mask.detach(), capacity, min_std))
            return gate_selectivity_penalty(log_gates, active_mask, capacity, min_std)
        handle = policy.denoising_network.memory_retriever.slot_gate.register_forward_hook(
            lambda module, args, output: raw_outputs.append(output.detach()),
        )
        with patch("imitation_learning.policies.history_denoising_policy.gate_selectivity_penalty", side_effect=regularizer):
            loss = policy.compute_loss(self.fixture.make_batch(anchors=10))["gate_selectivity"]
        handle.remove()
        expected = 0.0
        for (raw, mask, capacity, threshold), mlp_output in zip(captured, raw_outputs):
            torch.testing.assert_close(raw, mlp_output)
            self.assertLessEqual(int(mask.sum()), 4)
            if int(mask.sum()) == capacity:
                std = raw.exp()[0, mask[0]].std(dim=0, unbiased=False)
                expected += float((threshold - std).clamp_min(0).square().mean())
        self.assertTrue(any(int(mask.sum()) == 4 and mask.shape[1] == 7 for _, mask, _, _ in captured))
        torch.testing.assert_close(loss.detach(), torch.tensor(expected / 10))

    def test_shadow_future_gradients_survive_chunk_and_optimizer_boundaries(self):
        cfg = self.make_config()
        cfg.workspace.model.denoising_network_partial.max_history_len = 2
        cfg.workspace.model.memory_eviction.keep_recent = 0
        policy = self.fixture.instantiate_policy(cfg)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.001)
        batch = self.fixture.make_batch(anchors=10)
        retriever = policy.denoising_network.memory_retriever
        raw_outputs, shadow_masks = [], []
        gate_handle = retriever.slot_gate.register_forward_hook(
            lambda module, args, output: raw_outputs.append(output),
        )
        def before_read(module, args, kwargs):
            shadow_masks.append(kwargs["history_shadow_mask"].clone())
        read_handle = retriever.register_forward_pre_hook(before_read, with_kwargs=True)
        saw_shadow_gradient = False
        for start in (0, 3, 6):
            raw_outputs.clear()
            shadow_masks.clear()
            optimizer.zero_grad(set_to_none=True)
            losses = policy.compute_loss(self.chunk(batch, start, 3, start == 0, start == 6))
            selected = [(raw, mask) for raw, mask in zip(raw_outputs, shadow_masks) if mask.any()]
            if selected:
                gradients = torch.autograd.grad(losses["future_prediction"], [raw for raw, _ in selected], retain_graph=True)
                for gradient, (_, mask) in zip(gradients, selected):
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(float(gradient[mask].abs().sum()), 0)
                    saw_shadow_gradient = True
            total = losses["action"] + 0.05 * losses["future_prediction"] + 0.01 * losses["gate_selectivity"]
            total.backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in retriever.slot_gate.parameters()), 0)
            optimizer.step()
            if policy._streaming_training_state is not None:
                for slot in policy._streaming_training_state[0].slots:
                    self.assertIsNone(slot.latent.grad_fn)
                    self.assertFalse(slot.raw_features.requires_grad)
        gate_handle.remove()
        read_handle.remove()
        self.assertTrue(saw_shadow_gradient)
        self.assertIsNone(policy._streaming_training_state)

    def test_padding_and_lookahead_do_not_age_or_enter_shadow_bank(self):
        cfg = self.make_config()
        cfg.workspace.model.denoising_network_partial.max_history_len = 2
        cfg.workspace.model.memory_eviction.keep_recent = 0
        policy = self.fixture.instantiate_policy(cfg).eval()
        batch = self.fixture.make_batch(anchors=8)
        batch["entire_traj_is_padding"][:, 4:7] = True
        first = policy.compute_loss(self.chunk(batch, 0, 7, True, False))["gate_selectivity"].detach()
        bank = policy._streaming_training_state[0]
        expected = self.snapshot(bank)
        self.assertEqual(bank.next_anchor_index, 4)
        self.assertEqual((bank.active_count, bank.shadow_count), (2, 2))
        self.assertTrue(all(slot.slot_id < 4 for slot in bank.slots))
        batch["head_camera_feature"][:, 4:] += 100
        other = policy.compute_loss(self.chunk(batch, 0, 7, True, False))["gate_selectivity"].detach()
        self.assertEqual(self.snapshot(policy._streaming_training_state[0]), expected)
        torch.testing.assert_close(first, other)

    def test_default_eight_plus_eight_capacity_and_independent_episode_clocks(self):
        cfg = self.make_config()
        cfg.workspace.model.denoising_network_partial.max_history_len = 8
        cfg.workspace.model.denoising_network_partial.shadow_horizon = 8
        cfg.workspace.model.memory_eviction.keep_recent = 2
        policy = self.fixture.instantiate_policy(cfg).eval()
        batch = {key: value.repeat(2, *([1] * (value.ndim - 1)))
                 for key, value in self.fixture.make_batch(anchors=18).items()}
        batch["entire_traj_is_padding"][1, 8:] = True
        losses = policy.compute_loss(self.chunk(batch, 0, 18, True, False))
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        training_banks = policy._streaming_training_state
        expected = [self.snapshot(bank) for bank in training_banks]
        self.assertEqual([bank.next_anchor_index for bank in training_banks], [18, 8])
        self.assertEqual([(bank.active_count, bank.shadow_count) for bank in training_banks], [(8, 8), (8, 0)])
        self.assertEqual(policy.denoising_network.history_time_embedding.shape[1], 16)
        for anchor in range(18):
            with torch.inference_mode():
                policy.predict_action({
                    "head_camera_feature": batch["head_camera_feature"][:, anchor],
                    "agent_pos": batch["agent_pos"][:, anchor], "episode_idx": torch.tensor([4, 9]),
                    "entire_traj_is_padding": batch["entire_traj_is_padding"][:, anchor],
                })
        self.assertEqual([self.snapshot(policy.online_memory_banks[episode]) for episode in (4, 9)], expected)
        self.assertEqual([policy.online_memory_banks[episode].next_anchor_index for episode in (4, 9)], [18, 8])
        self.assertEqual([(policy.online_memory_banks[e].active_count, policy.online_memory_banks[e].shadow_count)
                          for e in (4, 9)], [(8, 8), (8, 0)])

    def test_disabled_window_hard_deletes_in_training_and_online_in_both_feature_modes(self):
        for mode in ("pooled", "tokens"):
            with self.subTest(mode=mode):
                cfg = self.make_config(mode)
                network_cfg = cfg.workspace.model.denoising_network_partial
                network_cfg.max_history_len = 8
                network_cfg.shadow_window = False
                policy = self.fixture.instantiate_policy(cfg).eval()
                batch = self.fixture.make_batch(anchors=13)
                reads = []
                original_pack = policy._pack_history

                def capture(banks, reference):
                    packed = original_pack(banks, reference)
                    self.assertEqual(packed["history_mask"].shape[1], 8)
                    self.assertFalse(bool(packed["history_shadow_mask"].any()))
                    self.assertFalse(bool(packed["history_shadow_age"].any()))
                    self.assertTrue(all(bank.shadow_count == 0 and len(bank) <= 8 for bank in banks))
                    reads.append(tuple(self.snapshot(bank) for bank in banks))
                    return packed

                with patch.object(policy, "_pack_history", side_effect=capture):
                    losses = policy.compute_loss(self.chunk(batch, 0, 12, True, False))
                self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                training_reads = list(reads)
                final_bank = self.snapshot(policy._streaming_training_state[0])
                self.assertEqual(len(final_bank), 8)
                reads.clear()
                with patch.object(policy, "_pack_history", side_effect=capture):
                    for anchor in range(12):
                        with torch.inference_mode():
                            output = policy.predict_action({
                                "head_camera_feature": batch["head_camera_feature"][:, anchor],
                                "agent_pos": batch["agent_pos"][:, anchor], "episode_idx": torch.tensor([0]),
                            })
                        self.assertTrue(torch.isfinite(output["action"]).all())
                self.assertEqual(reads, training_reads)
                bank = policy.online_memory_banks[0]
                self.assertEqual(self.snapshot(bank), final_bank)
                self.assertEqual((bank.storage_capacity, bank.shadow_horizon), (8, 0))
                snapshot = policy.latest_online_memory_query_dict[0]
                torch.testing.assert_close(torch.tensor(snapshot["slot_gates"]),
                                           torch.tensor(snapshot["slot_effective_gates"]))
                del capture, original_pack, policy

    def test_shadow_window_checkpoint_default_and_flag_preserve_parameter_shapes(self):
        cfg = self.make_config()
        network_cfg = cfg.workspace.model.denoising_network_partial
        policy = self.fixture.instantiate_policy(cfg)
        state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        del policy
        with open_dict(network_cfg):
            del network_cfg["shadow_window"]
        policy = self.fixture.instantiate_policy(cfg)
        policy.load_state_dict(state, strict=True)
        self.assertTrue(policy.denoising_network.shadow_window)
        self.assertEqual(policy._new_memory_bank().shadow_horizon, 3)
        del policy
        with open_dict(network_cfg):
            network_cfg.shadow_window = False
        policy = self.fixture.instantiate_policy(cfg)
        policy.load_state_dict(state, strict=True)
        self.assertEqual(policy._new_memory_bank().shadow_horizon, 0)
        self.assertEqual(policy._new_memory_bank().storage_capacity, 4)

    def test_raw_image_training_uses_shadow_in_both_feature_modes(self):
        for mode in ("pooled", "tokens"):
            with self.subTest(mode=mode):
                policy = self.fixture.instantiate_policy(self.make_config(mode, cached=False))
                losses = policy.compute_loss(self.fixture.make_batch(cached=False, anchors=7))
                total = losses["action"] + 0.05 * losses["future_prediction"] + 0.01 * losses["gate_selectivity"]
                self.assertTrue(torch.isfinite(total))
                total.backward()
                self.assertIn("shadow", policy.latest_training_slot_states[0])
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.denoising_network.memory_retriever.slot_gate.parameters()))
                del policy

    def test_projection_accepts_physical_capacity_and_rejects_overflow(self):
        policy = self.fixture.instantiate_policy(self.make_config())
        network = policy.denoising_network
        self.assertEqual(tuple(network.history_time_embedding.shape), (1, 7, 32))
        data = {
            "noisy_action": torch.randn(1, 16, 14), "step": torch.zeros(1, dtype=torch.long),
            "global_cond": torch.randn(1, 16, 8), "local_cond": torch.randn(1, 1, 8),
            "history_img_features": torch.randn(1, 7, 16, 8),
            "history_mask": torch.ones(1, 7, dtype=torch.bool),
            "history_shadow_mask": torch.tensor([[False] * 4 + [True] * 3]),
        }
        inputs = network.prepare_single_memory_retrieval(data)
        self.assertEqual(inputs["history_latents"].shape[1], 7)
        self.assertTrue(torch.isfinite(network.retrieve_decision_memory(inputs)).all())
        data["history_img_features"] = torch.randn(1, 8, 16, 8)
        with self.assertRaisesRegex(ValueError, "storage capacity"):
            network.prepare_single_memory_retrieval(data)

    def test_shadow_configuration_defaults_and_validation(self):
        cfg = OmegaConf.load(future_fixture.CONFIG_DIR / "workspace/policy/denoising_network/memory_transformer.yaml")
        # The horizon is an experiment setting, rather than a fixed model size.
        self.assertIsInstance(cfg.shadow_horizon, int)
        self.assertGreaterEqual(cfg.shadow_horizon, 0)
        self.assertTrue(math.isfinite(cfg.shadow_gate_scale) and 0 < cfg.shadow_gate_scale <= 1)
        self.assertIs(cfg.shadow_window, True)
        for field, value in (("shadow_window", "false"), ("shadow_window", 0), ("shadow_horizon", -1), ("shadow_horizon", True), ("shadow_gate_scale", 0), ("shadow_gate_scale", math.nan), ("shadow_gate_scale", 1.1)):
            with self.subTest(field=field, value=value):
                cfg = self.make_config()
                cfg.workspace.model.denoising_network_partial[field] = value
                with self.assertRaises(hydra.errors.InstantiationException):
                    self.fixture.instantiate_policy(cfg)

    def test_checkpoint_config_without_shadow_settings_keeps_original_shape(self):
        cfg = self.make_config()
        network_cfg = cfg.workspace.model.denoising_network_partial
        network_cfg.shadow_horizon = 0
        policy = self.fixture.instantiate_policy(cfg)
        state = policy.state_dict()
        del policy
        with open_dict(network_cfg):
            del network_cfg["shadow_horizon"]
            del network_cfg["shadow_gate_scale"]
        policy = self.fixture.instantiate_policy(cfg)
        policy.load_state_dict(state, strict=True)
        self.assertEqual(policy.denoising_network.shadow_horizon, 0)
        self.assertEqual(tuple(policy.denoising_network.history_time_embedding.shape), (1, 4, 32))
        self.assertEqual(policy._new_memory_bank().storage_capacity, 4)


if __name__ == "__main__":
    unittest.main()
