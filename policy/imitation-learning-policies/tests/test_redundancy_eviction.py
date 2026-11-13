import gc
import unittest
from unittest.mock import patch

import torch

from imitation_learning.models.encoders.image_encoders import SharedModelManager
from imitation_learning.models.memory.gate_memory import GateMemoryBank
from imitation_learning.models.memory.redundancy_eviction import (
    RedundancyGateMemoryBank,
    token_mean_cosine,
)
import test_future_prediction_mode as future_fixture


MODULE = "imitation_learning.models.memory.redundancy_eviction"


class RedundancyEvictionTest(unittest.TestCase):
    def fill(self, bank, features):
        for i, feature in enumerate(features):
            bank.append(feature, i * 10, torch.ones(len(bank)), raw_features=feature)

    def features(self, count=5):
        vectors = [[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1.]]
        return [torch.tensor(vectors[i % 5]).repeat(2, 1) for i in range(count)]

    def test_similarity_averages_corresponding_token_cosines_in_fp32(self):
        left = torch.tensor([[1., 0.], [0., 2.]], requires_grad=True)
        right = torch.tensor([[1., 0.], [0., -1.]], requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            similarity = token_mean_cosine(left.bfloat16(), right.bfloat16())
        self.assertEqual(similarity.dtype, torch.float32)
        self.assertEqual(float(similarity), 0.)
        self.assertFalse(similarity.requires_grad)
        # Pooling tokens first gives a different, nonzero cosine.
        self.assertGreater(abs(float(torch.nn.functional.cosine_similarity(
            left.mean(0), right.mean(0), dim=0,
        ))), 0.1)
        self.assertEqual(float(token_mean_cosine(torch.zeros(2, 3), torch.zeros(2, 3))), 0.)
        with self.assertRaises(ValueError):
            token_mean_cosine(left, right[:1])

    def test_combined_importance_changes_eviction_and_respects_protection(self):
        bank = RedundancyGateMemoryBank(5, keep_recent=1, redundancy_alpha=1.)
        self.fill(bank, self.features())
        gates = torch.tensor([1., .55, .5, .4, 1.], requires_grad=True)
        combined = bank._eviction_scores(gates)
        # Edges: [1,1,0,1]; neighbor-weighted redundancies below.
        redundancy = torch.tensor([.55, .75, .275, .5, .4])
        torch.testing.assert_close(combined, gates.detach() + 1 - redundancy)
        self.assertFalse(combined.requires_grad)
        original = gates.detach().clone()
        self.assertEqual(bank.append(torch.ones(2, 2), 50, gates), 1)
        torch.testing.assert_close(gates, original)
        self.assertEqual([slot.slot_id for slot in bank.slots], [0, 2, 3, 4, 5])

    def test_cache_computes_each_edge_once_and_reuses_it_for_new_gates(self):
        bank = RedundancyGateMemoryBank(5, keep_recent=1, redundancy_alpha=.5)
        self.fill(bank, self.features())
        with patch(f"{MODULE}.token_mean_cosine", wraps=token_mean_cosine) as cosine:
            first = bank._eviction_scores(torch.ones(5))
            second = bank._eviction_scores(torch.full((5,), .2))
            self.assertEqual(cosine.call_count, 1)
            self.assertEqual(cosine.call_args.args[0].shape[0], 4)
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(set(bank._similarity_cache), {(0, 1), (1, 2), (2, 3), (3, 4)})
        self.assertTrue(all(not value.requires_grad for value in bank._similarity_cache.values()))

    def test_demotion_skips_shadow_and_only_computes_new_active_edges(self):
        bank = RedundancyGateMemoryBank(5, keep_recent=1, shadow_horizon=3, redundancy_alpha=1.)
        self.fill(bank, self.features())
        with patch(f"{MODULE}.token_mean_cosine", wraps=token_mean_cosine) as cosine:
            self.assertEqual(bank.append(torch.ones(2, 2), 50, torch.tensor([1., .55, .5, .4, 1.])), 1)
            self.assertEqual(set(bank._similarity_cache), {(2, 3), (3, 4)})
            # Shadow 1 lies between Active 0 and 2 in the retained list.
            gates = torch.tensor([.8, .001, .3, .4, .5, .6])
            scores = bank._eviction_scores(gates)
            self.assertEqual(cosine.call_count, 2)
            self.assertEqual(cosine.call_args.args[0].shape[0], 2)  # (0,2), (4,5)
            self.assertEqual(set(bank._similarity_cache), {(0, 2), (2, 3), (3, 4), (4, 5)})
            changed = gates.clone()
            changed[1] = .99
            torch.testing.assert_close(scores[[0, 2, 3, 4, 5]], bank._eviction_scores(changed)[[0, 2, 3, 4, 5]])
            self.assertEqual(cosine.call_count, 2)

    def test_reprojection_invalidates_cache_but_detach_preserves_it(self):
        bank = RedundancyGateMemoryBank(5, keep_recent=1, redundancy_alpha=1.)
        self.fill(bank, self.features())
        gates = torch.tensor([1., .55, .5, .4, 1.])
        old = bank._eviction_scores(gates)
        bank.detach()
        self.assertEqual(len(bank._similarity_cache), 4)
        projector = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            projector.weight.copy_(torch.tensor([[1., 1.], [0., 0.]]))
        bank.reproject(projector)
        self.assertEqual(bank._similarity_cache, {})
        with patch(f"{MODULE}.token_mean_cosine", wraps=token_mean_cosine) as cosine:
            new = bank._eviction_scores(gates)
            self.assertEqual(cosine.call_args.args[0].shape[0], 4)
        self.assertFalse(torch.equal(old, new))
        # Scoring must not destroy the task gradient through projected latents.
        sum(slot.latent.sum() for slot in bank.slots).backward()
        self.assertGreater(float(projector.weight.grad.abs().sum()), 0.)

    def test_append_preserves_incoming_task_gradient(self):
        bank = RedundancyGateMemoryBank(4, keep_recent=1, redundancy_alpha=1.)
        self.fill(bank, self.features(4))
        source = torch.randn(2, 2, requires_grad=True)
        bank.append(source * 2, 40, torch.ones(4))
        bank.slots[-1].latent.sum().backward()
        torch.testing.assert_close(source.grad, torch.full_like(source, 2.))

    def test_zero_alpha_exactly_matches_original_lifecycle(self):
        old = GateMemoryBank(5, keep_recent=2, shadow_horizon=3)
        new = RedundancyGateMemoryBank(5, keep_recent=2, shadow_horizon=3, redundancy_alpha=0.)
        generator = torch.Generator().manual_seed(4)
        with patch(f"{MODULE}.token_mean_cosine", side_effect=AssertionError("disabled")):
            for i in range(30):
                gates = torch.rand(len(old), generator=generator)
                latent = torch.randn(2, 3, generator=generator)
                self.assertEqual(old.append(latent, i * 10, gates), new.append(latent, i * 10, gates))
                self.assertEqual(
                    [(slot.slot_id, slot.state, slot.expire_at) for slot in old.slots],
                    [(slot.slot_id, slot.state, slot.expire_at) for slot in new.slots],
                )
        self.assertEqual(new._similarity_cache, {})

    def test_long_stream_has_bounded_cache_and_exact_shadow_lifetime(self):
        bank = RedundancyGateMemoryBank(6, shadow_horizon=3, redundancy_alpha=.7)
        demoted_at, shadow_reads = {}, {}
        generator = torch.Generator().manual_seed(9)
        for anchor in range(40):
            for slot in bank.slots:
                if slot.state == "shadow":
                    shadow_reads.setdefault(slot.slot_id, []).append(anchor)
            demoted = bank.append(torch.randn(3, 4, generator=generator), anchor * 10,
                                  torch.rand(len(bank), generator=generator))
            if demoted is not None:
                demoted_at[demoted] = anchor
            self.assertLessEqual(len(bank._similarity_cache), bank.capacity - 1)
            self.assertLessEqual(len(bank), bank.storage_capacity)
            active_ids = [slot.slot_id for slot in bank.slots if slot.state == "active"]
            self.assertEqual(active_ids[0], 0)
            self.assertEqual(active_ids[-min(anchor + 1, 3):], list(range(max(0, anchor - 2), anchor + 1)))
        for slot_id, anchor in demoted_at.items():
            self.assertEqual(shadow_reads.get(slot_id, []), list(range(anchor + 1, min(anchor + 4, 40))))

    def test_invalid_alpha_is_rejected(self):
        for alpha in (-.1, True, float("nan"), float("inf"), "0.1"):
            with self.subTest(alpha=alpha), self.assertRaisesRegex(ValueError, "redundancy_alpha"):
                RedundancyGateMemoryBank(5, redundancy_alpha=alpha)


class RedundancyPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        future_fixture.FuturePredictionModeTest.setUpClass()

    def setUp(self):
        self.fixture = future_fixture.FuturePredictionModeTest()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def test_config_default_and_old_checkpoint_without_field(self):
        cfg = self.fixture.make_config()
        policy = self.fixture.instantiate_policy(cfg).eval()
        self.assertEqual(policy._new_memory_bank().redundancy_alpha, .1)
        state = policy.state_dict()
        del policy
        del cfg.workspace.model.memory_eviction.redundancy_alpha
        legacy = self.fixture.instantiate_policy(cfg).eval()
        self.assertEqual(legacy._new_memory_bank().redundancy_alpha, 0.)
        legacy.load_state_dict(state, strict=True)

    def test_training_and_online_choose_identical_active_and_shadow_slots(self):
        cfg = self.fixture.make_config("tokens")
        cfg.workspace.model.denoising_network_partial.max_history_len = 5
        cfg.workspace.model.denoising_network_partial.shadow_horizon = 3
        cfg.workspace.model.memory_eviction.keep_recent = 2
        cfg.workspace.model.memory_eviction.redundancy_alpha = .7
        policy = self.fixture.instantiate_policy(cfg).eval()
        batch = self.fixture.make_batch(anchors=13)
        traces = []
        original = RedundancyGateMemoryBank.append

        def record(bank, *args, **kwargs):
            result = original(bank, *args, **kwargs)
            traces.append(tuple((slot.slot_id, slot.state, slot.expire_at) for slot in bank.slots))
            return result

        with patch.object(RedundancyGateMemoryBank, "append", record):
            losses = policy.compute_loss(batch)
        sum(losses.values()).backward()
        training_traces = traces[:]
        traces.clear()
        with patch.object(RedundancyGateMemoryBank, "append", record):
            for anchor in range(13):
                with torch.inference_mode():
                    policy.predict_action({
                        "head_camera_feature": batch["head_camera_feature"][:, anchor],
                        "agent_pos": batch["agent_pos"][:, anchor],
                        "episode_idx": torch.tensor([0]),
                    })
        self.assertEqual(traces, training_traces)
        self.assertTrue(any(state == "shadow" for _, state, _ in traces[-1]))
        policy.reset()
        self.assertEqual(policy.online_memory_banks, {})


if __name__ == "__main__":
    unittest.main()
