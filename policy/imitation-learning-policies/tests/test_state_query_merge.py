import copy
import gc
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf, open_dict
import torch

from imitation_learning.models.common.state_query_merger import StateQueryPatchMerger
from imitation_learning.models.encoders.image_encoders import SharedModelManager
import test_future_prediction_mode as fixture


class StateQueryMergerTest(unittest.TestCase):
    def test_groups_and_masks_keep_spatial_and_view_boundaries(self):
        merger = StateQueryPatchMerger(1, 8, 16, 2, 2)
        x = torch.arange(32).reshape(1, 32, 1)
        expected = torch.tensor([
            [0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15],
            [16, 17, 20, 21], [18, 19, 22, 23], [24, 25, 28, 29], [26, 27, 30, 31],
        ]).unsqueeze(0).unsqueeze(-1)
        torch.testing.assert_close(merger._group(x), expected)
        mask = torch.ones(1, 32, dtype=torch.bool)
        mask[0, 5] = False
        self.assertEqual(merger.merge_mask(mask).tolist(), [[False, True, True, True, True, True, True, True]])

    def test_identity_initialization_and_detached_images_train_queries_and_modulation(self):
        for size, views in ((4, 1), (8, 3)):
            with self.subTest(size=size, views=views):
                merger = StateQueryPatchMerger(8, 32, 256, size, 4)
                images = torch.randn(2, 3, views * 256, 8).detach()
                states = torch.randn(2, 3, 1, 32)
                queries = []
                handle = merger.attention.register_forward_pre_hook(
                    lambda module, args: queries.append(args[0].detach().clone())
                )
                result = merger(images, states)
                other = merger(images, states + 10)
                handle.remove()
                self.assertEqual(result.shape, (2, 3, views * (16 // size)**2, 32))
                torch.testing.assert_close(result, other, rtol=0, atol=0)
                expected_queries = merger.query_tokens.repeat(views, 1).expand(2, 3, -1, -1).reshape(-1, 1, 32)
                torch.testing.assert_close(queries[0], expected_queries, rtol=0, atol=0)
                result.square().mean().backward()
                self.assertIsNone(images.grad)
                self.assertGreater(float(merger.query_tokens.grad.abs().sum()), 0)
                self.assertGreater(float(merger.state_modulation[-1].weight.grad.abs().sum()), 0)

    def test_trained_modulation_changes_reads_without_a_state_residual(self):
        merger = StateQueryPatchMerger(8, 32, 256, 4, 4)
        with torch.no_grad():
            merger.state_modulation[-1].weight.normal_(std=0.1)
        images = torch.randn(2, 256, 8)
        states = torch.randn(2, 1, 32, requires_grad=True)
        first = merger(images, states)
        second = merger(images, -states)
        self.assertGreater(float((first - second).abs().max()), 1e-5)
        first.square().mean().backward()
        self.assertGreater(float(states.grad.abs().sum()), 0)
        # Zero image values produce zero output even with nonzero queries and
        # modulation. A hidden q + attention residual would violate this.
        with torch.no_grad():
            merger.image_projector.bias.zero_()
            merger.attention.kv.bias.zero_()
            merger.attention.linear.bias.zero_()
        zero = merger(torch.zeros_like(images), states.detach())
        torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)

    def test_attention_is_local_and_views_are_independent(self):
        merger = StateQueryPatchMerger(8, 32, 256, 8, 4)
        images = torch.randn(2, 3 * 256, 8)
        state = torch.randn(2, 1, 32)
        original = merger(images, state)
        changed = images.clone()
        changed[:, 2 * 256, 0] += 10
        result = merger(changed, state)
        difference = (original - result).abs().amax(dim=-1)
        self.assertTrue(torch.equal(difference[:, :8], torch.zeros_like(difference[:, :8])))
        self.assertGreater(float(difference[:, 8].max()), 0)
        self.assertTrue(torch.equal(difference[:, 9:], torch.zeros_like(difference[:, 9:])))

    def test_masked_states_are_ignored_and_bad_shapes_rejected(self):
        merger = StateQueryPatchMerger(8, 32, 256, 4, 4)
        with torch.no_grad():
            merger.state_modulation[-1].weight.normal_(std=0.1)
        images, states = torch.randn(2, 256, 8), torch.randn(2, 2, 32)
        mask = torch.tensor([[True, False], [False, False]])
        expected = merger(images, states, mask)
        actual = merger(images, states.masked_fill(~mask[..., None], float('nan')), mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(actual).all())
        with self.assertRaisesRegex(ValueError, 'complete'):
            merger(torch.zeros(2, 255, 8), states)
        with self.assertRaisesRegex(ValueError, 'state tokens'):
            merger(images, torch.zeros(3, 1, 32))
        with self.assertRaisesRegex(ValueError, 'state_mask'):
            merger(images, states, torch.ones(2, 1))


class StateQueryMergePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.FuturePredictionModeTest.setUpClass()

    def setUp(self):
        self.helper = fixture.FuturePredictionModeTest()

    def tearDown(self):
        gc.collect()
        SharedModelManager.reset()

    def config(self, mode='pooled', cached=True, views=1, size=4):
        cfg = self.helper.make_config(mode, cached, merge_size=size)
        cfg.visual_encoder.state_query_merge = True
        cfg.workspace.train_dataset.image_keys = ['head_camera', 'left_camera', 'right_camera'][:views]
        cfg.workspace.model.denoising_network_partial.state_as_dit_token = True
        return cfg

    def batch(self, cfg, anchors=3):
        cached = cfg.workspace.train_dataset.use_cached_image_features
        batch = self.helper.make_batch(cached=cached, anchors=anchors)
        suffix = '_feature' if cached else ''
        for name in cfg.workspace.train_dataset.image_keys[1:]:
            batch[name + suffix] = batch['head_camera' + suffix].clone() + 0.1
        return batch

    def test_cached_online_multiview_and_future_modes_train_and_infer(self):
        for cached, mode, views, size in ((True, 'pooled', 1, 4), (False, 'pooled', 1, 8),
                                          (True, 'tokens', 1, 4), (True, 'pooled', 3, 8)):
            with self.subTest(cached=cached, mode=mode, views=views, size=size):
                cfg = self.config(mode, cached, views, size)
                policy = self.helper.instantiate_policy(cfg)
                network = policy.denoising_network
                count = views * (16 // size)**2
                self.assertTrue(network.state_query_merge)
                self.assertFalse(network.patch_merge)
                self.assertEqual(policy.global_cond_encoder.token_num, 256 * views)
                self.assertEqual(network.global_cond_token_num, count)
                self.assertEqual(network.history_img_features_token_num, count)
                self.assertEqual(policy.future_img_feature_encoder.token_num, views if mode == 'pooled' else count)
                self.assertEqual(cfg.workspace.train_dataset.cached_image_feature_kind, 'patches')
                self.assertIsNone(network.history_img_features_projector)
                batch = self.batch(cfg)
                _, targets = policy._encode_input_multi_traj(batch)
                expected_targets = targets['future_features'].detach().clone()
                losses = policy.compute_loss(batch)
                total = losses['action'] + 0.05 * losses['future_prediction']
                total.backward()
                merger = network.global_cond_projector
                self.assertGreater(float(merger.query_tokens.grad.abs().sum()), 0)
                self.assertGreater(float(merger.state_modulation[-1].weight.grad.abs().sum()), 0)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None))
                for backbone in policy.shared_model_manager.vit_models.values():
                    self.assertTrue(all(p.grad is None for p in backbone.parameters()))
                with torch.no_grad():
                    merger.query_tokens.add_(0.1)
                _, new_targets = policy._encode_input_multi_traj(batch)
                torch.testing.assert_close(new_targets['future_features'], expected_targets, rtol=0, atol=0)
                policy.eval()
                policy.reset()
                seen_images, dit_images, retrieval_queries = [], [], []
                handles = [
                    network.memory_retriever.register_forward_pre_hook(
                        lambda module, args, kwargs: seen_images.append(kwargs['global_cond'].clone()), with_kwargs=True),
                    network.blocks[0].cross_attn.register_forward_pre_hook(
                        lambda module, args: dit_images.append(args[1].clone())),
                    network.memory_retriever.perception_retrieval_blocks[0].register_forward_pre_hook(
                        lambda module, args, kwargs: retrieval_queries.append((len(seen_images) - 1, kwargs['query'].clone())), with_kwargs=True),
                ]
                suffix = '_feature' if cached else ''
                with patch.object(policy.future_feature_predictor, 'forward', side_effect=AssertionError('online future predictor')):
                    for anchor in range(2):
                        obs = {name + suffix: batch[name + suffix][:, anchor] for name in cfg.workspace.train_dataset.image_keys}
                        obs.update(agent_pos=batch['agent_pos'][:, anchor], episode_idx=torch.tensor([0]))
                        with torch.inference_mode():
                            output = policy.predict_action(obs)
                        self.assertEqual(output['action'].shape, (1, 16, 14))
                for handle in handles:
                    handle.remove()
                self.assertEqual(len(seen_images), 2)
                for a, b in zip(seen_images, dit_images):
                    self.assertEqual(a.shape, (1, count, 32))
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                for anchor_index, query in retrieval_queries:
                    torch.testing.assert_close(query, seen_images[anchor_index], rtol=0, atol=0)
                self.assertTrue(all(slot.latent.shape == (count, 32) and slot.raw_state is None and slot.raw_features is None
                                    for slot in policy.online_memory_banks[0].slots))
                del policy

    def test_current_and_history_share_pooling_and_cached_online_features_match(self):
        policy = self.helper.instantiate_policy(self.config(cached=False, size=8))
        batch = self.batch(self.config(cached=False, size=8))
        images = {'head_camera': batch['head_camera']}
        raw = policy.global_cond_encoder(images)
        cls = policy.future_img_feature_encoder(images)
        cache = torch.cat([cls, raw], dim=-2).unsqueeze(-3)
        cached = policy.global_cond_encoder({'head_camera_feature': cache})
        torch.testing.assert_close(raw, cached, rtol=0, atol=0)
        states = policy.local_cond_encoder({'agent_pos': batch['agent_pos'].reshape(-1, 1, 14)})
        states = states.reshape(1, 3, 1, 8)
        network = policy.denoising_network
        current = network._project_global_cond(raw, states)
        historical = network.project_history_image_features(cached, states)
        torch.testing.assert_close(current, historical, rtol=0, atol=0)
        self.assertFalse(any('history_img_features_projector.' in k for k in policy.state_dict()))
        with self.assertRaisesRegex(ValueError, "own state"):
            network.project_history_image_features(raw)

    def test_streaming_reencodes_own_raw_states_and_trains_from_retained_history(self):
        cfg = self.config(size=8)
        cfg.workspace.model.denoising_network_partial.max_history_len = 4
        cfg.workspace.model.memory_eviction.keep_recent = 2
        policy = self.helper.instantiate_policy(cfg)
        batch = self.batch(cfg, anchors=9)
        for start in (0, 4):
            chunk = {key: value[:, start:start + 5] for key, value in batch.items()}
            chunk.update(_streaming_supervised_length=4, _streaming_reset=start == 0, _streaming_final=False)
            losses = policy.compute_loss(chunk)
            (losses['action'] + 0.05 * losses['future_prediction']).backward()
            bank = policy._streaming_training_state[0]
            for slot in bank.slots:
                torch.testing.assert_close(slot.raw_state['agent_pos'], batch['agent_pos'][0, slot.frame_index // 10], rtol=0, atol=0)
                self.assertEqual(slot.raw_features.shape, (256, 8))
                self.assertFalse(slot.raw_state['agent_pos'].requires_grad)
                self.assertFalse(slot.latent.requires_grad)
            policy.zero_grad(set_to_none=True)
            if start == 0:
                old_latent = bank.slots[0].latent.clone()
                with torch.no_grad():
                    policy.denoising_network.global_cond_projector.state_modulation[-1].weight.normal_(std=0.1)
                    policy.local_cond_encoder.encoder_dict['agent_pos'].model.model.weight.add_(0.1)
                bank.reproject(policy._reproject_state_query_history)
                self.assertGreater(float((bank.slots[0].latent - old_latent).abs().max()), 0)
                slot = bank.slots[0]
                expected = policy._reproject_state_query_history(
                    slot.raw_features[None], {key: value[None] for key, value in slot.raw_state.items()},
                )[0]
                torch.testing.assert_close(slot.latent, expected)
                sum(s.latent.square().mean() for s in bank.slots).backward()
                self.assertGreater(float(policy.local_cond_encoder.encoder_dict['agent_pos'].model.model.weight.grad.abs().sum()), 0)
                bank.detach()
                policy.zero_grad(set_to_none=True)

    def test_training_and_online_keep_identical_shadow_membership_and_clock(self):
        cfg = self.config(size=8)
        network_cfg = cfg.workspace.model.denoising_network_partial
        network_cfg.max_history_len = 4
        network_cfg.shadow_horizon = 2
        network_cfg.shadow_gate_scale = 0.5
        cfg.workspace.model.memory_eviction.keep_recent = 2
        policy = self.helper.instantiate_policy(cfg)
        batch = self.batch(cfg, anchors=13)
        for start in (0, 4, 8):
            chunk = {key: value[:, start:start + 5] for key, value in batch.items()}
            chunk.update(_streaming_supervised_length=4, _streaming_reset=start == 0, _streaming_final=False)
            losses = policy.compute_loss(chunk)
            (losses['action'] + 0.05 * losses['future_prediction'] + 0.01 * losses['gate_selectivity']).backward()
            policy.zero_grad(set_to_none=True)
        bank = policy._streaming_training_state[0]
        expected = [(s.slot_id, s.state, s.expire_at) for s in bank.slots]
        self.assertEqual(bank.active_count, 4)
        self.assertEqual(bank.shadow_count, 2)
        self.assertTrue(all(s.raw_state is not None for s in bank.slots))
        policy.eval()
        for anchor in range(12):
            with torch.inference_mode():
                policy.predict_action({
                    'head_camera_feature': batch['head_camera_feature'][:, anchor],
                    'agent_pos': batch['agent_pos'][:, anchor], 'episode_idx': torch.tensor([0]),
                })
        self.assertEqual([(s.slot_id, s.state, s.expire_at) for s in policy.online_memory_banks[0].slots], expected)

    def test_missing_flag_preserves_legacy_state_layout_and_outputs(self):
        cfg = self.helper.make_config()
        policy = self.helper.instantiate_policy(cfg)
        state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        expected = {k: v.detach().clone() for k, v in policy.compute_loss(self.helper.make_batch()).items()}
        del policy
        legacy = copy.deepcopy(cfg)
        with open_dict(legacy.visual_encoder):
            del legacy.visual_encoder.state_query_merge
        with open_dict(legacy.workspace.model.denoising_network_partial):
            del legacy.workspace.model.denoising_network_partial.state_query_merge
        policy = self.helper.instantiate_policy(legacy)
        self.assertFalse(policy.denoising_network.state_query_merge)
        policy.load_state_dict(state, strict=True)
        for k, v in policy.state_dict().items():
            torch.testing.assert_close(v, state[k], rtol=0, atol=0)
        for k, v in policy.compute_loss(self.helper.make_batch()).items():
            torch.testing.assert_close(v, expected[k], rtol=0, atol=0)

    def test_checkpoint_roundtrip_and_conflicting_flags(self):
        cfg = self.config(size=8)
        policy = self.helper.instantiate_policy(cfg)
        with torch.no_grad():
            policy.denoising_network.global_cond_projector.state_modulation[-1].weight.normal_(std=0.1)
        data = torch.randn(2, 256, 8)
        states = torch.randn(2, 1, 8)
        expected = policy.denoising_network.project_history_image_features(data, states).detach()
        state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        del policy
        restored = self.helper.instantiate_policy(OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=False)))
        restored.load_state_dict(state, strict=True)
        torch.testing.assert_close(restored.denoising_network.project_history_image_features(data, states), expected, rtol=0, atol=0)
        del restored
        cfg.visual_encoder.patch_merge = True
        with self.assertRaisesRegex(Exception, 'mutually exclusive'):
            self.helper.instantiate_policy(cfg)


if __name__ == '__main__':
    unittest.main()
