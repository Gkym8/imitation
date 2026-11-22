import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import hydra
import numpy as np
import torch
from omegaconf import open_dict

import policy_fixture as fixture
from deploy_policy import eval as deploy_eval
from imitation_learning.models.encoders.image_encoders import SharedModelManager
from imitation_learning.utils.config_utils import compose_hydra_config


class ActionHistoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.PolicyFixture.setUpClass()

    def tearDown(self):
        SharedModelManager.reset()

    @staticmethod
    def configure_legacy_policy(cfg, enabled):
        with open_dict(cfg.workspace.model):
            cfg.workspace.model._target_ = (
                "imitation_learning.policies.action_history_policy.ActionHistoryPolicy"
            )
            cfg.workspace.model.action_history = enabled
            cfg.workspace.model.action_history_num_layers = 3

    def make(self):
        helper = fixture.PolicyFixture()
        cfg = helper.make_config(merge_size=[8, 4])
        cfg.visual_encoder.state_query_merge = True
        cfg.workspace.model.denoising_network_partial.shadow_horizon = 2
        cfg.workspace.model.denoising_network_partial.state_as_dit_token = True
        cfg.workspace.model.denoising_network_partial.record_data_entries = [
            "history_cross_attention_slot_weights", "history_cross_attention_frame_logits",
        ]
        self.configure_legacy_policy(cfg, enabled=True)
        cfg.workspace.model.memory_eviction.redundancy_alpha = 0.4
        cfg.workspace.model.noise_scheduler.inference_step_num = 2
        return helper.instantiate_policy(cfg), helper.make_batch(anchors=6)

    def test_training_config_uses_visual_history_policy(self):
        with hydra.initialize_config_dir(config_dir=str(fixture.CONFIG_DIR), version_base=None):
            cfg = compose_hydra_config("rmbench", "diffusion_memory_transformer")
        self.assertNotIn("action_history", cfg.workspace.model)
        self.assertNotIn("action_history_num_layers", cfg.workspace.model)
        self.assertTrue(cfg.workspace.model._target_.endswith(".HistoryDenoisingPolicy"))

    def test_streaming_causality_and_gradients(self):
        policy, batch = self.make()
        encoder = policy.action_history_encoder
        self.assertEqual(len(encoder.sequence_layers), 2)
        network = policy.denoising_network
        seen = []

        def inspect_tokens(module, args, kwargs):
            x = kwargs["x"]
            self.assertEqual(x.shape[1:], (19, 32))
            seen.append(x[:, 2].detach().clone())

        with torch.no_grad():
            state = encoder.initial_state(1)
            expected = [state[0]]
            for anchor in range(4):
                state = encoder(state, batch["action"][:, anchor, :10])
                expected.append(state[0])
            expected = torch.stack(expected) + network.action_history_type_embedding

        hook = network.blocks[0].register_forward_pre_hook(inspect_tokens, with_kwargs=True)
        try:
            for start, length, final in ((0, 2, False), (2, 3, True)):
                chunk = {key: value[:, start:start + length + 1] for key, value in batch.items()}
                chunk.update(_streaming_supervised_length=length, _streaming_reset=start == 0,
                             _streaming_final=final)
                loss = policy.compute_streaming_loss(chunk)
                (loss["action"]).backward()
                self.assertTrue(torch.isfinite(loss["action"]))
                if not final:
                    self.assertFalse(policy._training_action_history[0].requires_grad)
        finally:
            hook.remove()
        torch.testing.assert_close(torch.cat(seen), expected)
        self.assertGreater(float(encoder.action_projection.weight.grad.abs().sum()), 0)
        for layer in encoder.sequence_layers:
            self.assertGreater(float(layer.self_attn.in_proj_weight.grad.abs().sum()), 0)
        self.assertIsNone(policy._training_action_history)
        self.assertIsNone(policy._streaming_training_state)

    def test_online_confirmed_prefix_and_fixed_denoising_context(self):
        policy, batch = self.make()
        policy.eval()
        encoder = policy.action_history_encoder
        network = policy.denoising_network
        cls_tokens = []

        def inspect_tokens(module, args, kwargs):
            cls_tokens.append(kwargs["x"][:, 2].clone())

        hook = network.blocks[0].register_forward_pre_hook(inspect_tokens, with_kwargs=True)
        try:
            with torch.inference_mode(), patch.object(encoder, "forward", wraps=encoder.forward) as update:
                expected = encoder.initial_state(1)
                for anchor, steps in enumerate((10, 10, 10, 10, 10, 6)):
                    observation = {key: batch[key][:, anchor]
                                   for key in ("head_camera_feature", "agent_pos")}
                    observation["episode_idx"] = torch.tensor([0])
                    cls_tokens.clear()
                    output = policy.predict_action(observation)["action"]
                    self.assertEqual(output.shape, (1, 16, 14))
                    self.assertEqual(update.call_count, anchor)
                    self.assertEqual(len(cls_tokens), 2)
                    for token in cls_tokens:
                        torch.testing.assert_close(token, expected + network.action_history_type_embedding)
                    policy.record_executed_actions(0, steps)
                    args, _ = update.call_args
                    torch.testing.assert_close(args[0], expected)
                    torch.testing.assert_close(args[1], output[:, :steps])
                    expected = policy._online_action_history[0][None]
                self.assertEqual(update.call_count, 6)
                self.assertFalse(policy._pending_actions)
        finally:
            hook.remove()
        bank = policy.online_memory_banks[0]
        self.assertTrue(any(slot.state == "shadow" for slot in bank.slots))
        self.assertTrue(all(slot.latent.shape == (8, 32) for slot in bank.slots))
        policy.reset()
        self.assertEqual(policy._online_action_history, {})
        self.assertEqual(policy.online_memory_banks, {})

    def test_visual_history_policy_matches_disabled_legacy_checkpoint(self):
        helper = fixture.PolicyFixture()
        cfg = helper.make_config(merge_size=[8, 4])
        legacy_cfg = deepcopy(cfg)
        self.configure_legacy_policy(legacy_cfg, enabled=False)
        legacy = helper.instantiate_policy(legacy_cfg)
        old = {key: value.clone() for key, value in legacy.state_dict().items()}
        batch = helper.make_batch()
        torch.manual_seed(5)
        expected = legacy.compute_loss(batch)
        del legacy
        policy = helper.instantiate_policy(cfg)
        policy.load_state_dict(old, strict=True)
        self.assertEqual(set(old), set(policy.state_dict()))
        torch.manual_seed(5)
        observed = policy.compute_loss(batch)
        for key in expected:
            self.assertTrue(torch.equal(expected[key], observed[key]), key)

    def test_gaps_and_temporal_order(self):
        policy, batch = self.make()
        encoder = policy.action_history_encoder
        initial = encoder.initial_state(1)
        chunk = batch["action"][:, 0, :10]
        self.assertFalse(torch.allclose(encoder(initial, chunk), encoder(initial, chunk.flip(1))))
        batch["traj_idx"][0, :4, 0] = torch.tensor([0, 10, 30, 40])
        batch["entire_traj_is_padding"][0, 1] = True
        read, _ = policy._training_history_tokens(batch, 4, None)
        torch.testing.assert_close(read[:, :3], initial[:, None].expand(-1, 3, -1))
        torch.testing.assert_close(read[:, 3], encoder(initial, batch["action"][:, 2, :10]))

    def test_deployment_reports_only_the_executed_prefix(self):
        env = SimpleNamespace(eval_success=False, take_action_cnt=0, get_obs=Mock())

        def execute(action, *, action_type):
            if int(action[0]) < 6:
                env.take_action_cnt += 1

        env.take_action = Mock(side_effect=execute)
        actions = np.broadcast_to(np.arange(10, dtype=np.float32)[:, None], (10, 14))
        endpoint = SimpleNamespace(
            get_action_for_execution=Mock(return_value={"actions": actions, "record_execution": True}),
            record_executed_actions=Mock(),
        )
        proxy = SimpleNamespace(call=lambda func_name, obs: getattr(endpoint, func_name)(obs))
        for model in (endpoint, proxy):
            with self.subTest(model=model):
                env.take_action.reset_mock()
                endpoint.record_executed_actions.reset_mock()
                with patch("deploy_policy.encode_obs", return_value={}):
                    deploy_eval(env, model, {})
                self.assertEqual(env.take_action.call_count, 10)
                endpoint.record_executed_actions.assert_called_once_with(6)


if __name__ == "__main__":
    unittest.main()
