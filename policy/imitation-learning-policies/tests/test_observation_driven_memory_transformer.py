import unittest

import torch

from imitation_learning.models.denoising_networks.memory_transformer import (
    MemoryTransformer,
)
from imitation_learning.models.denoising_networks.modules import (
    HistoryCrossAttention,
)


def make_transformer(
    *,
    include_action_history: bool = False,
    history_time_encoding: str = "relative",
    memory_retrieval_mode: str = "learned_queries",
) -> MemoryTransformer:
    return MemoryTransformer(
        max_history_len=3,
        freeze_non_history_modules=False,
        history_attention_type="token_wise",
        record_data_entries=["history_cross_attention"],
        ssmax_scaling_param=None,
        include_action_history=include_action_history,
        history_action_num_per_chunk=2,
        decision_memory_token_num=4,
        skip_history_attn=False,
        add_memory_gate_token=False,
        binary_gating=True,
        straight_through="",
        add_additional_self_attn=True,
        history_img_features_dim=5,
        history_img_features_token_num=2,
        history_time_encoding=history_time_encoding,
        memory_retrieval_mode=memory_retrieval_mode,
        perception_retrieval_layers=2,
        action_dim=2,
        action_token_num=4,
        global_cond_dim=5,
        global_cond_token_num=6,
        local_cond_dim=3,
        local_cond_token_num=2,
        head_num=2,
        layer_num=2,
        hidden_dim=8,
        projector_type="linear",
        global_cond_pos_emb_type="1d",
        seed=7,
    )


def make_compact_three_view_transformer() -> MemoryTransformer:
    return MemoryTransformer(
        max_history_len=3,
        freeze_non_history_modules=False,
        history_attention_type="token_wise",
        record_data_entries=[],
        ssmax_scaling_param=None,
        include_action_history=True,
        history_action_num_per_chunk=2,
        decision_memory_token_num=4,
        skip_history_attn=False,
        binary_gating=True,
        add_additional_self_attn=True,
        history_img_features_dim=5,
        history_img_features_token_num=3,
        history_time_encoding="absolute",
        memory_retrieval_mode="perception_patches",
        perception_retrieval_layers=2,
        compact_multiview_slot=True,
        history_view_num=3,
        action_dim=2,
        action_token_num=4,
        global_cond_dim=5,
        global_cond_token_num=3,
        local_cond_dim=3,
        local_cond_token_num=2,
        head_num=2,
        layer_num=2,
        hidden_dim=8,
        projector_type="linear",
        seed=7,
    )


def make_state_conditioned_single_view_transformer() -> MemoryTransformer:
    return MemoryTransformer(
        max_history_len=3,
        freeze_non_history_modules=False,
        history_attention_type="token_wise",
        record_data_entries=[],
        ssmax_scaling_param=None,
        include_action_history=False,
        history_action_num_per_chunk=2,
        decision_memory_token_num=4,
        skip_history_attn=False,
        binary_gating=True,
        add_additional_self_attn=True,
        history_img_features_dim=5,
        history_img_features_token_num=2,
        history_time_encoding="absolute",
        memory_retrieval_mode="perception_patches",
        perception_retrieval_layers=2,
        state_conditioned_history_slot=True,
        action_dim=2,
        action_token_num=4,
        global_cond_dim=5,
        global_cond_token_num=2,
        local_cond_dim=3,
        local_cond_token_num=2,
        head_num=2,
        layer_num=2,
        hidden_dim=8,
        projector_type="linear",
        seed=7,
    )


class ObservationDrivenMemoryTransformerTest(unittest.TestCase):
    def test_additive_attention_bias_uses_raw_bias_and_restores_std(self) -> None:
        attention = HistoryCrossAttention(
            dim=4,
            head_num=1,
            qkv_bias=True,
            qk_norm=False,
            norm_layer=torch.nn.LayerNorm,
            attention_type="token_wise",
            ssmax_scaling_param=None,
            attention_bias_mode="additive",
            attention_bias_lambda=0.5,
            attention_bias_epsilon=1.0e-6,
        )
        raw = torch.tensor(
            [[[[0.2, -0.1, 1.0, 0.4, -0.7, 0.3]]]],
            requires_grad=True,
        )
        bias = torch.tensor([[0.5, -0.2, 0.1]], requires_grad=True)
        mask = torch.ones(1, 3, dtype=torch.bool)

        transformed = attention._apply_history_attention_bias(
            raw, bias, mask, history_token_num=2
        )
        raw_by_slot = raw.reshape(1, 1, 1, 3, 2)
        transformed_by_slot = transformed.reshape(1, 1, 1, 3, 2)
        raw_slot_logits = torch.logsumexp(raw_by_slot, dim=-1)
        transformed_slot_logits = torch.logsumexp(
            transformed_by_slot, dim=-1
        )

        slot_std = raw_slot_logits.std(dim=-1, unbiased=False, keepdim=True)
        biased_slot_logits = raw_slot_logits + 0.5 * bias[:, None]
        expected = biased_slot_logits * slot_std / (
            biased_slot_logits.std(dim=-1, unbiased=False, keepdim=True)
            + 1.0e-6
        )
        torch.testing.assert_close(transformed_slot_logits, expected)
        torch.testing.assert_close(
            transformed_slot_logits.std(
                dim=-1, unbiased=False, keepdim=True
            ),
            slot_std,
            rtol=1.0e-5,
            atol=1.0e-6,
        )

        # A slot-level delta is shared by all patches in that slot.
        torch.testing.assert_close(
            transformed_by_slot[..., 0] - transformed_by_slot[..., 1],
            raw_by_slot[..., 0] - raw_by_slot[..., 1],
        )
        gradient = torch.autograd.grad(transformed.sum(), bias)[0]
        self.assertTrue(torch.isfinite(gradient).all())

    def test_standardized_attention_bias_uses_total_slot_logits(self) -> None:
        attention = HistoryCrossAttention(
            dim=4,
            head_num=1,
            qkv_bias=True,
            qk_norm=False,
            norm_layer=torch.nn.LayerNorm,
            attention_type="token_wise",
            ssmax_scaling_param=None,
            attention_bias_mode="standardized",
            attention_bias_lambda=0.5,
            attention_bias_epsilon=1.0e-6,
        )
        # One head/query, three slots, and two patch tokens per slot.
        raw = torch.tensor(
            [[[[0.2, -0.1, 1.0, 0.4, -0.7, 0.3]]]],
            requires_grad=True,
        )
        bias = torch.tensor([[0.5, -0.2, 0.1]], requires_grad=True)
        mask = torch.ones(1, 3, dtype=torch.bool)

        transformed = attention._apply_history_attention_bias(
            raw, bias, mask, history_token_num=2
        )
        raw_by_slot = raw.reshape(1, 1, 1, 3, 2)
        transformed_by_slot = transformed.reshape(1, 1, 1, 3, 2)
        raw_slot_logits = torch.logsumexp(raw_by_slot, dim=-1)
        transformed_slot_logits = torch.logsumexp(
            transformed_by_slot, dim=-1
        )

        slot_std = raw_slot_logits.std(dim=-1, unbiased=False, keepdim=True)
        z = (bias - bias.mean(dim=-1, keepdim=True)) / (
            bias.std(dim=-1, unbiased=False, keepdim=True) + 1.0e-6
        )
        biased_slot_logits = raw_slot_logits + 0.5 * slot_std * z[:, None]
        expected = biased_slot_logits * slot_std / (
            biased_slot_logits.std(dim=-1, unbiased=False, keepdim=True)
            + 1.0e-6
        )
        torch.testing.assert_close(transformed_slot_logits, expected)

        # Broadcasting one slot delta to all its patches must preserve the
        # within-image patch-logit differences exactly.
        torch.testing.assert_close(
            transformed_by_slot[..., 0] - transformed_by_slot[..., 1],
            raw_by_slot[..., 0] - raw_by_slot[..., 1],
        )
        gradient = torch.autograd.grad(transformed.sum(), bias)[0]
        self.assertTrue(torch.isfinite(gradient).all())

    def test_standardized_attention_bias_skips_single_valid_slot(self) -> None:
        attention = HistoryCrossAttention(
            dim=4,
            head_num=1,
            qkv_bias=True,
            qk_norm=False,
            norm_layer=torch.nn.LayerNorm,
            attention_type="token_wise",
            ssmax_scaling_param=None,
            attention_bias_mode="standardized",
        )
        raw = torch.randn(1, 1, 2, 6)
        bias = torch.tensor([[0.5, -0.2, 0.1]], requires_grad=True)
        mask = torch.tensor([[False, True, False]])

        transformed = attention._apply_history_attention_bias(
            raw, bias, mask, history_token_num=2
        )
        torch.testing.assert_close(transformed, raw)
        gradient = torch.autograd.grad(transformed.sum(), bias)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    def test_single_view_training_stores_proprio_modulated_patches(self) -> None:
        model = make_state_conditioned_single_view_transformer().eval()
        batch_size, traj_num = 1, 4
        global_cond = torch.randn(batch_size, traj_num, 2, 5)
        local_cond = torch.randn(batch_size, traj_num, 2, 3)
        projected = model.prepare_parallel_forward(
            {
                "noisy_action": torch.randn(batch_size, traj_num, 4, 2),
                "step": torch.tensor([3]),
                "global_cond": global_cond,
                "local_cond": local_cond,
                "history_noisy_actions": torch.randn(
                    batch_size, traj_num, 2, 2
                ),
                "history_frame_indices": torch.tensor([[0, 10, 20, 30]]),
            }
        )

        stored = projected["all_history_latents"]
        self.assertEqual(stored.shape, (batch_size, traj_num, 2, 8))
        self.assertIsInstance(model.history_img_features_projector, torch.nn.Identity)
        expected = model.project_state_conditioned_slot_features(
            global_cond=global_cond[:, 0],
            local_cond=local_cond[:, 0],
        )
        torch.testing.assert_close(stored[:, 0], expected)

        without_proprio = model.project_state_conditioned_slot_features(
            global_cond=global_cond[:, 0],
            local_cond=torch.zeros_like(local_cond[:, 0]),
        )
        self.assertFalse(torch.allclose(stored[:, 0], without_proprio))

    def test_compact_three_view_metadata_is_added_only_to_history_keys(
        self,
    ) -> None:
        model = make_compact_three_view_transformer().eval()
        slot_values = torch.randn(1, 2, 5, 8)
        untouched_values = slot_values.clone()
        time_embedding = model.encode_history_frame_indices(
            torch.tensor([[0, 10]])
        )

        history_keys = model.add_history_key_encoding(
            slot_values, time_embedding
        )

        torch.testing.assert_close(slot_values, untouched_values)
        key_delta = history_keys - slot_values
        torch.testing.assert_close(
            key_delta[:, :, :2], time_embedding[:, :, None].expand(-1, -1, 2, -1)
        )
        assert model.history_view_embedding is not None
        expected_visual_delta = (
            time_embedding[:, :, None]
            + model.history_view_embedding[:, None]
        )
        torch.testing.assert_close(key_delta[:, :, 2:], expected_visual_delta)

    def test_compact_three_view_training_slots_are_pure_conditioned_tokens(
        self,
    ) -> None:
        model = make_compact_three_view_transformer().eval()
        batch_size, traj_num = 1, 4
        projected = model.prepare_parallel_forward(
            {
                "noisy_action": torch.randn(batch_size, traj_num, 4, 2),
                "step": torch.tensor([3]),
                "global_cond": torch.randn(batch_size, traj_num, 3, 5),
                "local_cond": torch.randn(batch_size, traj_num, 2, 3),
                "history_noisy_actions": torch.randn(
                    batch_size, traj_num, 2, 2
                ),
                "history_frame_indices": torch.tensor([[0, 10, 20, 30]]),
            }
        )

        current_slots = projected["all_current_slot_features"]
        self.assertEqual(current_slots.shape, (batch_size, traj_num, 3, 8))
        self.assertEqual(
            projected["all_history_latents"].shape,
            (batch_size, traj_num, 5, 8),
        )
        torch.testing.assert_close(
            projected["all_history_latents"][:, :, 2:], current_slots
        )

        # The last anchor reads anchors 0..2. Values remain pure; only keys
        # carry absolute-frame and per-view metadata.
        values = projected["history_latents"].reshape(
            batch_size, traj_num, 3, 5, 8
        )[0, 3]
        keys = projected["history_key_latents"].reshape(
            batch_size, traj_num, 3, 5, 8
        )[0, 3]
        torch.testing.assert_close(
            values, projected["all_history_latents"][0, :3]
        )
        self.assertFalse(torch.allclose(keys, values))

    def test_compact_three_view_decision_memory_conditions_every_dit_layer(
        self,
    ) -> None:
        model = make_compact_three_view_transformer().eval()
        seen_memory_token_nums: list[int] = []
        handles = [
            block.decision_memory_cross_attn.register_forward_hook(
                lambda _module, inputs, _output: seen_memory_token_nums.append(
                    inputs[1].shape[1]
                )
            )
            for block in model.blocks
        ]
        try:
            output = model(
                {
                    "noisy_action": torch.randn(1, 4, 2),
                    "step": torch.tensor([3]),
                    "global_cond": torch.randn(1, 3, 5),
                    "local_cond": torch.randn(1, 2, 3),
                    "history_noisy_actions": torch.randn(1, 3, 2, 2),
                    # Online compact slots are already hidden-dim S_t values.
                    "history_img_features": torch.randn(1, 3, 3, 8),
                    "history_frame_indices": torch.tensor([[0, 10, 20]]),
                    "history_mask": torch.ones(1, 3, dtype=torch.bool),
                }
            )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(output["decision_memory"].shape, (1, 3, 8))
        self.assertEqual(seen_memory_token_nums, [3, 3])

    def test_perception_mode_keeps_all_current_patch_tokens(self) -> None:
        model = make_transformer(memory_retrieval_mode="perception_patches")
        batch_size = 2
        output = model(
            {
                "noisy_action": torch.randn(batch_size, 4, 2),
                "step": torch.tensor([3, 6]),
                "global_cond": torch.randn(batch_size, 6, 5),
                "local_cond": torch.randn(batch_size, 2, 3),
                "history_noisy_actions": torch.randn(batch_size, 3, 2, 2),
                "history_img_features": torch.randn(batch_size, 3, 2, 5),
                "history_mask": torch.tensor(
                    [[False, False, False], [False, True, True]]
                ),
            }
        )

        self.assertEqual(output["action"].shape, (batch_size, 4, 2))
        # h_t has one token per current image patch, not a learned-query count.
        self.assertEqual(output["decision_memory"].shape, (batch_size, 6, 8))
        self.assertTrue(torch.isfinite(output["decision_memory"]).all())
        # Two perception blocks run, while diagnostics intentionally retain
        # only the final retrieval so existing logging remains one-read/anchor.
        self.assertEqual(
            len(model.recorded_data_dict["history_cross_attention"]), 1
        )

    def test_perception_mode_uses_position_for_keys_not_slot_values(self) -> None:
        model = make_transformer(
            memory_retrieval_mode="perception_patches",
            history_time_encoding="absolute",
        ).eval()
        common = {
            "noisy_action": torch.randn(1, 4, 2),
            "step": torch.tensor([5]),
            "global_cond": torch.randn(1, 6, 5),
            "local_cond": torch.randn(1, 2, 3),
            "history_noisy_actions": torch.randn(1, 3, 2, 2),
            "history_img_features": torch.randn(1, 3, 2, 5),
            "history_mask": torch.ones(1, 3, dtype=torch.bool),
        }
        first = model.prepare_single_memory_retrieval(
            {**common, "history_frame_indices": torch.tensor([[0, 10, 20]])}
        )
        second = model.prepare_single_memory_retrieval(
            {**common, "history_frame_indices": torch.tensor([[0, 10, 30]])}
        )

        torch.testing.assert_close(
            first["history_latents"], second["history_latents"]
        )
        torch.testing.assert_close(
            first["history_key_latents"][:, :2],
            second["history_key_latents"][:, :2],
        )
        self.assertFalse(
            torch.allclose(
                first["history_key_latents"][:, 2],
                second["history_key_latents"][:, 2],
            )
        )

    def test_perception_mode_slot_bias_has_a_finite_gradient(self) -> None:
        model = make_transformer(
            memory_retrieval_mode="perception_patches"
        ).eval()
        retrieval_inputs = model.prepare_single_memory_retrieval(
            {
                "noisy_action": torch.randn(1, 4, 2),
                "step": torch.tensor([7]),
                "global_cond": torch.randn(1, 6, 5),
                "local_cond": torch.randn(1, 2, 3),
                "history_noisy_actions": torch.randn(1, 3, 2, 2),
                "history_img_features": torch.randn(1, 3, 2, 5),
                "history_mask": torch.ones(1, 3, dtype=torch.bool),
            }
        )
        read_bias = torch.zeros(1, 3, requires_grad=True)
        decision_memory = model.retrieve_decision_memory(
            retrieval_inputs, history_attention_bias=read_bias
        )
        gradient = torch.autograd.grad(
            decision_memory.square().mean(), read_bias
        )[0]

        self.assertEqual(decision_memory.shape, (1, 6, 8))
        self.assertEqual(gradient.shape, (1, 3))
        self.assertTrue(torch.isfinite(gradient).all())

    def test_perception_mode_has_no_trainable_unused_retriever_params(self) -> None:
        model = make_transformer(memory_retrieval_mode="perception_patches")
        output = model(
            {
                "noisy_action": torch.randn(1, 4, 2),
                "step": torch.tensor([7]),
                "global_cond": torch.randn(1, 6, 5),
                "local_cond": torch.randn(1, 2, 3),
                "history_noisy_actions": torch.randn(1, 3, 2, 2),
                "history_img_features": torch.randn(1, 3, 2, 5),
                "history_mask": torch.ones(1, 3, dtype=torch.bool),
            }
        )
        (output["action"].square().mean()
         + output["decision_memory"].square().mean()).backward()
        unused = [
            name
            for name, parameter in model.memory_retriever.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(unused, [])

    def test_absolute_history_time_uses_source_frame_indices(self) -> None:
        model = make_transformer(history_time_encoding="absolute").eval()
        common = {
            "noisy_action": torch.randn(1, 4, 2),
            "step": torch.tensor([5]),
            "global_cond": torch.randn(1, 6, 5),
            "local_cond": torch.randn(1, 2, 3),
            "history_noisy_actions": torch.randn(1, 3, 2, 2),
            "history_img_features": torch.randn(1, 3, 2, 5),
            "history_mask": torch.ones(1, 3, dtype=torch.bool),
        }

        first = model.prepare_single_memory_retrieval(
            {**common, "history_frame_indices": torch.tensor([[0, 10, 20]])}
        )["history_latents"]
        second = model.prepare_single_memory_retrieval(
            {**common, "history_frame_indices": torch.tensor([[0, 10, 30]])}
        )["history_latents"]

        torch.testing.assert_close(first[:, :2], second[:, :2])
        self.assertFalse(torch.allclose(first[:, 2], second[:, 2]))

    def test_absolute_history_time_requires_frame_indices(self) -> None:
        model = make_transformer(history_time_encoding="absolute").eval()
        with self.assertRaisesRegex(ValueError, "history_frame_indices"):
            model.prepare_single_memory_retrieval(
                {
                    "noisy_action": torch.randn(1, 4, 2),
                    "step": torch.tensor([5]),
                    "global_cond": torch.randn(1, 6, 5),
                    "local_cond": torch.randn(1, 2, 3),
                    "history_noisy_actions": torch.randn(1, 3, 2, 2),
                    "history_img_features": torch.randn(1, 3, 2, 5),
                    "history_mask": torch.ones(1, 3, dtype=torch.bool),
                }
            )

    def test_parallel_absolute_time_gathers_true_source_frames(self) -> None:
        model = make_transformer(history_time_encoding="absolute").eval()
        batch_size, traj_num = 1, 4
        frame_indices = torch.tensor([[3, 13, 23, 33]])
        projected = model.prepare_parallel_forward(
            {
                "noisy_action": torch.randn(batch_size, traj_num, 4, 2),
                "step": torch.tensor([3]),
                "global_cond": torch.randn(batch_size, traj_num, 6, 5),
                "local_cond": torch.randn(batch_size, traj_num, 2, 3),
                "history_noisy_actions": torch.randn(
                    batch_size, traj_num, 2, 2
                ),
                "history_img_features": torch.randn(
                    batch_size, traj_num, 2, 5
                ),
                "history_frame_indices": frame_indices,
            }
        )

        # Anchor 3 sees source anchors 0, 1, 2. Its history must keep those
        # source anchors' absolute positions instead of renumbering them 0..2.
        expected = projected["all_history_latents"][0, :3]
        expected = expected + model.encode_history_frame_indices(
            frame_indices[0, :3]
        )[:, None, :]
        actual = projected["history_latents"].reshape(
            batch_size, traj_num, 3, 2, 8
        )[0, 3]
        torch.testing.assert_close(actual, expected)

    def test_action_and_image_tokens_share_their_slot_attention_bias(self) -> None:
        model = make_transformer(include_action_history=True).eval()
        batch_size, history_len = 1, 3
        common = {
            "noisy_action": torch.randn(batch_size, 4, 2),
            "step": torch.tensor([5]),
            "global_cond": torch.randn(batch_size, 6, 5),
            "local_cond": torch.randn(batch_size, 2, 3),
            "history_noisy_actions": torch.randn(
                batch_size, history_len, 2, 2
            ),
            "history_img_features": torch.randn(
                batch_size, history_len, 2, 5
            ),
            "history_mask": torch.ones(
                batch_size, history_len, dtype=torch.bool
            ),
        }
        retrieval_inputs = model.prepare_single_memory_retrieval(common)

        # Each slot contains two action tokens followed by two image tokens.
        self.assertEqual(
            retrieval_inputs["history_latents"].shape,
            (batch_size, history_len, 4, 8),
        )

        zero_records = {"history_attention_logits": []}
        biased_records = {"history_attention_logits": []}
        zero_bias = torch.zeros(batch_size, history_len)
        slot_bias = torch.tensor([[0.30, -0.20, 0.10]])
        with torch.no_grad():
            model.retrieve_decision_memory(
                retrieval_inputs,
                history_attention_bias=zero_bias,
                record_data_dict=zero_records,
            )
            model.retrieve_decision_memory(
                retrieval_inputs,
                history_attention_bias=slot_bias,
                record_data_dict=biased_records,
            )

        logit_delta = (
            biased_records["history_attention_logits"][0]
            - zero_records["history_attention_logits"][0]
        )
        logit_delta = logit_delta.reshape(
            batch_size,
            model.head_num,
            model.decision_memory_token_num,
            history_len,
            4,
        )
        # Variance restoration changes the numerical slot delta, but every
        # action/image token within one slot must still receive the same delta.
        expected_delta = logit_delta[..., :1].expand_as(logit_delta)
        torch.testing.assert_close(logit_delta, expected_delta)

    def test_single_trajectory_forward_has_fixed_query_size(self) -> None:
        model = make_transformer()
        batch_size = 2
        output = model(
            {
                "noisy_action": torch.randn(batch_size, 4, 2),
                "step": torch.tensor([3, 6]),
                "global_cond": torch.randn(batch_size, 6, 5),
                "local_cond": torch.randn(batch_size, 2, 3),
                "history_noisy_actions": torch.randn(batch_size, 3, 2, 2),
                "history_img_features": torch.randn(batch_size, 3, 2, 5),
                "history_mask": torch.tensor(
                    [[False, False, False], [False, True, True]]
                ),
            }
        )

        self.assertEqual(output["action"].shape, (batch_size, 4, 2))
        self.assertEqual(output["decision_memory"].shape, (batch_size, 4, 8))
        self.assertTrue(torch.isfinite(output["decision_memory"]).all())
        # Raw history is read exactly once, not once per transformer layer.
        self.assertEqual(
            len(model.recorded_data_dict["history_cross_attention"]), 1
        )

    def test_each_block_queries_current_observation_before_memory(self) -> None:
        model = make_transformer().eval()
        call_order: list[tuple[int, str, int]] = []
        handles = []

        def record_call(block_idx: int, source: str):
            def hook(_module, inputs, _output) -> None:
                call_order.append((block_idx, source, inputs[1].shape[1]))

            return hook

        for block_idx, block in enumerate(model.blocks):
            handles.append(
                block.cross_attn.register_forward_hook(
                    record_call(block_idx, "current_observation")
                )
            )
            handles.append(
                block.decision_memory_cross_attn.register_forward_hook(
                    record_call(block_idx, "decision_memory")
                )
            )

        try:
            with torch.no_grad():
                model(
                    {
                        "noisy_action": torch.randn(1, 4, 2),
                        "step": torch.tensor([4]),
                        "global_cond": torch.randn(1, 6, 5),
                        "local_cond": torch.randn(1, 2, 3),
                        "history_noisy_actions": torch.randn(1, 3, 2, 2),
                        "history_img_features": torch.randn(1, 3, 2, 5),
                        "history_mask": torch.ones(1, 3, dtype=torch.bool),
                    }
                )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(
            call_order,
            [
                (0, "current_observation", 6),
                (0, "decision_memory", 4),
                (1, "current_observation", 6),
                (1, "decision_memory", 4),
            ],
        )

    def test_empty_history_reduces_to_current_observation(self) -> None:
        model = make_transformer().eval()
        common = {
            "noisy_action": torch.randn(1, 4, 2),
            "step": torch.tensor([2]),
            "global_cond": torch.randn(1, 6, 5),
            "local_cond": torch.randn(1, 2, 3),
            "history_noisy_actions": torch.randn(1, 3, 2, 2),
            "history_img_features": torch.randn(1, 3, 2, 5),
        }
        with torch.no_grad():
            masked = model(
                {**common, "history_mask": torch.zeros(1, 3, dtype=torch.bool)}
            )["decision_memory"]
            model.set_skip_history_attn(True)
            skipped = model(common)["decision_memory"]
        torch.testing.assert_close(masked, skipped)

    def test_multi_trajectory_forward_returns_one_memory_per_anchor(self) -> None:
        model = make_transformer()
        batch_size, traj_num = 2, 4
        output = model.parallel_forward(
            {
                "noisy_action": torch.randn(batch_size, traj_num, 4, 2),
                "step": torch.tensor([3, 6]),
                "global_cond": torch.randn(batch_size, traj_num, 6, 5),
                "local_cond": torch.randn(batch_size, traj_num, 2, 3),
                "history_noisy_actions": torch.randn(
                    batch_size, traj_num, 2, 2
                ),
                "history_img_features": torch.randn(
                    batch_size, traj_num, 2, 5
                ),
                "entire_traj_is_padding": torch.zeros(
                    batch_size, traj_num, dtype=torch.bool
                ),
            }
        )
        self.assertEqual(output["action"].shape, (batch_size, traj_num, 4, 2))
        self.assertEqual(
            output["decision_memory"].shape, (batch_size, traj_num, 4, 8)
        )

    def test_precomputed_memory_is_reused_across_denoising_steps(self) -> None:
        model = make_transformer().eval()
        first = model(
            {
                "noisy_action": torch.randn(1, 4, 2),
                "step": torch.tensor([7]),
                "global_cond": torch.randn(1, 6, 5),
                "local_cond": torch.randn(1, 2, 3),
                "history_noisy_actions": torch.randn(1, 3, 2, 2),
                "history_img_features": torch.randn(1, 3, 2, 5),
                "history_mask": torch.ones(1, 3, dtype=torch.bool),
            }
        )
        cached = first["decision_memory"].detach()
        second = model(
            {
                "noisy_action": torch.randn(1, 4, 2),
                "step": torch.tensor([3]),
                "global_cond": torch.randn(1, 6, 5),
                "local_cond": torch.randn(1, 2, 3),
                "decision_memory": cached,
            }
        )
        torch.testing.assert_close(second["decision_memory"], cached)

    def test_single_memory_retrieval_can_be_recomputed_for_bias_gradient(
        self,
    ) -> None:
        model = make_transformer().eval()
        retrieval_inputs = model.prepare_single_memory_retrieval(
            {
                "noisy_action": torch.randn(1, 4, 2),
                "step": torch.tensor([7]),
                "global_cond": torch.randn(1, 6, 5),
                "local_cond": torch.randn(1, 2, 3),
                "history_noisy_actions": torch.randn(1, 3, 2, 2),
                "history_img_features": torch.randn(1, 3, 2, 5),
                "history_mask": torch.ones(1, 3, dtype=torch.bool),
                "history_attention_bias": torch.zeros(1, 3),
            }
        )
        read_bias = torch.zeros(1, 3, requires_grad=True)

        decision_memory = model.retrieve_decision_memory(
            retrieval_inputs, history_attention_bias=read_bias
        )
        gradient = torch.autograd.grad(
            decision_memory.square().mean(), read_bias
        )[0]

        self.assertEqual(decision_memory.shape, (1, 4, 8))
        self.assertEqual(gradient.shape, (1, 3))
        self.assertTrue(torch.isfinite(gradient).all())

    def test_prepared_parallel_forward_accepts_causal_decision_memory(self) -> None:
        model = make_transformer()
        batch_size, traj_num = 2, 4
        projected = model.prepare_parallel_forward(
            {
                "noisy_action": torch.randn(batch_size, traj_num, 4, 2),
                "step": torch.tensor([3, 6]),
                "global_cond": torch.randn(batch_size, traj_num, 6, 5),
                "local_cond": torch.randn(batch_size, traj_num, 2, 3),
                "history_noisy_actions": torch.randn(
                    batch_size, traj_num, 2, 2
                ),
                "history_img_features": torch.randn(
                    batch_size, traj_num, 2, 5
                ),
                "entire_traj_is_padding": torch.zeros(
                    batch_size, traj_num, dtype=torch.bool
                ),
            }
        )
        decision_memory = torch.randn(batch_size, traj_num, 4, 8)
        self.assertEqual(
            projected["all_history_latents"].shape,
            (batch_size, traj_num, 2, 8),
        )
        projected["decision_memory"] = decision_memory.reshape(
            batch_size * traj_num, 4, 8
        )

        output = model.finish_parallel_forward(
            projected, batch_size=batch_size, traj_num=traj_num
        )

        self.assertEqual(output["action"].shape, (batch_size, traj_num, 4, 2))
        torch.testing.assert_close(output["decision_memory"], decision_memory)

    def test_epoch_attention_stats_treat_causal_anchors_as_samples(self) -> None:
        model = make_transformer().train()
        record_data = {
            "history_cross_attention_frame_logits": [],
            "history_cross_attention_slot_weights": [],
        }
        for valid_history in (1, 2):
            history_mask = torch.zeros(2, 3, dtype=torch.bool)
            history_mask[:, -valid_history:] = True
            model.memory_retriever(
                global_cond=torch.randn(2, 6, 8),
                local_cond=torch.randn(2, 2, 8),
                history_latents=torch.randn(2, 3, 2, 8),
                history_mask=history_mask,
                record_data_dict=record_data,
            )

        model._accumulate_history_attention_epoch_stats(record_data)
        stats = model.pop_history_attention_epoch_stats()

        self.assertEqual(stats["frame_logit"]["mean"].shape, (1, 3, 2))
        self.assertEqual(stats["slot_logit"]["mean"].shape, (1, 3))
        self.assertGreater(int(stats["slot_logit"]["count"].sum()), 0)



if __name__ == "__main__":
    unittest.main()
