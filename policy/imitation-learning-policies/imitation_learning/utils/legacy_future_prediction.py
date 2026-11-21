def is_future_prediction_parameter(name: str) -> bool:
    return name.startswith(("future_feature_predictor.", "future_img_feature_encoder."))


def remove_future_prediction_optimizer_state(
    state_dict, legacy_model_state, model, trainer,
):
    """Recover legacy optimizer parameter names from the saved model order."""
    if not any(is_future_prediction_parameter(name) for name in legacy_model_state):
        return state_dict

    current = {name for name, param in model.named_parameters() if param.requires_grad}
    legacy = [
        name for name in legacy_model_state
        if name in current or name.startswith("future_feature_predictor.")
    ]

    def groups(names):
        # Match BaseTrainer.init_optimizer's group order, including regular overrides.
        finetune = [name for name in names if any(key in name for key in trainer.finetune_keywords)]
        overrides = [name for name in names if any(key in name for key in trainer.regular_keywords)]
        return (
            [name for name in finetune if name not in overrides],
            [name for name in names if name not in finetune] + overrides,
        )

    old_groups = state_dict["param_groups"]
    legacy_groups = groups(legacy)
    current_groups = groups([name for name, _ in model.named_parameters() if name in current])
    if len(old_groups) != len(current_groups):
        raise ValueError("Legacy optimizer group count does not match the policy trainer")
    kept_groups = []
    kept_ids = set()
    for old_group, legacy_group, current_group in zip(old_groups, legacy_groups, current_groups):
        old_ids = old_group["params"]
        if len(old_ids) == len(current_group):
            kept_groups.append(old_group)
            kept_ids.update(old_ids)
            continue
        kept_names = [name for name in legacy_group if name in current]
        if len(old_ids) != len(legacy_group) or kept_names != current_group:
            raise ValueError("Cannot align legacy optimizer parameters after removing future prediction")
        ids = [
            param_id for param_id, name in zip(old_ids, legacy_group) if name in current
        ]
        kept_groups.append({**old_group, "params": ids})
        kept_ids.update(ids)
    return {
        **state_dict, "param_groups": kept_groups,
        "state": {key: state for key, state in state_dict["state"].items() if key in kept_ids},
    }
