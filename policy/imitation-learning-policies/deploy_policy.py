"""RMBench deployment entry points for the memory diffusion policy."""

from pathlib import Path

import numpy as np


CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")


def _rgb_to_chw(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got shape {image.shape}")
    return np.ascontiguousarray(
        np.moveaxis(image, -1, 0), dtype=np.float32
    ) / 255.0


def encode_obs(observation):
    """Map the standard RMBench observation to the policy-side names."""

    camera_observation = observation["observation"]
    encoded = {
        camera_name: _rgb_to_chw(camera_observation[camera_name]["rgb"])
        for camera_name in CAMERA_NAMES
    }
    encoded["agent_pos"] = np.asarray(
        observation["joint_action"]["vector"], dtype=np.float32
    )
    return encoded


def get_model(usr_args):
    """Construct the model adapter.

    The RMBench-facing code is intentionally independent of checkpoint
    internals. ``rmbench_model.py`` will provide the adapter once checkpoint
    composition and normalizer loading are connected.
    """

    checkpoint_path = usr_args.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        raise ValueError(
            "checkpoint_path is required. Pass it through eval.sh or "
            "deploy_policy.yml."
        )

    try:
        from .rmbench_model import RMBenchMemoryPolicy
    except ImportError as exc:
        raise RuntimeError(
            "The RMBench platform adapter is ready, but model checkpoint "
            "loading is not connected yet (missing rmbench_model.py)."
        ) from exc
    return RMBenchMemoryPolicy(
        checkpoint_path=str(checkpoint_path),
        device=usr_args.get("device", "cuda:0"),
        eval_output_dir=usr_args.get("resolved_eval_output_dir"),
    )


def _get_actions(model, obs):
    """Support both in-process evaluation and RMBench's model-server proxy."""

    if hasattr(model, "call"):
        return model.call(func_name="get_action", obs=obs)
    return model.get_action(obs)


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    actions = np.asarray(_get_actions(model, obs), dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Model actions must have shape [T,D], got {actions.shape}")

    # One model call corresponds to one memory slot/action chunk. The outer
    # RMBench loop supplies the representative observation for the next slot.
    for action_idx, action in enumerate(actions):
        TASK_ENV.take_action(action, action_type="qpos")
        if TASK_ENV.eval_success:
            break
        # RMBench records video frames from TASK_ENV.now_obs at the start of
        # take_action(). Refresh it between actions so a predicted chunk does
        # not appear as one frozen frame repeated for the whole chunk. These
        # intermediate observations are deliberately not sent to the policy:
        # policy memory must still advance once per configured action chunk.
        if action_idx + 1 < len(actions):
            TASK_ENV.get_obs()


def reset_model(model):
    """Clear all per-episode history before RMBench starts a new rollout."""

    if hasattr(model, "call"):
        model.call(func_name="reset_model")
    elif hasattr(model, "reset_model"):
        model.reset_model()
    elif hasattr(model, "reset"):
        model.reset()
    else:
        raise AttributeError("Model adapter must implement reset_model() or reset()")
