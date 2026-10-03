"""Convert RMBench demonstrations to the native episodic Zarr layout.

The temporal alignment intentionally matches RMBench's Diffusion Policy:

    state[t]  = joint_action/vector[t]
    action[t] = joint_action/vector[t + 1]

An episode containing N recorded observations therefore contributes N - 1
training samples. The output is consumed directly by the existing
``EpisodicDataset`` and ``MultiTrajDataset`` implementations::

    dataset.zarr/
      episode_0/
        head_camera  # [T,H,W,3], uint8
        left_camera  # [T,H,W,3], uint8
        right_camera # [T,H,W,3], uint8
        agent_pos    # [T,D], float32
        action       # [T,D], float32
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import zarr


POLICY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = POLICY_DIR.parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "data"
# Keep converted policy data next to the RMBench datasets, but in a dedicated
# directory so it cannot be confused with the original HDF5 demonstrations in
# ``RMBench/data/data``.
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "imitation_learning_policies"
CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
IMAGE_CHUNK_LENGTH = 10


def _episode_number(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Unexpected episode filename: {path.name}")
    return int(match.group(1))


def _decode_rgb_frame(encoded_frame: np.ndarray, source: Path, frame_idx: int) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(encoded_frame, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError(f"Failed to decode frame {frame_idx} from {source}")
    # Do not apply BGR2RGB here. RMBench writes the live RGB byte order
    # directly through cv2.imencode, and policy/DP preserves that byte order.
    return image


def load_episode(
    path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Return aligned HWC images, states, and actions for one episode."""

    with h5py.File(path, "r") as root:
        joint_key = "/joint_action/vector"
        if joint_key not in root:
            raise KeyError(f"Missing {joint_key} in {path}")

        joint_vector = root[joint_key][:].astype(np.float32, copy=False)
        encoded_images_by_camera = {}
        for camera_name in CAMERA_NAMES:
            image_key = f"/observation/{camera_name}/rgb"
            if image_key not in root:
                raise KeyError(f"Missing {image_key} in {path}")
            encoded_images_by_camera[camera_name] = root[image_key][:]

    if joint_vector.ndim != 2:
        raise ValueError(
            f"Expected {joint_key} to have shape [T,D], got {joint_vector.shape} in {path}"
        )
    if len(joint_vector) < 2:
        raise ValueError(f"Episode must contain at least two frames: {path}")

    images_by_camera = {}
    for camera_name, encoded_images in encoded_images_by_camera.items():
        if len(joint_vector) != len(encoded_images):
            raise ValueError(
                f"{camera_name}/state lengths differ in {path}: "
                f"{len(encoded_images)} != {len(joint_vector)}"
            )
        images = np.stack(
            [
                _decode_rgb_frame(encoded_images[idx], path, idx)
                for idx in range(len(encoded_images) - 1)
            ],
            axis=0,
        )
        images_by_camera[camera_name] = np.ascontiguousarray(
            images, dtype=np.uint8
        )

    states = np.ascontiguousarray(joint_vector[:-1], dtype=np.float32)
    actions = np.ascontiguousarray(joint_vector[1:], dtype=np.float32)
    return images_by_camera, states, actions


def convert_dataset(
    task_name: str,
    task_config: str,
    expert_data_num: int,
    data_root: Path,
    output_path: Path,
) -> Path:
    if expert_data_num <= 0:
        raise ValueError("expert_data_num must be positive")

    episode_dir = data_root / task_name / task_config / "data"
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"RMBench episode directory not found: {episode_dir}")

    episode_paths = sorted(
        episode_dir.glob("episode*.hdf5"),
        key=_episode_number,
    )
    if len(episode_paths) < expert_data_num:
        raise FileNotFoundError(
            f"Requested {expert_data_num} episodes, but only found "
            f"{len(episode_paths)} in {episode_dir}"
        )
    episode_paths = episode_paths[:expert_data_num]

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.group(str(output_path))
    # RMBench history sampling reads sparse individual frames. Keep image
    # chunks short and use the same fast, lossless codec as the original
    # imitation-learning datasets.
    compressor = zarr.Blosc(cname="lz4", clevel=5, shuffle=1)
    episode_frame_nums: dict[str, int] = {}
    total_count = 0
    expected_image_shapes: dict[str, tuple[int, ...]] = {}
    expected_state_dim = None

    for episode_idx, episode_path in enumerate(episode_paths):
        print(
            f"Processing episode {episode_idx + 1}/{expert_data_num}: "
            f"{episode_path.name}",
            flush=True,
        )
        images_by_camera, states, actions = load_episode(episode_path)
        episode_length = len(states)

        if expected_state_dim is None:
            expected_state_dim = states.shape[1]
        else:
            if states.shape[1] != expected_state_dim:
                raise ValueError(
                    f"State dimension changed at {episode_path}: "
                    f"{states.shape[1]} != {expected_state_dim}"
                )
            if actions.shape[1] != expected_state_dim:
                raise ValueError(
                    f"Action dimension changed at {episode_path}: "
                    f"{actions.shape[1]} != {expected_state_dim}"
                )

        episode_group = root.create_group(f"episode_{episode_idx}")
        for camera_name, images in images_by_camera.items():
            image_shape = tuple(images.shape[1:])
            expected_image_shape = expected_image_shapes.setdefault(
                camera_name, image_shape
            )
            if image_shape != expected_image_shape:
                raise ValueError(
                    f"{camera_name} shape changed at {episode_path}: "
                    f"{image_shape} != {expected_image_shape}"
                )
            episode_group.create_dataset(
                camera_name,
                data=images,
                chunks=(min(IMAGE_CHUNK_LENGTH, episode_length), *image_shape),
                dtype=np.uint8,
                compressor=compressor,
            )
        episode_group.create_dataset(
            "agent_pos",
            data=states,
            chunks=(min(100, episode_length), states.shape[1]),
            dtype=np.float32,
            compressor=compressor,
        )
        episode_group.create_dataset(
            "action",
            data=actions,
            chunks=(min(100, episode_length), actions.shape[1]),
            dtype=np.float32,
            compressor=compressor,
        )
        episode_group.attrs.update(
            {
                "source_episode_id": _episode_number(episode_path),
                "source_path": str(episode_path),
            }
        )
        total_count += episode_length
        episode_frame_nums[str(episode_idx)] = episode_length

    root.attrs.update(
        {
            "task_name": task_name,
            "task_config": task_config,
            "expert_data_num": expert_data_num,
            "episode_frame_nums": episode_frame_nums,
            "camera_names": list(CAMERA_NAMES),
            "image_chunk_length": IMAGE_CHUNK_LENGTH,
            "compressor": "blosc-lz4-clevel5-shuffle1",
            "state_action_alignment": "state[t]=vector[t], action[t]=vector[t+1]",
        }
    )
    print(f"Saved {total_count} samples from {expert_data_num} episodes to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RMBench HDF5 demonstrations to episodic Zarr."
    )
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"RMBench data root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output .zarr path. Defaults to RMBench/data/"
            "imitation_learning_policies/<task>-<config>-<num>.zarr."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_path
    if output_path is None:
        output_path = (
            DEFAULT_OUTPUT_ROOT
            / f"{args.task_name}-{args.task_config}-{args.expert_data_num}.zarr"
        )
    convert_dataset(
        task_name=args.task_name,
        task_config=args.task_config,
        expert_data_num=args.expert_data_num,
        data_root=args.data_root.expanduser().resolve(),
        output_path=output_path.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
