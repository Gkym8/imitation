"""Precompute DINOv2 features for a converted RMBench Zarr.

The script leaves the original uint8 images untouched. Two cache layouts are
supported:

``cls`` (compact three-view training)::

    episode_0/head_camera_feature   # [T, 768]

``patches`` (full-patch one/two-view training)::

    episode_0/head_camera_patch_feature   # [T, 257, 768]

The patch cache contains ``[CLS, 256 spatial patches]``. Register tokens are
discarded. The same array therefore supplies the CLS future-prediction target
and the 256 current/history patch tokens without another backbone forward.

Preprocessing intentionally mirrors the RMBench training path: uint8 HWC
images are converted to float32 CHW in [0, 1], resized with Kornia's
``Resize(..., antialias=True)``, normalized with ImageNet mean/std, and passed
through the configured DINOv2 backbone.  For models with register tokens, the
features exactly match ``SharedDinoV2ImageEncoder``.

Incomplete writes use a separate ``__incomplete`` array. Re-running the
script resumes that array from its last completed batch; training therefore
never sees a partially written ``*_feature`` array.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import zarr
from tqdm import tqdm


POLICY_ROOT = Path(__file__).resolve().parents[1]
if str(POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICY_ROOT))

from imitation_learning.common.dataclasses import DataMeta
from imitation_learning.datasets.transforms import BaseTransforms


CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
DEFAULT_MODEL_NAME = "facebook/dinov2-with-registers-base"
DEFAULT_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEATURE_DTYPE = np.dtype(np.float32)
CLS_FEATURE_SUFFIX = "_feature"
PATCH_FEATURE_SUFFIX = "_patch_feature"
INCOMPLETE_SUFFIX = "__incomplete"
EXTRACTOR_VERSION = 2


def _episode_number(name: str) -> int:
    match = re.fullmatch(r"episode_(\d+)", name)
    if match is None:
        raise ValueError(f"Unexpected episode group name: {name!r}")
    return int(match.group(1))


def _episode_names(root: zarr.Group) -> list[str]:
    names = [name for name in root.group_keys() if name.startswith("episode_")]
    if not names:
        raise ValueError("The Zarr store does not contain any episode_* groups")
    return sorted(names, key=_episode_number)


def _parse_cameras(value: str) -> tuple[str, ...]:
    cameras = tuple(item.strip() for item in value.split(",") if item.strip())
    if not cameras:
        raise argparse.ArgumentTypeError("at least one camera is required")
    if len(set(cameras)) != len(cameras):
        raise argparse.ArgumentTypeError(f"duplicate camera in {value!r}")
    unsupported = sorted(set(cameras) - set(CAMERA_NAMES))
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"unsupported camera(s) {unsupported}; choose from {CAMERA_NAMES}"
        )
    return cameras


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {resolved}")
    return resolved


def _model_metadata(
    model: torch.nn.Module,
    model_name: str,
    input_size: int,
    feature_kind: str,
) -> dict[str, Any]:
    config = model.config
    model_type = str(getattr(config, "model_type", ""))
    if "dinov2" not in model_type.lower():
        raise ValueError(
            f"{model_name!r} is not a DINOv2 model (model_type={model_type!r})"
        )
    feature_dim = int(config.hidden_size)
    patch_size = int(config.patch_size)
    if input_size % patch_size != 0:
        raise ValueError(
            f"input size {input_size} must be divisible by patch size {patch_size}"
        )
    return {
        "extractor_version": EXTRACTOR_VERSION,
        "model_name": model_name,
        "model_type": model_type,
        "model_commit_hash": getattr(config, "_commit_hash", None),
        "input_size": input_size,
        "resize": "BaseTransforms/KorniaVideoSequential.Resize",
        "resize_antialias": True,
        "image_scale": "uint8/255.0",
        "image_mean": list(IMAGENET_MEAN),
        "image_std": list(IMAGENET_STD),
        "feature_kind": feature_kind,
        "feature_dim": feature_dim,
        "feature_dtype": FEATURE_DTYPE.name,
        "num_register_tokens": int(getattr(config, "num_register_tokens", 0)),
        "patch_size": patch_size,
        "patch_token_num": (input_size // patch_size) ** 2,
    }


class DinoV2FeatureExtractor:
    def __init__(
        self,
        model_name: str,
        input_size: int,
        device: torch.device,
        feature_kind: str,
    ) -> None:
        from transformers import AutoModel

        self.device = device
        self.input_size = input_size
        self.model = AutoModel.from_pretrained(model_name)
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(device=device, dtype=torch.float32)
        if feature_kind not in {"cls", "patches"}:
            raise ValueError(f"Unsupported feature kind: {feature_kind!r}")
        self.feature_kind = feature_kind
        self.metadata = _model_metadata(
            self.model, model_name, input_size, feature_kind
        )
        self.feature_dim = int(self.metadata["feature_dim"])
        resize_meta = DataMeta(
            name="image",
            shape=(3, input_size, input_size),
            data_type="image",
            length=1,
            normalizer="identity",
            augmentation=[
                {
                    "name": "Resize",
                    "size": [input_size, input_size],
                    "antialias": True,
                }
            ],
            source_entry_names=["image"],
        )
        self.transforms = BaseTransforms(
            {"image": resize_meta},
            apply_image_augmentation_in_cpu=True,
            seed=0,
        )
        self.transforms.to(device)
        self.mean = torch.tensor(
            IMAGENET_MEAN, dtype=torch.float32, device=device
        )[None, :, None, None]
        self.std = torch.tensor(
            IMAGENET_STD, dtype=torch.float32, device=device
        )[None, :, None, None]

    @torch.inference_mode()
    def __call__(self, images: np.ndarray) -> np.ndarray:
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"expected uint8 HWC images with shape [N,H,W,3], got {images.shape}"
            )
        if images.dtype != np.uint8:
            raise ValueError(f"expected uint8 images, got {images.dtype}")

        tensor = torch.from_numpy(np.ascontiguousarray(images))
        tensor = tensor.permute(0, 3, 1, 2).to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        tensor = tensor.div_(255.0)
        tensor = self.transforms.apply({"image": tensor})["image"]
        tensor = (tensor - self.mean) / self.std
        output = self.model(pixel_values=tensor).last_hidden_state
        expected_token_num = (
            1
            + int(self.metadata["num_register_tokens"])
            + int(self.metadata["patch_token_num"])
        )
        if output.shape[1] != expected_token_num:
            raise RuntimeError(
                "unexpected DINOv2 token count: "
                f"{output.shape[1]} != {expected_token_num}"
            )
        if self.feature_kind == "cls":
            features = output[:, 0, :]
            expected_shape = (len(images), self.feature_dim)
        else:
            patch_start = 1 + int(self.metadata["num_register_tokens"])
            features = torch.cat(
                [output[:, :1, :], output[:, patch_start:, :]], dim=1
            )
            expected_shape = (
                len(images),
                1 + int(self.metadata["patch_token_num"]),
                self.feature_dim,
            )
        if features.shape != expected_shape:
            raise RuntimeError(
                "unexpected cached feature shape: "
                f"{tuple(features.shape)} != {expected_shape}"
            )
        if not torch.isfinite(features).all():
            raise RuntimeError("DINOv2 produced non-finite features")
        return features.float().cpu().numpy()


def _metadata_matches(attrs: Any, expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if attrs.get(key) != expected_value:
            return False
    return True


def _describe_mismatch(attrs: Any, expected: dict[str, Any]) -> str:
    differences = {
        key: {"found": attrs.get(key), "expected": value}
        for key, value in expected.items()
        if attrs.get(key) != value
    }
    return json.dumps(differences, ensure_ascii=False, sort_keys=True)


def _prepare_incomplete_array(
    episode: zarr.Group,
    camera_name: str,
    episode_length: int,
    feature_shape: tuple[int, ...],
    feature_suffix: str,
    feature_chunk_length: int,
    compressor: Any,
    metadata: dict[str, Any],
    overwrite: bool,
) -> tuple[zarr.Array | None, int]:
    final_name = f"{camera_name}{feature_suffix}"
    incomplete_name = f"{final_name}{INCOMPLETE_SUFFIX}"
    expected_shape = (episode_length, *feature_shape)

    if final_name in episode:
        final_array = episode[final_name]
        final_valid = (
            final_array.shape == expected_shape
            and np.dtype(final_array.dtype) == FEATURE_DTYPE
            and bool(final_array.attrs.get("complete", False))
            and _metadata_matches(final_array.attrs, metadata)
        )
        if final_valid and not overwrite:
            return None, episode_length
        if not overwrite:
            raise RuntimeError(
                f"{episode.path}/{final_name} exists but is incompatible: "
                f"shape={final_array.shape}, dtype={final_array.dtype}, "
                f"metadata_diff={_describe_mismatch(final_array.attrs, metadata)}. "
                "Pass --overwrite to replace it."
            )

    if overwrite and incomplete_name in episode:
        del episode[incomplete_name]

    if incomplete_name in episode:
        incomplete = episode[incomplete_name]
        valid = (
            incomplete.shape == expected_shape
            and np.dtype(incomplete.dtype) == FEATURE_DTYPE
            and _metadata_matches(incomplete.attrs, metadata)
        )
        if not valid:
            raise RuntimeError(
                f"Cannot resume incompatible {episode.path}/{incomplete_name}: "
                f"shape={incomplete.shape}, dtype={incomplete.dtype}, "
                f"metadata_diff={_describe_mismatch(incomplete.attrs, metadata)}. "
                "Pass --overwrite to discard it."
            )
        frames_written = int(incomplete.attrs.get("frames_written", 0))
        if not 0 <= frames_written <= episode_length:
            raise RuntimeError(
                f"Invalid frames_written={frames_written} in "
                f"{episode.path}/{incomplete_name}"
            )
        return incomplete, frames_written

    incomplete = episode.create_dataset(
        incomplete_name,
        shape=expected_shape,
        chunks=(min(feature_chunk_length, episode_length), *feature_shape),
        dtype=FEATURE_DTYPE,
        compressor=compressor,
        overwrite=False,
    )
    incomplete.attrs.update(metadata)
    incomplete.attrs.update(
        {
            "camera_name": camera_name,
            "complete": False,
            "frames_written": 0,
        }
    )
    return incomplete, 0


def _finalize_array(
    episode: zarr.Group, camera_name: str, feature_suffix: str
) -> None:
    final_name = f"{camera_name}{feature_suffix}"
    incomplete_name = f"{final_name}{INCOMPLETE_SUFFIX}"
    incomplete = episode[incomplete_name]
    incomplete.attrs["frames_written"] = int(incomplete.shape[0])
    incomplete.attrs["complete"] = True
    if final_name in episode:
        del episode[final_name]
    episode.move(incomplete_name, final_name)


def extract_dataset(
    dataset_path: Path,
    cameras: Iterable[str],
    model_name: str,
    input_size: int,
    batch_size: int,
    feature_chunk_length: int,
    device: torch.device,
    overwrite: bool,
    feature_kind: str,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if feature_chunk_length <= 0:
        raise ValueError("feature_chunk_length must be positive")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Zarr dataset not found: {dataset_path}")

    root = zarr.open_group(str(dataset_path), mode="a")
    episode_names = _episode_names(root)
    cameras = tuple(cameras)
    extractor = DinoV2FeatureExtractor(
        model_name, input_size, device, feature_kind
    )
    if feature_kind == "cls":
        feature_shape = (extractor.feature_dim,)
        feature_suffix = CLS_FEATURE_SUFFIX
        cache_attr_name = "dinov2_feature_cache"
    else:
        feature_shape = (
            1 + int(extractor.metadata["patch_token_num"]),
            extractor.feature_dim,
        )
        feature_suffix = PATCH_FEATURE_SUFFIX
        cache_attr_name = "dinov2_patch_feature_cache"
    compressor = zarr.Blosc(cname="lz4", clevel=5, shuffle=1)

    print(
        f"Dataset: {dataset_path}\n"
        f"Episodes: {len(episode_names)}\n"
        f"Cameras: {list(cameras)}\n"
        f"Model: {model_name}\n"
        f"Device: {device}\n"
        f"Feature kind: {feature_kind}\n"
        f"Output: float32 {list(feature_shape)}",
        flush=True,
    )

    for episode_name in episode_names:
        episode = root[episode_name]
        lengths = {}
        for camera_name in cameras:
            if camera_name not in episode:
                raise KeyError(f"Missing {episode.path}/{camera_name}")
            images = episode[camera_name]
            if images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError(
                    f"{episode.path}/{camera_name} must have shape [T,H,W,3], "
                    f"got {images.shape}"
                )
            if np.dtype(images.dtype) != np.dtype(np.uint8):
                raise ValueError(
                    f"{episode.path}/{camera_name} must be uint8, got {images.dtype}"
                )
            lengths[camera_name] = int(images.shape[0])
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Camera lengths differ in {episode.path}: {lengths}"
            )
        episode_length = next(iter(lengths.values()))

        for camera_name in cameras:
            feature_array, start = _prepare_incomplete_array(
                episode=episode,
                camera_name=camera_name,
                episode_length=episode_length,
                feature_shape=feature_shape,
                feature_suffix=feature_suffix,
                feature_chunk_length=feature_chunk_length,
                compressor=compressor,
                metadata=extractor.metadata,
                overwrite=overwrite,
            )
            final_name = f"{camera_name}{feature_suffix}"
            if feature_array is None:
                print(f"Skipping complete {episode.path}/{final_name}", flush=True)
                continue

            progress = tqdm(
                total=episode_length,
                initial=start,
                desc=f"{episode_name}/{camera_name}",
                unit="frame",
            )
            images = episode[camera_name]
            for begin in range(start, episode_length, batch_size):
                end = min(begin + batch_size, episode_length)
                feature_array[begin:end] = extractor(images[begin:end])
                feature_array.attrs["frames_written"] = end
                progress.update(end - begin)
            progress.close()
            _finalize_array(episode, camera_name, feature_suffix)

    root.attrs[cache_attr_name] = {
        **extractor.metadata,
        "camera_names": list(cameras),
        "array_suffix": feature_suffix,
    }
    print("Feature extraction complete.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add float32 DINOv2 CLS or CLS+patch features to an existing "
            "converted RMBench Zarr dataset."
        )
    )
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument(
        "--cameras",
        type=_parse_cameras,
        default=CAMERA_NAMES,
        help=(
            "Comma-separated cameras (default: "
            "head_camera,left_camera,right_camera)."
        ),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument(
        "--feature-kind",
        choices=("cls", "patches"),
        default="cls",
        help=(
            "cls stores one token per frame for compact three-view training; "
            "patches stores CLS plus all spatial patches for full-patch "
            "single-view training."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-chunk-length", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute complete or incomplete feature arrays.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_dataset(
        dataset_path=args.dataset_path.expanduser().resolve(),
        cameras=args.cameras,
        model_name=args.model_name,
        input_size=args.input_size,
        batch_size=args.batch_size,
        feature_chunk_length=args.feature_chunk_length,
        device=_resolve_device(args.device),
        overwrite=args.overwrite,
        feature_kind=args.feature_kind,
    )


if __name__ == "__main__":
    main()
