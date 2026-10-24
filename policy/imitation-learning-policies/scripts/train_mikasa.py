"""Prepare a Mikasa dataset and launch the shared streaming-memory trainer."""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

POLICY_ROOT = Path(__file__).resolve().parents[1]
if str(POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICY_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from imitation_learning.utils.config_utils import compose_hydra_config
from robot_utils.config_utils import register_resolvers


def dataset_path(data_root: Path, name: str) -> Path:
    for candidate in (
        data_root / name / "episode_data.zarr",
        data_root / f"{name}.zarr",
        data_root / name,
    ):
        if (candidate / ".zgroup").is_file():
            return candidate
    raise FileNotFoundError(f"No Zarr dataset for {name!r} under {data_root}")


def load_config(task_name: str, overrides: list[str] | None = None) -> DictConfig:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task_name):
        raise ValueError("task_name must be a task YAML name without its extension")
    if not (POLICY_ROOT / "imitation_learning/configs/task" / f"{task_name}.yaml").is_file():
        raise ValueError(f"Unknown Mikasa task: {task_name}")
    if not OmegaConf.has_resolver("eval"):
        register_resolvers()
    with hydra.initialize_config_dir(
        config_dir=str(POLICY_ROOT / "imitation_learning/configs"), version_base=None
    ), contextlib.redirect_stdout(io.StringIO()):
        cfg = compose_hydra_config(task_name, "diffusion_memory_transformer")
    OmegaConf.set_struct(cfg, False)
    overrides = overrides or []
    if any("=" not in item or item.startswith(("+", "~")) for item in overrides):
        raise ValueError("--override expects KEY=VALUE without Hydra +/~ prefixes")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if cfg.dataset_type != "mikasa":
        raise ValueError("This entry point requires a Mikasa task YAML")
    return cfg


def validate_streaming_config(cfg: DictConfig) -> None:
    dataset, model = cfg.workspace.train_dataset, cfg.workspace.model
    if dataset.anchor_sampling_mode != "streaming":
        raise ValueError("Mikasa memory training requires anchor_sampling_mode=streaming")
    if dataset.down_sample_steps != 1 or not (
        dataset.traj_interval_min == dataset.traj_interval_max == model.execution_chunk_length
    ):
        raise ValueError("Mikasa streaming interval must match execution_chunk_length, with down_sample_steps=1")
    if not 1 <= model.execution_chunk_length <= model.action_length:
        raise ValueError("execution_chunk_length must be in [1, action_length]")
    if dataset.use_cached_image_features:
        if not cfg.visual_encoder.frozen:
            raise ValueError("Cached features require visual_encoder.frozen=true")
        if not str(cfg.visual_encoder._target_).endswith(".SharedDinoV2ImageEncoder"):
            raise ValueError("The Mikasa feature cache requires SharedDinoV2ImageEncoder")
        if dataset.cached_image_feature_kind not in {"patches", "auto"}:
            raise ValueError("Mikasa current/history tokens require a CLS+patch cache")


def choose_data_root(cfg: DictConfig, requested: str | None) -> Path:
    if not requested and not OmegaConf.is_missing(cfg.workspace.train_dataset, "root_dir"):
        requested = str(cfg.workspace.train_dataset.root_dir)
    if requested:
        root = Path(requested).expanduser().resolve()
        dataset_path(root, cfg.workspace.train_dataset.name)
        return root
    candidates = [POLICY_ROOT / "data/mikasa"]
    if len(POLICY_ROOT.parents) > 1:
        candidates.append(POLICY_ROOT.parents[1] / "data/mikasa")
    if len(POLICY_ROOT.parents) > 2:
        candidates.append(POLICY_ROOT.parents[2] / "gated-memory-policy/data/mikasa-zarr-250")
    for root in candidates:
        try:
            dataset_path(root, cfg.workspace.train_dataset.name)
        except FileNotFoundError:
            continue
        return root.resolve()
    raise FileNotFoundError("Set --data-root or MIKASA_DATA_DIR to the directory containing Mikasa datasets")


def ensure_feature_cache(cfg: DictConfig, path: Path, batch_size: int, device: str) -> None:
    import zarr
    from imitation_learning.datasets.cached_image_dataset import _validate_cached_feature_store

    dataset = cfg.workspace.train_dataset
    root = zarr.open_group(str(path), mode="r")
    # Inspect metadata only. Complete caches avoid loading the visual backbone.
    lengths = {
        int(name.removeprefix("episode_")): int(root[name][dataset.image_keys[0]].shape[0])
        for name in root.group_keys() if re.fullmatch(r"episode_\d+", name)
    }
    view = SimpleNamespace(
        zarr_path=str(path), zarr_store=root,
        used_episode_indices=sorted(lengths), episode_frame_nums=lengths,
    )
    validation = dict(
        image_keys=list(dataset.image_keys), feature_dim=dataset.cached_image_feature_dim,
        model_name=dataset.cached_image_feature_model_name,
        input_size=dataset.cached_image_feature_input_size,
        feature_dtype=dataset.cached_image_feature_dtype, feature_kind="patches",
        patch_token_num=dataset.cached_image_feature_patch_token_num,
    )
    try:
        _validate_cached_feature_store(view, **validation)
        return
    except RuntimeError:
        pass
    from scripts.extract_rmbench_dinov2_features import _resolve_device, extract_dataset

    extract_dataset(
        dataset_path=path, cameras=tuple(dataset.image_keys),
        model_name=dataset.cached_image_feature_model_name,
        input_size=dataset.cached_image_feature_input_size, batch_size=batch_size,
        feature_chunk_length=10, device=_resolve_device(device),
        overwrite=False, feature_kind="patches",
    )
    # Reopen so validation sees final metadata after the extractor commits it.
    view.zarr_store = zarr.open_group(str(path), mode="r")
    _validate_cached_feature_store(view, **validation)


def ensure_normalizer(cfg: DictConfig) -> Path:
    from imitation_learning.datasets.normalizer import FixedNormalizer

    config = cfg.workspace.train_dataset
    location = Path(config.normalizer_dir)
    path = location if location.suffix == ".json" else location / f"{config.name}_normalizer.json"
    existing = {}
    if path.is_file():
        existing = json.loads(path.read_text())
        if not isinstance(existing, dict):
            raise ValueError(f"Normalizer must contain a field dictionary: {path}")
    metadata = OmegaConf.to_container(config.output_data_meta, resolve=True)
    missing = [
        name for name, meta in metadata.items()
        if meta["normalizer"] != "identity" and meta["length"] > 0 and name not in existing
    ]
    if not missing:
        return path
    # Fit the same configured training split, using only low-dimensional fields.
    # A single-trajectory view avoids weighting frames by overlapping streams.
    # Legacy Mikasa files may have action statistics but no proprio statistics.
    # Keep the saved action transform exactly, including absolute-joint targets.
    options = OmegaConf.to_container(config, resolve=True)
    options["_target_"] = "imitation_learning.datasets.mikasa_dataset.MikasaSingleTrajDataset"
    for key in ("traj_num", "traj_interval_min", "traj_interval_max", "anchor_sampling_mode",
                "sample_range", "split_dataloader_cfg", "episode_starting_idx_max"):
        options.pop(key, None)
    options.update(use_cached_image_features=False, normalizer_dir="", index_pool_size_per_episode=-1,
                   repeat_dataset_num=1)
    dataset = hydra.utils.instantiate(options)
    if not len(dataset):
        raise ValueError("No valid Mikasa frames for normalizer fitting")
    statistics = dataset.calc_stats_worker_single_process(0, len(dataset), missing)
    normalizer = FixedNormalizer({name: dataset.output_data_meta[name] for name in missing})
    normalizer.from_dict({
        name: {"min": values["min"].amin(dim=0), "max": values["max"].amax(dim=0)}
        for name, values in statistics.items()
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        backup = path.with_name(f"{path.name}.backup.{time.time_ns()}")
        shutil.copy2(path, backup)
    existing.update(normalizer.as_dict("list"))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(existing))
    temporary.replace(path)
    print(f"Added normalizer fields {missing}; existing field statistics were preserved.")
    return path


def training_command(python: str, gpu_ids: list[str], overrides: list[str]) -> list[str]:
    command = [python]
    if len(gpu_ids) > 1:
        with socket.socket() as listener:
            listener.bind(("", 0))
            port = listener.getsockname()[1]
        command += ["-m", "accelerate.commands.launch", "--multi_gpu", "--gpu_ids", "all",
                    "--num_machines", "1", "--num_processes", str(len(gpu_ids)),
                    "--main_process_port", str(port)]
    command += [str(POLICY_ROOT / "scripts/train_policy.py")]
    command += ["+" + item for item in overrides]
    return command


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_name", help="Task YAML name, e.g. mikasa_remember_color_3")
    parser.add_argument("gpu_ids", nargs="?", default="0")
    parser.add_argument("run_name", nargs="?", default=None)
    parser.add_argument("--data-root", default=os.environ.get("MIKASA_DATA_DIR"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--skip-prepare", action="store_true", help="Use existing caches and normalizer")
    parser.add_argument("--dry-run", action="store_true", help="Print launch command without preparing data or training")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    gpu_ids = args.gpu_ids.split(",")
    if not gpu_ids or any(not item.isdecimal() for item in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        parser.error("gpu_ids must contain distinct comma-separated GPU indices")
    if args.feature_batch_size <= 0:
        parser.error("feature-batch-size must be positive")
    cfg = load_config(args.task_name, args.override)
    data_root = choose_data_root(cfg, args.data_root)
    dataset = cfg.workspace.train_dataset
    compressed_dir = data_root if OmegaConf.is_missing(dataset, "compressed_dir") else dataset.compressed_dir
    normalizer_dir = data_root if OmegaConf.is_missing(dataset, "normalizer_dir") else dataset.normalizer_dir
    run_name = args.run_name or f"{cfg.workspace.train_dataset.name}-seed{args.seed}"
    defaults = [
        f"task_name={args.task_name}", "policy_name=diffusion_memory_transformer",
        "project_name=mikasa", "logger_project_name=mikasa", "train_server_name=localhost",
        f"seed={args.seed}", f"run_name={run_name}", f"run_name_suffix=_{run_name}",
        f"workspace.train_dataset.root_dir={data_root}",
        f"workspace.train_dataset.compressed_dir={compressed_dir}",
        f"workspace.train_dataset.normalizer_dir={normalizer_dir}",
    ]
    # Dict insertion order keeps a single final value for every Hydra override.
    settings = dict(item.split("=", 1) for item in defaults + args.override)
    overrides = [f"{key}={value}" for key, value in settings.items()]
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    validate_streaming_config(cfg)
    path = dataset_path(Path(cfg.workspace.train_dataset.root_dir), cfg.workspace.train_dataset.name)
    python = sys.executable
    command = training_command(python, gpu_ids, overrides)
    if args.dry_run:
        print("CUDA_VISIBLE_DEVICES=" + ",".join(gpu_ids) + " " + shlex.join(command))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    if not args.skip_prepare:
        if cfg.workspace.train_dataset.use_cached_image_features:
            ensure_feature_cache(cfg, path, args.feature_batch_size, "cuda:0")
        ensure_normalizer(cfg)
        # Preparation runs before DDP. Release its temporary GPU allocations
        # before starting the training subprocesses on the same devices.
        import torch
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    subprocess.run(command, cwd=POLICY_ROOT, check=True)


if __name__ == "__main__":
    main()
