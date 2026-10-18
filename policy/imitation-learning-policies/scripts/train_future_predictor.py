"""Train a standalone, action-conditioned future-patch predictor.

Example: python scripts/train_future_predictor.py --dataset /path/task.zarr
         --output-dir outputs/future_predictor training.epochs=100
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import tqdm

POLICY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_ROOT))

from imitation_learning.datasets.future_transition_dataset import FutureTransitionDataset
from imitation_learning.models.future_prediction.state_action_future_predictor import (
    StateActionFuturePredictor, load_future_predictor_checkpoint,
    save_future_predictor_checkpoint,
)
from scripts.future_predictor_distributed import DistributedWeightedSampler, distributed_training


def resolve_config(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML merged over the standalone defaults")
    parser.add_argument("--dataset", action="append", help="Zarr path; repeat for compatible tasks")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Use every *.zarr in --data-dir when no dataset/sources are specified")
    parser.add_argument("--data-dir", type=Path,
                        default=POLICY_ROOT.parents[1] / "data/imitation_learning_policies")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate datasets and print the plan without training or writing checkpoints")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", help="cpu, cuda:0, or auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path, help="Resume a last.pt with the same config and data")
    args, overrides = parser.parse_known_args(argv)
    if any("=" not in value or value.startswith("--") for value in overrides):
        parser.error("Additional overrides must use dotted.key=value syntax")
    cfg = OmegaConf.load(POLICY_ROOT / "imitation_learning/configs/train_future_predictor.yaml")
    if args.resume:
        default_swanlab = cfg.swanlab
        # A changed default YAML must not silently change an existing run.
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        cfg = OmegaConf.create(saved["training_state"]["config"])
        if "per_process_batch_size" not in saved["training_state"]:
            # Earlier trainers stored a GLOBAL loader.batch_size. Preserve the
            # actual update size when migrating it to the per-process setting.
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if cfg.loader.batch_size % world_size:
                raise ValueError("The checkpoint's global batch size must be divisible by the process count")
            cfg.loader.batch_size //= world_size
        cfg.output_dir = str(args.resume.resolve().parent)
        if "swanlab" not in cfg:
            # Migrate logging only; preserve the original data/model/optimizer.
            cfg.swanlab = default_swanlab
            cfg.tensorboard = {"enabled": False}
        del saved
    if args.config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if args.dataset:
        cfg.data.sources = [{"path": path} for path in args.dataset]
    elif args.all_datasets and not cfg.data.sources:
        paths = sorted(path.resolve() for path in args.data_dir.expanduser().glob("*.zarr") if path.is_dir())
        if not paths:
            parser.error(f"No converted *.zarr datasets found in {args.data_dir}")
        cfg.data.sources = [{"path": str(path)} for path in paths]
    if args.all_datasets and not args.resume and not cfg.data.control_interface:
        # This launcher targets RMBench's converted joint-target datasets.
        # Dimension/alignment/cache compatibility is still checked by the loader.
        cfg.data.control_interface = "rmbench_joint_target_v1"
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if cfg.output_dir is None:
        cfg.output_dir = (str(args.resume.resolve().parent) if args.resume else
                          f"outputs/future_predictor/shared_{datetime.now():%Y%m%d_%H%M%S_%f}")
    if args.device:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    cfg.resume = str(args.resume.resolve()) if args.resume else None
    cfg.dry_run = args.dry_run
    return OmegaConf.to_container(cfg, resolve=True)


def build_datasets(cfg):
    train_dataset = FutureTransitionDataset(split="train", **cfg["data"])
    validation_dataset = FutureTransitionDataset(split="validation", **cfg["data"])
    if train_dataset.metadata() != validation_dataset.metadata():
        raise ValueError("Train/validation feature or data contracts differ")
    return train_dataset, validation_dataset


def source_summary(train_dataset, validation_dataset):
    return [dict(name=source["name"], path=source["path"],
                 selected_episodes=len(source["train_episodes"]) + len(source["validation_episodes"]),
                 train_episodes=len(source["train_episodes"]),
                 validation_episodes=len(source["validation_episodes"]),
                 train_samples=train_dataset.source_counts[index],
                 validation_samples=validation_dataset.source_counts[index])
            for index, source in enumerate(train_dataset.sources)]


def preview(cfg):
    train_dataset, validation_dataset = build_datasets(cfg)
    print(json.dumps(dict(dry_run=True, datasets=source_summary(train_dataset, validation_dataset),
                          train_samples=len(train_dataset), validation_samples=len(validation_dataset),
                          model=cfg["model"], control_interface=train_dataset.control_interface,
                          epochs=cfg["training"]["epochs"],
                          save_every_epochs=cfg["training"].get("save_every_epochs", 1),
                          output_dir=str(Path(cfg["output_dir"]).resolve())), indent=2), flush=True)


def to_device(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {name: batch[name].to(device, non_blocking=True)
            for name in ("current_features", "proprio", "actions", "target_features")}


def predict(model: StateActionFuturePredictor, batch: dict) -> torch.Tensor:
    return model(batch["current_features"], batch["proprio"], batch["actions"])


def build_optimizer(model, settings):
    # Missing keys keep the first standalone trainer's AdamW behavior available
    # to callers passing an old resolved config directly.
    name = settings.get("optimizer", "adamw").lower()
    classes = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if name not in classes:
        raise ValueError("training.optimizer must be adam or adamw")
    return classes[name](model.parameters(), lr=settings["learning_rate"],
                         weight_decay=settings["weight_decay"],
                         betas=tuple(settings.get("adam_betas", (0.9, 0.999))),
                         eps=settings.get("adam_eps", 1e-8))


@contextmanager
def tensorboard_writer(output, cfg, resume_step=None):
    # Keep logging settings outside `training`: changing logging must not affect
    # checkpoint compatibility. TensorBoard is now an optional legacy backend.
    settings = cfg.get("tensorboard", {})
    if not settings.get("enabled", False):
        yield None
        return
    interval = settings.get("log_every_steps", 100)
    if not isinstance(interval, int) or interval <= 0:
        raise ValueError("tensorboard.log_every_steps must be a positive integer")
    flush_secs = settings.get("flush_secs", 30)
    if flush_secs <= 0:
        raise ValueError("tensorboard.flush_secs must be positive")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise ImportError("TensorBoard is enabled; install it with `python -m pip install tensorboard` "
                          "or set tensorboard.enabled=false") from error
    writer = SummaryWriter(str(output / "tensorboard"), flush_secs=flush_secs,
                           # Retain the last completed step; hide any events
                           # emitted by an interrupted, uncheckpointed epoch.
                           purge_step=resume_step + 1 if resume_step is not None else None)
    print(f"TensorBoard log directory: {output / 'tensorboard'}", flush=True)
    try:
        yield writer
    finally:
        writer.close()


def swanlab_run_info(run, cfg):
    settings = cfg.get("swanlab", {})
    return dict(id=run.id, project=settings.get("project", "rmbench_future_predictor"),
                workspace=settings.get("workspace"), mode=settings.get("mode", "online"))


@contextmanager
def swanlab_run(output, cfg, previous_run=None):
    settings = cfg.get("swanlab", {})
    mode = settings.get("mode", "online")
    if not settings.get("enabled", True) or mode == "disabled":
        yield None
        return
    interval = settings.get("log_every_steps", 100)
    if type(interval) is not int or interval <= 0:
        raise ValueError("swanlab.log_every_steps must be a positive integer")
    if mode not in {"online", "offline", "local"}:
        raise ValueError("swanlab.mode must be online, offline, local or disabled")
    # Initializing a logger must not perturb the model's dropout/sampling RNG.
    torch_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
    try:
        try:
            import swanlab
        except ImportError as error:
            raise ImportError("SwanLab is enabled; install `swanlab==0.9.8` in RMBench/RMBench_90 "
                              "or set swanlab.enabled=false") from error
        project = settings.get("project", "rmbench_future_predictor")
        workspace = settings.get("workspace")
        kwargs = dict(project=project, workspace=workspace, mode=mode,
                      name=settings.get("experiment_name") or output.name,
                      log_dir=str(output / "swanlog"), config=cfg,
                      settings=swanlab.Settings(**dict({"interactive": False}, **settings.get("settings", {}))))
        if (previous_run and mode == previous_run.get("mode") == "online"
                and project == previous_run.get("project")
                and workspace == previous_run.get("workspace")):
            kwargs.update(id=previous_run["id"], resume="allow")
        run = swanlab.init(**kwargs)
    finally:
        torch.set_rng_state(torch_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
    state, error_message = "success", None
    try:
        (output / "swanlab_run.json").write_text(
            json.dumps(swanlab_run_info(run, cfg), indent=2), encoding="utf-8")
        yield run
    except BaseException as error:
        state = "aborted" if isinstance(error, KeyboardInterrupt) else "crashed"
        error_message = str(error) or type(error).__name__
        raise
    finally:
        run.finish(state=state, error=error_message)


def epoch_log_data(metrics, sources):
    validation = metrics["validation"]
    data = {"epoch/train_mse": metrics["train_mse"],
            "epoch/validation_mse": validation["mse"],
            "epoch/validation_macro_mse": validation["macro_mse"],
            "progress/epoch": metrics["epoch"]}
    for name, value in validation.items():
        if name not in {"mse", "macro_mse", "per_source_mse"}:
            data[f"validation/{name}"] = value
    for index, value in validation["per_source_mse"].items():
        data[f"validation/task_mse/{sources[int(index)]['name']}"] = value
    return data


def log_epoch_metrics(writer, metrics, sources):
    for name, value in epoch_log_data(metrics, sources).items():
        writer.add_scalar(name, value, metrics["global_step"])
    writer.flush()


def batch_progress(loader, *, desc, max_batches=None, settings=None):
    settings = settings or {}
    interval = settings.get("mininterval", 0.1)
    if interval < 0:
        raise ValueError("progress.mininterval must be nonnegative")
    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    return tqdm.tqdm(islice(loader, max_batches), total=total, desc=desc,
                     leave=False, mininterval=interval, dynamic_ncols=True,
                     disable=not settings.get("enabled", True))


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None, progress=None) -> dict:
    model.eval()
    totals, counts, per_source = {}, {}, {}

    def accumulate(name, values):
        totals[name] = totals.get(name, 0.0) + values.sum().item()
        counts[name] = counts.get(name, 0) + values.numel()

    with batch_progress(loader, desc="Validate", max_batches=max_batches, settings=progress) as bar:
        for raw in bar:
            batch = to_device(raw, device)
            target = batch["target_features"]
            prediction = predict(model, batch)
            mse = (prediction - target).square().mean(dim=(1, 2))
            accumulate("mse", mse)
            accumulate("copy_current_mse", (batch["current_features"] - target).square().mean(dim=(1, 2)))
            for key, actions in (("zero_actions_mse", torch.zeros_like(batch["actions"])),
                                 ("reversed_actions_mse", batch["actions"].flip(1))):
                other = model(batch["current_features"], batch["proprio"], actions)
                accumulate(key, (other - target).square().mean(dim=(1, 2)))
            # Swap actions only within the same task, avoiding mixed-source boundary batches.
            for source in raw["source_index"].unique().tolist():
                mask = raw["source_index"] == source
                indices = mask.nonzero(as_tuple=True)[0].to(device)
                source_mse = mse[indices]
                item = per_source.setdefault(str(source), {"sum": 0.0, "count": 0})
                item["sum"] += source_mse.sum().item()
                item["count"] += len(indices)
                if len(indices) > 1:
                    other = model(batch["current_features"][indices], batch["proprio"][indices],
                                  batch["actions"][indices].roll(1, dims=0))
                    accumulate("shuffled_actions_mse", (other - target[indices]).square().mean(dim=(1, 2)))
            bar.set_postfix(mse=f"{totals['mse'] / counts['mse']:.4f}", refresh=False)
    if not counts:
        raise ValueError("Validation produced no batches")
    metrics = {key: totals[key] / counts[key] for key in totals}
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Nonfinite validation metric")
    metrics["per_source_mse"] = {key: value["sum"] / value["count"] for key, value in per_source.items()}
    metrics["macro_mse"] = float(np.mean(list(metrics["per_source_mse"].values())))
    metrics["samples"] = counts["mse"]
    metrics["shuffled_samples"] = counts.get("shuffled_actions_mse", 0)
    return metrics


def train(cfg: dict) -> Path:
    with distributed_training(cfg["device"]) as distributed:
        # Rank zero supplies one timestamped directory and one resolved config.
        return train_worker(distributed.broadcast(cfg), distributed)


def train_worker(cfg: dict, distributed) -> Path:
    settings = cfg["training"]
    if settings["epochs"] <= 0 or cfg["loader"]["batch_size"] <= 0:
        raise ValueError("epochs and batch_size must be positive")
    per_process_batch_size = cfg["loader"]["batch_size"]
    global_batch_size = per_process_batch_size * distributed.world_size
    for name in ("max_train_batches", "max_val_batches"):
        if settings[name] is not None and settings[name] <= 0:
            raise ValueError(f"{name} must be null or positive")
    if settings["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    if settings["grad_clip_norm"] is not None and settings["grad_clip_norm"] <= 0:
        raise ValueError("grad_clip_norm must be null or positive")
    save_every_epochs = settings.get("save_every_epochs", 1)
    if not isinstance(save_every_epochs, int) or save_every_epochs <= 0:
        raise ValueError("save_every_epochs must be a positive integer")
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(settings["cpu_threads"])
    device = distributed.device
    output = Path(cfg["output_dir"]).expanduser().resolve()

    def check_output():
        # Never silently overwrite another run, including partial metrics.
        if not cfg["resume"] and output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output}; choose a new directory or --resume")
        if cfg["resume"] and Path(cfg["resume"]).resolve() != output / "last.pt":
            raise ValueError("Resume must use last.pt in the selected output directory")

    distributed.on_main(check_output)
    # Validate/index JuiceFS metadata once; dataset pickling excludes open Zarr
    # handles and all feature arrays. Each loader still reads its own samples.
    train_dataset, validation_dataset = distributed.on_main(lambda: build_datasets(cfg))
    metadata = train_dataset.metadata()
    model_config = dict(
        feature_dim=metadata["feature_contract"]["feature_dim"],
        patch_token_num=metadata["feature_contract"]["patch_token_num"],
        camera_names=metadata["feature_contract"]["camera_names"],
        action_dim=train_dataset.action_dim, proprio_dim=train_dataset.proprio_dim,
        execution_chunk_length=cfg["data"]["execution_chunk_length"],
        feedforward_dim=None, activation="gelu", norm_first=True, final_norm=True,
    )
    model_config.update(cfg["model"])
    checkpoint = None
    if cfg["resume"]:
        model, checkpoint = load_future_predictor_checkpoint(cfg["resume"], device, freeze=False)
        if checkpoint["metadata"] != metadata or model.model_config != model_config:
            raise ValueError("Resume data, split, representation or model config changed")
        previous = checkpoint["training_state"]["config"]
        training_contract = lambda value: {key: item for key, item in value.items() if key != "log_every_steps"}
        loader_contract = lambda value: {key: item for key, item in value.items() if key != "batch_size"}
        previous_global_batch = checkpoint["training_state"].get("global_batch_size", previous["loader"]["batch_size"])
        if previous_global_batch != global_batch_size:
            raise ValueError(f"Resume global batch size changed: checkpoint={previous_global_batch}, "
                             f"current={global_batch_size}; loader.batch_size is per process, "
                             "adjust it for the selected GPU/process count")
        if (training_contract(previous["training"]) != training_contract(settings)
                or loader_contract(previous["loader"]) != loader_contract(cfg["loader"]) or previous["seed"] != seed):
            raise ValueError("Resume training settings changed; use the original configuration")
    else:
        model = StateActionFuturePredictor(**model_config)
        model.set_normalizer(distributed.on_main(train_dataset.fit_normalizer))
        model.to(device)
    optimizer = build_optimizer(model, settings)
    start_epoch, global_step, best_metric = 0, 0, float("inf")
    if checkpoint is not None:
        state = checkpoint["training_state"]
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, global_step, best_metric = state["epoch"] + 1, state["global_step"], state["best_metric"]
    if start_epoch >= settings["epochs"]:
        raise ValueError("The requested run is already complete")
    if distributed.world_size > 1:
        training_model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            # The shared predictor also defines a history type embedding that
            # this standalone state/action model intentionally never consumes.
            find_unused_parameters=True)
    else:
        training_model = model
    if checkpoint is not None:
        distributed.restore_rng(checkpoint["training_state"], seed)
    elif distributed.rank:
        torch.manual_seed(seed + distributed.rank)

    def write_run_metadata():
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(cfg)), encoding="utf-8")
        (output / "data_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    distributed.on_main(write_run_metadata)
    loader_config = dict(cfg["loader"])
    loader_config.setdefault("pin_memory", device.type == "cuda")
    loader_config["drop_last"] = False
    loader_config["generator"] = torch.Generator().manual_seed(seed + distributed.rank)
    if loader_config["num_workers"] > 0:
        loader_config.setdefault("persistent_workers", True)
        loader_config["prefetch_factor"] = 2
        if distributed.world_size > 1:
            # NCCL and Gloo worker threads must not be inherited through fork.
            loader_config["multiprocessing_context"] = "spawn"
    else:
        loader_config["persistent_workers"] = False
    validation_loader = (DataLoader(validation_dataset, shuffle=False, **loader_config)
                         if distributed.is_main else None)
    train_weights = train_dataset.sampling_weights()
    if distributed.is_main:
        print(json.dumps(dict(device=str(device), world_size=distributed.world_size,
                              global_batch_size=global_batch_size,
                              per_process_batch_size=per_process_batch_size,
                              train_samples=len(train_dataset), validation_samples=len(validation_dataset),
                              datasets=source_summary(train_dataset, validation_dataset),
                              model_config=model_config, output_dir=str(output)), ensure_ascii=False), flush=True)
    previous_run = checkpoint["training_state"].get("swanlab_run") if checkpoint is not None else None
    with (tensorboard_writer(output, cfg, global_step if checkpoint is not None else None)
          if distributed.is_main else nullcontext(None)) as writer, (
          swanlab_run(output, cfg, previous_run)
          if distributed.is_main else nullcontext(None)) as run:
        tb_interval = cfg.get("tensorboard", {}).get("log_every_steps", 100)
        sw_interval = cfg.get("swanlab", {}).get("log_every_steps", 100)
        progress_settings = dict(cfg.get("progress", {}))
        if not distributed.is_main:
            progress_settings["enabled"] = False
        for epoch in range(start_epoch, settings["epochs"]):
            sampler = DistributedWeightedSampler(
                train_weights, len(train_dataset), distributed.rank, distributed.world_size, seed + epoch)
            train_loader = DataLoader(train_dataset, sampler=sampler, **loader_config)
            training_model.train()
            loss_sum, samples = 0.0, 0
            waiting_at = time.perf_counter()
            with batch_progress(train_loader, desc=f"Train {epoch + 1}/{settings['epochs']}",
                                max_batches=settings["max_train_batches"], settings=progress_settings) as bar:
                for raw in bar:
                    batch_ready_at = time.perf_counter()
                    data_wait = batch_ready_at - waiting_at
                    batch = to_device(raw, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.mse_loss(predict(training_model, batch), batch["target_features"])
                    mean_loss = distributed.mean_loss(loss)
                    if not math.isfinite(mean_loss):
                        raise FloatingPointError("Nonfinite future prediction loss")
                    loss.backward()
                    max_norm = settings["grad_clip_norm"]
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                  float("inf") if max_norm is None else max_norm,
                                                  error_if_nonfinite=True)
                    optimizer.step()
                    count = len(batch["actions"]) * distributed.world_size
                    loss_sum += mean_loss * count
                    samples += count
                    global_step += 1
                    if writer is not None and (global_step == 1 or global_step % tb_interval == 0):
                        writer.add_scalar("train/mse", mean_loss, global_step)
                        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    if run is not None and (global_step == 1 or global_step % sw_interval == 0):
                        run.log({"train/mse": mean_loss,
                                 "train/learning_rate": optimizer.param_groups[0]["lr"]}, step=global_step)
                    bar.set_postfix(loss=f"{mean_loss:.4f}", data_s=f"{data_wait:.2f}",
                                    step_s=f"{time.perf_counter() - batch_ready_at:.2f}", refresh=False)
                    waiting_at = time.perf_counter()
            if not samples:
                raise ValueError("Training produced no batches")
            validation = distributed.on_main(lambda: evaluate(
                model, validation_loader, device, settings["max_val_batches"], progress=progress_settings))
            metrics = dict(epoch=epoch + 1, global_step=global_step, train_mse=loss_sum / samples,
                           validation=validation)
            metric = metrics["validation"]["macro_mse"]
            improved = metric < best_metric
            best_metric = min(best_metric, metric)
            rng_states = distributed.rng_states()
            state = dict(epoch=epoch, global_step=global_step, best_metric=best_metric,
                         optimizer=optimizer.state_dict(), config=cfg, metrics=metrics,
                         swanlab_run=swanlab_run_info(run, cfg) if run is not None else None,
                         torch_rng=rng_states[0]["torch_rng"], cuda_rng=rng_states[0]["cuda_rng"],
                         world_size=distributed.world_size, rank_rng_states=rng_states,
                         per_process_batch_size=per_process_batch_size, global_batch_size=global_batch_size)

            def finish_epoch():
                if (epoch + 1) % save_every_epochs == 0 or epoch + 1 == settings["epochs"]:
                    save_future_predictor_checkpoint(output / f"epoch_{epoch + 1:04d}.pt", model, metadata, state)
                save_future_predictor_checkpoint(output / "last.pt", model, metadata, state)
                if improved:
                    save_future_predictor_checkpoint(output / "best.pt", model, metadata, state)
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, allow_nan=False) + "\n")
                if writer is not None:
                    log_epoch_metrics(writer, metrics, train_dataset.sources)
                if run is not None:
                    run.log(epoch_log_data(metrics, train_dataset.sources), step=global_step)

            distributed.on_main(finish_epoch)
    return output / "best.pt"


if __name__ == "__main__":
    config = resolve_config()
    if config["dry_run"]:
        if int(os.environ.get("RANK", "0")) == 0:
            preview(config)
    else:
        train(config)
