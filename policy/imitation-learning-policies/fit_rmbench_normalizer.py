"""Fit the existing dataset normalizer with RMBench-specific overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
from hydra import initialize_config_dir
from omegaconf import OmegaConf

from imitation_learning.datasets.base_dataset import BaseDataset
from imitation_learning.utils.config_utils import compose_hydra_config
from robot_utils.config_utils import register_resolvers


POLICY_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = POLICY_ROOT / "imitation_learning" / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("action_dim", type=int)
    parser.add_argument("--quantile", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if args.action_dim <= 0:
        raise ValueError("action_dim must be positive")

    register_resolvers()
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR), version_base=None
    ):
        # Normalizer statistics are per state/action dimension. Use the
        # existing single-trajectory dataset path so the temporal dimension is
        # reduced exactly as expected by BaseDataset.calc_stats(). The fitted
        # normalizer is shared by the memory policy.
        cfg = compose_hydra_config(
            "rmbench", "diffusion_transformer"
        )

    OmegaConf.set_struct(cfg, False)
    dataset_cfg = cfg.workspace.train_dataset
    dataset_cfg.root_dir = str(dataset_path.parent)
    dataset_cfg.compressed_dir = str(dataset_path.parent)
    dataset_cfg.normalizer_dir = str(dataset_path.parent)
    dataset_cfg.name = dataset_path.name.removesuffix(".zarr")
    dataset_cfg.source_data_meta.agent_pos.shape = [args.action_dim]
    dataset_cfg.source_data_meta.action.shape = [args.action_dim]
    dataset_cfg.output_data_meta.agent_pos.shape = [args.action_dim]
    dataset_cfg.output_data_meta.action.shape = [args.action_dim]
    # Image fields use identity normalization and do not participate in the
    # fitted statistics. Keep normalizer fitting independent of whether the
    # optional DINOv2 feature cache has already been generated.
    dataset_cfg.use_cached_image_features = False

    dataset: BaseDataset = hydra.utils.instantiate(dataset_cfg)
    dataset.fit_normalizer(quantile=args.quantile)


if __name__ == "__main__":
    main()
