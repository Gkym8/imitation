"""Prepare Mikasa normalization statistics without extracting image features.

Existing absolute-action statistics are reused. Missing fields are fitted from
the configured training split, with a backup before updating an existing file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.train_mikasa import dataset_path, ensure_normalizer, load_config


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", type=Path, help="Task directory or its episode_data.zarr")
    parser.add_argument("task_name", help="Task YAML name, e.g. mikasa_shell_game_touch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalizer-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    path = args.dataset_path.expanduser().resolve()
    if path.name == "episode_data.zarr":
        root, name = path.parent.parent, path.parent.name
    else:
        root = path.parent
        name = path.stem if path.suffix == ".zarr" else path.name
    try:
        cfg = load_config(args.task_name)
        if name != cfg.workspace.train_dataset.name:
            raise ValueError(
                f"Dataset {name!r} does not match task {args.task_name!r}: "
                f"expected {cfg.workspace.train_dataset.name!r}"
            )
        dataset_path(root, name)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    cfg.seed = args.seed
    dataset = cfg.workspace.train_dataset
    dataset.root_dir = str(root)
    dataset.compressed_dir = ""
    dataset.normalizer_dir = str(
        args.normalizer_dir.expanduser().resolve() if args.normalizer_dir else root
    )
    result = ensure_normalizer(cfg)
    print(f"Normalizer ready: {result}")
    return result


if __name__ == "__main__":
    main()
