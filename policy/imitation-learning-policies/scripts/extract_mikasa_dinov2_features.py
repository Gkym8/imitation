"""Cache the fixed third-person Mikasa camera with the shared DINOv2 extractor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

POLICY_ROOT = Path(__file__).resolve().parents[1]
if str(POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICY_ROOT))

from scripts.extract_rmbench_dinov2_features import (
    DEFAULT_INPUT_SIZE,
    DEFAULT_MODEL_NAME,
    _resolve_device,
    extract_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-chunk-length", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract_dataset(
        dataset_path=args.dataset_path.expanduser().resolve(),
        cameras=("third_person_camera",),
        model_name=args.model_name,
        input_size=args.input_size,
        batch_size=args.batch_size,
        feature_chunk_length=args.feature_chunk_length,
        device=_resolve_device(args.device),
        overwrite=args.overwrite,
        feature_kind="patches",
    )


if __name__ == "__main__":
    main()
