"""Precision names shared by RMBench feature extraction and cache validation."""

from typing import Any, Mapping


def normalize_cached_image_feature_dtype(value: str) -> str:
    aliases = {
        "float32": "float32",
        "fp32": "float32",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
    }
    if isinstance(value, str) and value.strip().lower() in aliases:
        return aliases[value.strip().lower()]
    raise ValueError(
        "cached_image_feature_dtype must be float32/fp32 or bfloat16/bf16, "
        f"got {value!r}"
    )


def cached_feature_compute_dtype(metadata: Mapping[str, Any]) -> str:
    # Caches written before precision was configurable were extracted in FP32.
    return normalize_cached_image_feature_dtype(
        metadata.get("feature_compute_dtype", "float32")
    )
