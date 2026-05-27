"""Simple on-disk cache for CLIP video frame embeddings."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np


def make_video_embedding_cache_key(
    video_id: str,
    model_name: str,
    fps: float,
) -> str:
    """Build a stable filesystem-safe cache key for one video/model/FPS setup."""

    raw_key = f"{video_id}__{model_name}__fps_{float(fps):g}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw_key).strip("_")


def save_video_embeddings_cache(
    cache_dir: str | Path,
    key: str,
    frame_timestamps: list[float],
    image_embeddings: Any,
    metadata: dict,
) -> Path:
    """Save frame timestamps, image embeddings, and metadata as a `.npz` file."""

    cache_path = _cache_path(cache_dir, key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings_array = _to_numpy_array(image_embeddings)
    np.savez_compressed(
        cache_path,
        frame_timestamps=np.asarray(frame_timestamps, dtype=np.float32),
        image_embeddings=embeddings_array.astype(np.float32, copy=False),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )

    return cache_path


def load_video_embeddings_cache(
    cache_dir: str | Path,
    key: str,
) -> dict | None:
    """Load cached frame timestamps and image embeddings if a cache file exists."""

    cache_path = _cache_path(cache_dir, key)
    if not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as data:
        metadata_json = str(data["metadata_json"].item())
        return {
            "frame_timestamps": data["frame_timestamps"].astype(float).tolist(),
            "image_embeddings": data["image_embeddings"],
            "metadata": json.loads(metadata_json),
            "path": cache_path,
        }


def estimate_cache_file_size(path: str | Path) -> int:
    """Return cache file size in bytes."""

    return Path(path).stat().st_size


def _cache_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.npz"


def _to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()

    return np.asarray(value)
