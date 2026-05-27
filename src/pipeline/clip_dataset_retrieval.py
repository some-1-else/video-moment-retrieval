"""Dataset-level CLIP retrieval for small local video collections."""

from __future__ import annotations

from pathlib import Path
import time

from src.data.qvhighlights import MomentRetrievalSample
from src.evaluation.metrics import mean_iou, recall_at_iou
from src.models.clip_encoder import (
    compute_text_image_similarity,
    encode_images,
    encode_text,
    load_clip_model,
)
from src.models.embedding_cache import (
    estimate_cache_file_size,
    load_video_embeddings_cache,
    make_video_embedding_cache_key,
    save_video_embeddings_cache,
)
from src.retrieval.aggregation import aggregate_frame_scores_to_windows
from src.retrieval.scoring import select_best_window
from src.retrieval.windowing import generate_temporal_windows
from src.video.frame_extraction import extract_sampled_frames, get_video_duration


def run_clip_dataset_retrieval(
    samples: list[MomentRetrievalSample],
    videos_dir: str | Path,
    fps: float,
    window_size: float,
    stride: float,
    aggregation: str,
    model_name: str = "ViT-B/32",
    device: str = "auto",
    batch_size: int = 32,
    embeddings_cache_dir: str | Path | None = None,
    use_cache: bool = False,
) -> dict:
    """Run CLIP retrieval over a small local set of annotated videos."""

    if not samples:
        raise ValueError("samples must not be empty.")
    if fps <= 0:
        raise ValueError("fps must be a positive number.")
    if window_size <= 0:
        raise ValueError("window_size must be a positive number.")
    if stride <= 0:
        raise ValueError("stride must be a positive number.")

    start_time = time.perf_counter()
    model, preprocess, actual_device = load_clip_model(model_name, device=device)
    predictions: dict[str, list[float]] = {}
    ground_truth: dict[str, list[list[float]]] = {}
    prediction_windows: list[list[float]] = []
    ground_truth_windows: list[list[list[float]]] = []
    total_frames = 0
    cache_hits = 0
    cache_misses = 0
    embedding_cache_size_bytes = 0

    for sample in samples:
        video_path = resolve_video_path(videos_dir, sample.video_id)
        duration, frame_timestamps, image_embeddings, cache_hit, cache_size = (
            load_or_compute_video_image_embeddings(
                model=model,
                preprocess=preprocess,
                video_path=video_path,
                video_id=sample.video_id,
                model_name=model_name,
                fps=fps,
                device=actual_device,
                batch_size=batch_size,
                embeddings_cache_dir=embeddings_cache_dir,
                use_cache=use_cache,
            )
        )
        if use_cache and embeddings_cache_dir is not None:
            if cache_hit:
                cache_hits += 1
            else:
                cache_misses += 1
            embedding_cache_size_bytes += cache_size

        text_embedding = encode_text(model, sample.query, actual_device)
        frame_scores = compute_text_image_similarity(text_embedding, image_embeddings)
        windows = generate_temporal_windows(duration, window_size, stride)
        window_scores = aggregate_frame_scores_to_windows(
            frame_timestamps=frame_timestamps,
            frame_scores=frame_scores,
            windows=windows,
            method=aggregation,
        )
        prediction = select_best_window(windows, window_scores)

        predictions[sample.sample_id] = prediction
        ground_truth[sample.sample_id] = sample.relevant_windows
        prediction_windows.append(prediction)
        ground_truth_windows.append(sample.relevant_windows)
        total_frames += len(frame_timestamps)

    inference_time_sec = time.perf_counter() - start_time

    return build_result(
        fps=fps,
        window_size=window_size,
        stride=stride,
        aggregation=aggregation,
        model_name=model_name,
        device=actual_device,
        batch_size=batch_size,
        videos_dir=videos_dir,
        predictions=predictions,
        ground_truth=ground_truth,
        prediction_windows=prediction_windows,
        ground_truth_windows=ground_truth_windows,
        total_frames=total_frames,
        inference_time_sec=inference_time_sec,
        use_cache=use_cache,
        embeddings_cache_dir=embeddings_cache_dir,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        embedding_cache_size_bytes=embedding_cache_size_bytes,
    )


def resolve_video_path(videos_dir: str | Path, video_id: str) -> Path:
    """Resolve a sample video id to an MP4 path."""

    video_path = Path(videos_dir) / f"{video_id}.mp4"
    if not video_path.exists():
        raise ValueError(f"Missing video file for video_id '{video_id}': {video_path}")

    return video_path


def load_or_compute_video_image_embeddings(
    model,
    preprocess,
    video_path: str | Path,
    video_id: str,
    model_name: str,
    fps: float,
    device: str,
    batch_size: int,
    embeddings_cache_dir: str | Path | None = None,
    use_cache: bool = False,
) -> tuple[float, list[float], object, bool, int]:
    """Load cached image embeddings or compute and optionally cache them."""

    video_path = Path(video_path)
    cache_key = make_video_embedding_cache_key(video_id, model_name, fps)

    if use_cache and embeddings_cache_dir is not None:
        cached = load_video_embeddings_cache(embeddings_cache_dir, cache_key)
        if cached is not None:
            metadata = cached["metadata"]
            duration = (
                float(metadata["duration"])
                if "duration" in metadata
                else get_video_duration(video_path)
            )
            image_embeddings = _to_torch_embeddings(cached["image_embeddings"], device)
            cache_size = estimate_cache_file_size(cached["path"])
            return duration, cached["frame_timestamps"], image_embeddings, True, cache_size

    duration = get_video_duration(video_path)
    frame_timestamps, frames = extract_sampled_frames(video_path, fps)
    image_embeddings = encode_images(
        model,
        preprocess,
        frames,
        device,
        batch_size=batch_size,
    )

    cache_size = 0
    if use_cache and embeddings_cache_dir is not None:
        cache_path = save_video_embeddings_cache(
            embeddings_cache_dir,
            cache_key,
            frame_timestamps=frame_timestamps,
            image_embeddings=image_embeddings,
            metadata={
                "video_id": video_id,
                "model_name": model_name,
                "fps": float(fps),
                "duration": float(duration),
            },
        )
        cache_size = estimate_cache_file_size(cache_path)

    return duration, frame_timestamps, image_embeddings, False, cache_size


def _to_torch_embeddings(image_embeddings, device: str):
    if hasattr(image_embeddings, "to"):
        return image_embeddings.to(device)

    import torch

    return torch.as_tensor(image_embeddings, device=device)


def build_result(
    fps: float,
    window_size: float,
    stride: float,
    aggregation: str,
    model_name: str,
    device: str,
    batch_size: int,
    videos_dir: str | Path,
    predictions: dict[str, list[float]],
    ground_truth: dict[str, list[list[float]]],
    prediction_windows: list[list[float]],
    ground_truth_windows: list[list[list[float]]],
    total_frames: int,
    inference_time_sec: float,
    use_cache: bool = False,
    embeddings_cache_dir: str | Path | None = None,
    cache_hits: int = 0,
    cache_misses: int = 0,
    embedding_cache_size_bytes: int = 0,
) -> dict:
    """Build a serializable dataset-level retrieval result."""

    return {
        "config": {
            "videos_dir": str(videos_dir),
            "fps": float(fps),
            "window_size": float(window_size),
            "stride": float(stride),
            "aggregation": aggregation,
            "model_name": model_name,
            "device": device,
            "batch_size": batch_size,
            "use_cache": use_cache,
            "embeddings_cache_dir": (
                str(embeddings_cache_dir)
                if embeddings_cache_dir is not None
                else None
            ),
        },
        "predictions": predictions,
        "ground_truth": ground_truth,
        "metrics": {
            "R@1_IoU_0.3": recall_at_iou(
                prediction_windows,
                ground_truth_windows,
                threshold=0.3,
            ),
            "R@1_IoU_0.5": recall_at_iou(
                prediction_windows,
                ground_truth_windows,
                threshold=0.5,
            ),
            "R@1_IoU_0.7": recall_at_iou(
                prediction_windows,
                ground_truth_windows,
                threshold=0.7,
            ),
            "mIoU": mean_iou(prediction_windows, ground_truth_windows),
        },
        "stats": {
            "num_samples": len(predictions),
            "total_frames": total_frames,
            "avg_frames_per_sample": total_frames / len(predictions),
            "inference_time_sec": inference_time_sec,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "embedding_cache_size_bytes": embedding_cache_size_bytes,
        },
    }
