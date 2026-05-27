"""Run CLIP retrieval on one local video and one text query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.clip_encoder import (
    compute_text_image_similarity,
    encode_images,
    encode_text,
    load_clip_model,
)
from src.retrieval.aggregation import aggregate_frame_scores_to_windows
from src.retrieval.scoring import select_best_window
from src.retrieval.windowing import generate_temporal_windows
from src.video.frame_extraction import extract_sampled_frames, get_video_duration


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-video CLIP retrieval on a local video file."
    )
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Path to a local video file.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Text query.",
    )
    parser.add_argument(
        "--fps",
        required=True,
        type=float,
        help="Frame sampling rate in frames per second.",
    )
    parser.add_argument(
        "--window-size",
        required=True,
        type=float,
        help="Temporal window size in seconds.",
    )
    parser.add_argument(
        "--stride",
        required=True,
        type=float,
        help="Temporal window stride in seconds.",
    )
    parser.add_argument(
        "--aggregation",
        choices=("mean", "max"),
        default="mean",
        help="How to aggregate frame scores inside temporal windows.",
    )
    parser.add_argument(
        "--model-name",
        default="ViT-B/32",
        help="CLIP model name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path where the JSON result should be saved.",
    )

    return parser.parse_args(argv)


def run_from_args(args: argparse.Namespace) -> dict:
    duration = get_video_duration(args.video)
    frame_timestamps, frames = extract_sampled_frames(args.video, args.fps)
    model, preprocess, device = load_clip_model(args.model_name, device="auto")
    text_embedding = encode_text(model, args.query, device)
    image_embeddings = encode_images(model, preprocess, frames, device)
    frame_scores = compute_text_image_similarity(text_embedding, image_embeddings)
    windows = generate_temporal_windows(duration, args.window_size, args.stride)
    window_scores = aggregate_frame_scores_to_windows(
        frame_timestamps=frame_timestamps,
        frame_scores=frame_scores,
        windows=windows,
        method=args.aggregation,
    )
    prediction = select_best_window(windows, window_scores)

    result = build_result(
        video_path=args.video,
        query=args.query,
        fps=args.fps,
        window_size=args.window_size,
        stride=args.stride,
        aggregation=args.aggregation,
        model_name=args.model_name,
        device=device,
        duration=duration,
        frame_timestamps=frame_timestamps,
        frame_scores=frame_scores,
        windows=windows,
        window_scores=window_scores,
        prediction=prediction,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return result


def build_result(
    video_path: str | Path,
    query: str,
    fps: float,
    window_size: float,
    stride: float,
    aggregation: str,
    model_name: str,
    device: str,
    duration: float,
    frame_timestamps: list[float],
    frame_scores: list[float],
    windows: list[list[float]],
    window_scores: list[float],
    prediction: list[float],
) -> dict:
    """Build the serializable one-video retrieval result."""

    return {
        "config": {
            "video": str(video_path),
            "query": query,
            "fps": float(fps),
            "window_size": float(window_size),
            "stride": float(stride),
            "aggregation": aggregation,
            "model_name": model_name,
            "device": device,
        },
        "video": {
            "duration": float(duration),
            "num_frames": len(frame_timestamps),
        },
        "prediction": prediction,
        "frame_timestamps": frame_timestamps,
        "frame_scores": frame_scores,
        "windows": windows,
        "window_scores": window_scores,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_from_args(args)
    best_score = max(result["window_scores"]) if result["window_scores"] else None

    print(f"Device: {result['config']['device']}")
    print(f"Duration: {result['video']['duration']:.4f}")
    print(f"Number of frames: {result['video']['num_frames']}")
    print(f"Predicted window: {result['prediction']}")
    print(f"Best score: {best_score}")

    if args.output is not None:
        print(f"Saved result to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
