"""Run CLIP retrieval on a small local annotated video dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.qvhighlights import load_qvhighlights_annotations
from src.pipeline.clip_dataset_retrieval import run_clip_dataset_retrieval


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CLIP retrieval on a small local annotated video dataset."
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--window-size", required=True, type=float)
    parser.add_argument("--stride", required=True, type=float)
    parser.add_argument("--aggregation", choices=("mean", "max"), default="mean")
    parser.add_argument("--model-name", default="ViT-B/32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embeddings-cache-dir", type=Path, default=None)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=None)

    return parser.parse_args(argv)


def run_from_args(args: argparse.Namespace) -> dict:
    samples = load_qvhighlights_annotations(args.annotations)
    result = run_clip_dataset_retrieval(
        samples=samples,
        videos_dir=args.videos_dir,
        fps=args.fps,
        window_size=args.window_size,
        stride=args.stride,
        aggregation=args.aggregation,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        embeddings_cache_dir=args.embeddings_cache_dir,
        use_cache=args.use_cache,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_from_args(args)

    print("Metrics:")
    for metric_name, value in result["metrics"].items():
        print(f"{metric_name}: {value:.4f}")

    print("Stats:")
    for stat_name, value in result["stats"].items():
        if isinstance(value, float):
            print(f"{stat_name}: {value:.4f}")
        else:
            print(f"{stat_name}: {value}")

    if args.output is not None:
        print(f"Saved result to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
