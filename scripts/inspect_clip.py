"""Inspect pretrained CLIP text encoding without running retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.clip_encoder import encode_text, load_clip_model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect CLIP text encoding.")
    parser.add_argument(
        "--model-name",
        default="ViT-B/32",
        help="CLIP model name.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Text query to encode.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model, _, device = load_clip_model(args.model_name, device="auto")
    text_embedding = encode_text(model, args.text, device)

    print(f"Device: {device}")
    print(f"Text embedding shape: {tuple(text_embedding.shape)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
