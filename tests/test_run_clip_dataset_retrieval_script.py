from pathlib import Path

from scripts.run_clip_dataset_retrieval import parse_args


def test_run_clip_dataset_retrieval_parse_args() -> None:
    args = parse_args(
        [
            "--annotations",
            "data/sample/synthetic_clip_annotations.jsonl",
            "--videos-dir",
            "data/sample/synthetic_clip_videos",
            "--fps",
            "1",
            "--window-size",
            "2",
            "--stride",
            "1",
            "--aggregation",
            "mean",
            "--model-name",
            "ViT-B/32",
            "--use-cache",
            "--embeddings-cache-dir",
            "results/embeddings/synthetic",
            "--output",
            "results/synthetic_clip_dataset_retrieval.json",
        ]
    )

    assert args.annotations == Path("data/sample/synthetic_clip_annotations.jsonl")
    assert args.videos_dir == Path("data/sample/synthetic_clip_videos")
    assert args.fps == 1.0
    assert args.window_size == 2.0
    assert args.stride == 1.0
    assert args.aggregation == "mean"
    assert args.model_name == "ViT-B/32"
    assert args.use_cache is True
    assert args.embeddings_cache_dir == Path("results/embeddings/synthetic")
    assert args.output == Path("results/synthetic_clip_dataset_retrieval.json")
