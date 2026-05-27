import json

from scripts.create_qvhighlights_subset_manifest import (
    create_subset_manifest,
    parse_args as parse_create_args,
    select_subset_samples,
    write_manifest,
)
from scripts.inspect_qvhighlights_subset_manifest import (
    load_manifest,
    summarize_manifest,
)
from src.data.qvhighlights import MomentRetrievalSample


def test_create_qvhighlights_subset_manifest_tracks_available_and_missing_videos(
    tmp_path,
) -> None:
    annotations_path = tmp_path / "annotations.jsonl"
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    output_path = tmp_path / "manifest.json"

    records = [
        {
            "qid": "sample_1",
            "vid": "video_available_10.0_20.0",
            "query": "person opens a door",
            "duration": 10.0,
            "relevant_windows": [[2.0, 6.0]],
        },
        {
            "qid": "sample_2",
            "vid": "video_missing_20.0_30.0",
            "query": "person sits down",
            "duration": 8.0,
            "relevant_windows": [[1.0, 3.0]],
        },
    ]
    annotations_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    (videos_dir / "video_available_10.0_20.0.mp4").write_bytes(b"fake mp4")

    manifest = create_subset_manifest(
        annotations_path=annotations_path,
        videos_dir=videos_dir,
        limit=2,
    )
    write_manifest(manifest, output_path)

    assert manifest["annotations"] == str(annotations_path)
    assert manifest["videos_dir"] == str(videos_dir)
    assert manifest["limit"] == 2
    assert manifest["strategy"] == "first"
    assert manifest["num_samples"] == 2
    assert manifest["num_available_videos"] == 1
    assert manifest["num_missing_videos"] == 1
    assert manifest["samples"][0] == {
        "sample_id": "sample_1",
        "video_id": "video_available_10.0_20.0",
        "youtube_id": "video_available",
        "clip_start": 10.0,
        "clip_end": 20.0,
        "query": "person opens a door",
        "duration": 10.0,
        "relevant_windows": [[2.0, 6.0]],
        "video_path": str(videos_dir / "video_available_10.0_20.0.mp4"),
        "video_exists": True,
    }
    assert manifest["samples"][1]["video_path"] == str(
        videos_dir / "video_missing_20.0_30.0.mp4"
    )
    assert manifest["samples"][1]["youtube_id"] == "video_missing"
    assert manifest["samples"][1]["clip_start"] == 20.0
    assert manifest["samples"][1]["clip_end"] == 30.0
    assert manifest["samples"][1]["video_exists"] is False

    loaded_manifest = load_manifest(output_path)
    assert loaded_manifest == manifest


def test_qvhighlights_subset_manifest_respects_limit(tmp_path) -> None:
    annotations_path = tmp_path / "annotations.jsonl"
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    records = [
        {
            "sample_id": f"sample_{index}",
            "video_id": f"video_{index}",
            "query": "query",
            "duration": 5.0,
            "relevant_windows": [[1.0, 2.0]],
        }
        for index in range(3)
    ]
    annotations_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    manifest = create_subset_manifest(
        annotations_path=annotations_path,
        videos_dir=videos_dir,
        limit=2,
    )

    assert manifest["num_samples"] == 2
    assert [sample["sample_id"] for sample in manifest["samples"]] == [
        "sample_0",
        "sample_1",
    ]


def test_select_subset_samples_first_strategy_keeps_order() -> None:
    samples = [
        MomentRetrievalSample(
            sample_id=f"sample_{index}",
            video_id=f"youtube_{index}_0.0_10.0",
            query="query",
            duration=10.0,
            relevant_windows=[[1.0, 2.0]],
        )
        for index in range(3)
    ]

    selected = select_subset_samples(samples, limit=2, strategy="first")

    assert [sample.sample_id for sample in selected] == ["sample_0", "sample_1"]


def test_select_subset_samples_diverse_youtube_takes_first_per_youtube_id() -> None:
    samples = [
        MomentRetrievalSample(
            sample_id="sample_1",
            video_id="youtube_a_0.0_10.0",
            query="query",
            duration=10.0,
            relevant_windows=[[1.0, 2.0]],
        ),
        MomentRetrievalSample(
            sample_id="sample_2",
            video_id="youtube_a_10.0_20.0",
            query="query",
            duration=10.0,
            relevant_windows=[[1.0, 2.0]],
        ),
        MomentRetrievalSample(
            sample_id="sample_3",
            video_id="youtube_b_0.0_10.0",
            query="query",
            duration=10.0,
            relevant_windows=[[1.0, 2.0]],
        ),
    ]

    selected = select_subset_samples(samples, limit=3, strategy="diverse-youtube")

    assert [sample.sample_id for sample in selected] == ["sample_1", "sample_3"]


def test_create_qvhighlights_subset_manifest_diverse_youtube_strategy(tmp_path) -> None:
    annotations_path = tmp_path / "annotations.jsonl"
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    records = [
        {
            "sample_id": "sample_1",
            "video_id": "youtube_a_0.0_10.0",
            "query": "query 1",
            "duration": 10.0,
            "relevant_windows": [[1.0, 2.0]],
        },
        {
            "sample_id": "sample_2",
            "video_id": "youtube_a_10.0_20.0",
            "query": "query 2",
            "duration": 10.0,
            "relevant_windows": [[2.0, 3.0]],
        },
        {
            "sample_id": "sample_3",
            "video_id": "youtube_b_0.0_10.0",
            "query": "query 3",
            "duration": 10.0,
            "relevant_windows": [[3.0, 4.0]],
        },
    ]
    annotations_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    (videos_dir / "youtube_b_0.0_10.0.mp4").write_bytes(b"fake mp4")

    manifest = create_subset_manifest(
        annotations_path=annotations_path,
        videos_dir=videos_dir,
        limit=3,
        strategy="diverse-youtube",
    )

    assert manifest["strategy"] == "diverse-youtube"
    assert manifest["num_samples"] == 2
    assert manifest["num_available_videos"] == 1
    assert manifest["num_missing_videos"] == 1
    assert [sample["sample_id"] for sample in manifest["samples"]] == [
        "sample_1",
        "sample_3",
    ]
    assert [sample["youtube_id"] for sample in manifest["samples"]] == [
        "youtube_a",
        "youtube_b",
    ]


def test_inspect_qvhighlights_subset_manifest_summary() -> None:
    manifest = {
        "strategy": "diverse-youtube",
        "num_samples": 2,
        "num_available_videos": 1,
        "num_missing_videos": 1,
        "samples": [
            {
                "sample_id": "sample_1",
                "video_id": "video_available_10.0_20.0",
                "youtube_id": "video_available",
                "clip_start": 10.0,
                "clip_end": 20.0,
                "query": "red square",
                "relevant_windows": [[1.0, 3.0]],
                "video_exists": True,
            },
            {
                "sample_id": "sample_2",
                "video_id": "video_missing_20.0_30.0",
                "youtube_id": "video_missing",
                "clip_start": 20.0,
                "clip_end": 30.0,
                "query": "blue square",
                "relevant_windows": [[2.0, 4.0]],
                "video_exists": False,
            },
        ],
    }

    summary = summarize_manifest(manifest, preview_limit=5)

    assert summary == {
        "strategy": "diverse-youtube",
        "num_samples": 2,
        "num_available_videos": 1,
        "num_missing_videos": 1,
        "num_parsed_clip_ranges": 2,
        "num_unique_youtube_ids": 2,
        "unique_youtube_ids": ["video_available", "video_missing"],
        "missing_video_ids": ["video_missing_20.0_30.0"],
        "available_video_ids": ["video_available_10.0_20.0"],
        "youtube_ids": ["video_available", "video_missing"],
        "clip_ranges": [[10.0, 20.0], [20.0, 30.0]],
        "examples": [
            {
                "sample_id": "sample_1",
                "video_id": "video_available_10.0_20.0",
                "youtube_id": "video_available",
                "clip_start": 10.0,
                "clip_end": 20.0,
                "query": "red square",
                "relevant_windows": [[1.0, 3.0]],
            },
            {
                "sample_id": "sample_2",
                "video_id": "video_missing_20.0_30.0",
                "youtube_id": "video_missing",
                "clip_start": 20.0,
                "clip_end": 30.0,
                "query": "blue square",
                "relevant_windows": [[2.0, 4.0]],
            },
        ],
    }


def test_create_qvhighlights_subset_manifest_parse_args() -> None:
    args = parse_create_args(
        [
            "--annotations",
            "data/qvhighlights/highlight_val_release.jsonl",
            "--videos-dir",
            "data/qvhighlights/videos",
            "--limit",
            "20",
            "--strategy",
            "diverse-youtube",
            "--output",
            "data/qvhighlights/qvhighlights_val_subset_manifest.json",
        ]
    )

    assert str(args.annotations) == "data/qvhighlights/highlight_val_release.jsonl"
    assert str(args.videos_dir) == "data/qvhighlights/videos"
    assert args.limit == 20
    assert args.strategy == "diverse-youtube"
    assert str(args.output) == "data/qvhighlights/qvhighlights_val_subset_manifest.json"
