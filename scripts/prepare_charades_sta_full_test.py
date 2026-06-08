"""Prepare and validate the full Charades-STA test set for final evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.charades_sta_loader import find_charades_video, parse_charades_sta_line


MANIFEST_FIELDS = [
    "sample_id",
    "row_index",
    "video_id",
    "query",
    "video_path",
    "duration_sec",
    "raw_start",
    "raw_end",
    "clipped_start",
    "clipped_end",
    "was_clipped",
    "fps",
    "frame_count",
    "moment_detr_sampled_clips",
]


def read_video_metadata(video_path: Path) -> dict[str, Any]:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    try:
        can_open = bool(capture.isOpened())
        if not can_open:
            raise ValueError(f"OpenCV could not open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            raise ValueError(f"Video has invalid fps={fps}: {video_path}")
        if frame_count <= 0:
            raise ValueError(
                f"Video has invalid frame_count={frame_count}: {video_path}"
            )

        can_read_first_frame = bool(capture.read()[0])
        if not can_read_first_frame:
            raise ValueError(f"Could not read first frame from video: {video_path}")

        duration_sec = frame_count / fps
        if duration_sec <= 0:
            raise ValueError(
                f"Video has invalid duration_sec={duration_sec}: {video_path}"
            )

        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration_sec,
            "can_read_first_frame": can_read_first_frame,
        }
    finally:
        capture.release()


def build_rows(
    annotations_path: Path,
    videos_dir: Path,
    max_moment_detr_clips: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_cache: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    jsonl_rows: list[dict[str, Any]] = []

    for row_index, line in enumerate(annotations_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue

        parsed = parse_charades_sta_line(line, row_index)
        video_id = str(parsed["video_id"])
        video_path = find_charades_video(videos_dir, video_id)
        if video_path is None:
            raise ValueError(
                f"Missing local video for row {row_index}, video_id={video_id}."
            )

        if video_id not in metadata_cache:
            metadata_cache[video_id] = read_video_metadata(video_path)
        metadata = metadata_cache[video_id]
        duration = float(metadata["duration_sec"])

        raw_start = float(parsed["start_time"])
        raw_end = float(parsed["end_time"])
        clipped_start = min(max(raw_start, 0.0), duration)
        clipped_end = min(max(raw_end, 0.0), duration)
        if not 0.0 <= clipped_start < clipped_end <= duration:
            raise ValueError(
                "Invalid GT window after clipping at row "
                f"{row_index}: start={clipped_start}, end={clipped_end}, "
                f"duration={duration}."
            )

        sampled_clips = int(math.ceil(duration / 2.0))
        if sampled_clips > max_moment_detr_clips:
            raise ValueError(
                "Moment-DETR clip limit exceeded for row "
                f"{row_index}, video_id={video_id}: {sampled_clips} clips "
                f"> {max_moment_detr_clips}."
            )

        sample_id = f"charades_sta_test_{row_index}"
        was_clipped = clipped_start != raw_start or clipped_end != raw_end
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "row_index": row_index,
                "video_id": video_id,
                "query": str(parsed["query"]),
                "video_path": str(video_path),
                "duration_sec": duration,
                "raw_start": raw_start,
                "raw_end": raw_end,
                "clipped_start": clipped_start,
                "clipped_end": clipped_end,
                "was_clipped": was_clipped,
                "fps": metadata["fps"],
                "frame_count": metadata["frame_count"],
                "moment_detr_sampled_clips": sampled_clips,
            }
        )
        jsonl_rows.append(
            {
                "qid": sample_id,
                "query": str(parsed["query"]),
                "duration": duration,
                "vid": video_id,
                "relevant_windows": [[clipped_start, clipped_end]],
            }
        )

    return manifest_rows, jsonl_rows


def validate_rows(
    manifest_rows: list[dict[str, Any]],
    jsonl_rows: list[dict[str, Any]],
    expected_queries: int,
    expected_unique_videos: int,
    max_moment_detr_clips: int,
) -> None:
    if len(manifest_rows) != expected_queries:
        raise ValueError(
            f"Expected {expected_queries} manifest rows, found {len(manifest_rows)}."
        )
    if len(jsonl_rows) != expected_queries:
        raise ValueError(
            f"Expected {expected_queries} JSONL rows, found {len(jsonl_rows)}."
        )

    sample_ids = [str(row["sample_id"]) for row in manifest_rows]
    qids = [str(row["qid"]) for row in jsonl_rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Manifest sample_id values are not unique.")
    if len(set(qids)) != len(qids):
        raise ValueError("JSONL qid values are not unique.")
    if sample_ids != qids:
        raise ValueError("Manifest sample_id order does not match JSONL qid order.")

    unique_videos = {str(row["video_id"]) for row in manifest_rows}
    if len(unique_videos) != expected_unique_videos:
        raise ValueError(
            f"Expected {expected_unique_videos} unique videos, found "
            f"{len(unique_videos)}."
        )

    for index, (manifest_row, jsonl_row) in enumerate(zip(manifest_rows, jsonl_rows)):
        duration = float(manifest_row["duration_sec"])
        start = float(manifest_row["clipped_start"])
        end = float(manifest_row["clipped_end"])
        if not 0.0 <= start < end <= duration:
            raise ValueError(
                f"Invalid manifest GT window at row {index}: {start}, {end}, "
                f"duration={duration}."
            )
        if int(manifest_row["moment_detr_sampled_clips"]) > max_moment_detr_clips:
            raise ValueError(f"Moment-DETR clip limit exceeded at row {index}.")
        if str(manifest_row["video_id"]) != str(jsonl_row["vid"]):
            raise ValueError(f"Manifest/JSONL video_id mismatch at row {index}.")
        if str(manifest_row["query"]) != str(jsonl_row["query"]):
            raise ValueError(f"Manifest/JSONL query mismatch at row {index}.")
        json_window = jsonl_row["relevant_windows"][0]
        if [float(json_window[0]), float(json_window[1])] != [start, end]:
            raise ValueError(f"Manifest/JSONL window mismatch at row {index}.")


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}.") from error
    return rows


def write_sha256(path: Path, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for file_path in files:
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {file_path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and validate full Charades-STA test artifacts."
    )
    parser.add_argument(
        "--annotations-path",
        type=Path,
        default=Path("data/charades_sta/charades_sta_test.txt"),
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("data/raw/charades/videos"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/processed/charades_sta_full_test_manifest.csv"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/processed/charades_sta_full_test_moment_detr.jsonl"),
    )
    parser.add_argument(
        "--output-sha256",
        type=Path,
        default=Path("data/processed/charades_sta_full_test_manifest.sha256"),
    )
    parser.add_argument("--expected-queries", type=int, default=3720)
    parser.add_argument("--expected-unique-videos", type=int, default=1334)
    parser.add_argument("--max-moment-detr-clips", type=int, default=75)
    parser.add_argument("--validate-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.validate_existing:
        if not args.output_manifest.exists():
            raise FileNotFoundError(f"Manifest does not exist: {args.output_manifest}")
        if not args.output_jsonl.exists():
            raise FileNotFoundError(f"JSONL does not exist: {args.output_jsonl}")
        manifest_rows = read_manifest(args.output_manifest)
        jsonl_rows = read_jsonl(args.output_jsonl)
        expected_manifest_rows, expected_jsonl_rows = build_rows(
            annotations_path=args.annotations_path,
            videos_dir=args.videos_dir,
            max_moment_detr_clips=args.max_moment_detr_clips,
        )
        expected_sample_ids = [row["sample_id"] for row in expected_manifest_rows]
        expected_qids = [row["qid"] for row in expected_jsonl_rows]
        if [row["sample_id"] for row in manifest_rows] != expected_sample_ids:
            raise ValueError("Existing manifest sample_id order does not match annotations.")
        if [row["qid"] for row in jsonl_rows] != expected_qids:
            raise ValueError("Existing JSONL qid order does not match annotations.")
    else:
        manifest_rows, jsonl_rows = build_rows(
            annotations_path=args.annotations_path,
            videos_dir=args.videos_dir,
            max_moment_detr_clips=args.max_moment_detr_clips,
        )
        validate_rows(
            manifest_rows,
            jsonl_rows,
            expected_queries=args.expected_queries,
            expected_unique_videos=args.expected_unique_videos,
            max_moment_detr_clips=args.max_moment_detr_clips,
        )
        write_manifest(args.output_manifest, manifest_rows)
        write_jsonl(args.output_jsonl, jsonl_rows)
        write_sha256(args.output_sha256, [args.output_manifest, args.output_jsonl])

    validate_rows(
        manifest_rows,
        jsonl_rows,
        expected_queries=args.expected_queries,
        expected_unique_videos=args.expected_unique_videos,
        max_moment_detr_clips=args.max_moment_detr_clips,
    )

    print(f"Validated queries: {len(manifest_rows)}")
    print(f"Validated unique videos: {len({row['video_id'] for row in manifest_rows})}")
    print(f"Manifest: {args.output_manifest}")
    print(f"Moment-DETR JSONL: {args.output_jsonl}")
    print(f"SHA256 manifest: {args.output_sha256}")
    return 0


def _import_cv2():
    try:
        import cv2
    except ImportError as error:
        raise ImportError("opencv-python is required for full-test validation.") from error
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
