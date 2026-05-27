"""Validate locally available QVHighlights video files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def validate_video_file(video_path: str | Path) -> dict:
    path = Path(video_path)
    result = {
        "exists": path.exists(),
        "readable": False,
        "duration": None,
        "frame_count": None,
        "error": None,
    }

    if not path.exists():
        result["error"] = f"video file does not exist: {path}"
        return result

    if not path.is_file():
        result["error"] = f"video path is not a file: {path}"
        return result

    try:
        cv2 = _import_cv2()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            result["error"] = f"OpenCV could not open video file: {path}"
            return result

        try:
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv2.CAP_PROP_FPS)
            result["frame_count"] = frame_count
            if frame_count <= 0:
                result["error"] = "video frame_count is not positive"
                return result
            if fps <= 0:
                result["error"] = "video fps is not positive"
                return result

            duration = frame_count / fps
            result["duration"] = duration
            if duration <= 0:
                result["error"] = "video duration is not positive"
                return result

            success, _frame = capture.read()
            if not success:
                result["error"] = "OpenCV could not read the first frame"
                return result
        finally:
            capture.release()

    except Exception as exc:
        result["error"] = str(exc)
        return result

    result["readable"] = True
    return result


def build_validation_report(
    manifest: dict,
    manifest_path: str | Path,
) -> dict:
    videos = []
    for sample in manifest.get("samples", []):
        if sample.get("video_exists") is not True:
            continue

        validation = validate_video_file(sample.get("video_path", ""))
        videos.append(
            {
                "video_id": sample.get("video_id", ""),
                "video_path": sample.get("video_path", ""),
                **validation,
            }
        )

    num_readable = sum(1 for video in videos if video["readable"])
    num_unreadable = len(videos) - num_readable

    return {
        "manifest": str(manifest_path),
        "num_checked": len(videos),
        "num_readable": num_readable,
        "num_unreadable": num_unreadable,
        "videos": videos,
    }


def load_manifest(manifest_path: str | Path) -> dict:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def write_validation_report(report: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_validation_summary(report: dict) -> None:
    print(f"Checked videos: {report['num_checked']}")
    print(f"Readable videos: {report['num_readable']}")
    print(f"Unreadable videos: {report['num_unreadable']}")

    unreadable_videos = [
        video for video in report["videos"] if video.get("readable") is not True
    ]
    if unreadable_videos:
        print("Unreadable videos:")
        for video in unreadable_videos:
            print(f"- {video['video_id']}: {video.get('error')}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate locally available QVHighlights video files."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    report = build_validation_report(manifest, args.manifest)
    write_validation_report(report, args.output)
    print_validation_summary(report)
    print(f"Saved validation report to: {args.output}")

    return 0


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "opencv-python is required for video validation. "
            "Install project requirements before validating videos."
        ) from error

    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
