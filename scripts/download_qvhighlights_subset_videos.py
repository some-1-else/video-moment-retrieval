"""Download a tiny local QVHighlights video subset from a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Sequence


def load_manifest(manifest_path: str | Path) -> dict:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def get_missing_samples(manifest: dict) -> list[dict]:
    return [
        sample
        for sample in manifest.get("samples", [])
        if sample.get("video_exists") is not True
    ]


def build_download_plan(
    manifest: dict,
    output_dir: str | Path,
    max_downloads: int,
) -> list[dict]:
    if max_downloads <= 0:
        raise ValueError("max_downloads must be a positive integer")

    output_dir = Path(output_dir)
    planned_samples = get_missing_samples(manifest)[:max_downloads]
    plan = []

    for sample in planned_samples:
        video_id = sample["video_id"]
        youtube_id = sample.get("youtube_id") or video_id
        output_path = output_dir / f"{video_id}.mp4"
        plan.append(
            {
                "sample_id": sample.get("sample_id", ""),
                "video_id": video_id,
                "youtube_id": youtube_id,
                "clip_start": sample.get("clip_start"),
                "clip_end": sample.get("clip_end"),
                "youtube_url": f"https://www.youtube.com/watch?v={youtube_id}",
                "output_path": str(output_path),
            }
        )

    return plan


def _require_yt_dlp() -> object:
    try:
        import yt_dlp
    except ImportError as exc:
        raise ImportError(
            "yt-dlp is required for video download. Install project requirements with "
            "`.venv/bin/python -m pip install -r requirements.txt`."
        ) from exc

    return yt_dlp


def _require_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg was not found. Install ffmpeg before downloading and trimming "
            "QVHighlights video clips."
        )

    return ffmpeg_path


def _download_source_video(youtube_url: str, temp_dir: Path) -> Path:
    yt_dlp = _require_yt_dlp()
    output_template = str(temp_dir / "source.%(ext)s")
    options = {
        "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "outtmpl": output_template,
        "quiet": True,
        "noplaylist": True,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        downloaded_path = Path(ydl.prepare_filename(info))

    merged_path = downloaded_path.with_suffix(".mp4")
    if merged_path.exists():
        return merged_path
    if downloaded_path.exists():
        return downloaded_path

    candidates = list(temp_dir.glob("source.*"))
    if candidates:
        return candidates[0]

    raise RuntimeError(f"yt-dlp did not create a source file for {youtube_url}")


def _trim_video_clip(
    source_path: Path,
    output_path: Path,
    clip_start: float,
    clip_end: float,
) -> None:
    if clip_start is None or clip_end is None:
        raise ValueError("clip_start and clip_end are required to trim a video clip")
    if clip_end <= clip_start:
        raise ValueError("clip_end must be greater than clip_start")

    ffmpeg_path = _require_ffmpeg()
    duration = clip_end - clip_start
    command = [
        ffmpeg_path,
        "-y",
        "-ss",
        str(clip_start),
        "-i",
        str(source_path),
        "-t",
        str(duration),
        "-c",
        "copy",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to trim clip: {completed.stderr.strip()}")


def download_planned_sample(plan_entry: dict) -> str:
    output_path = Path(plan_entry["output_path"])
    if output_path.exists():
        return "skipped_existing"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        source_path = _download_source_video(plan_entry["youtube_url"], temp_dir)
        _trim_video_clip(
            source_path=source_path,
            output_path=output_path,
            clip_start=plan_entry["clip_start"],
            clip_end=plan_entry["clip_end"],
        )

    return "downloaded"


def download_subset_videos(
    manifest: dict,
    output_dir: str | Path,
    max_downloads: int,
    dry_run: bool = False,
) -> dict:
    plan = build_download_plan(manifest, output_dir, max_downloads)
    summary = {
        "attempted": 0,
        "downloaded": 0,
        "skipped_existing": 0,
        "failed": 0,
        "plan": plan,
        "failures": [],
    }

    for entry in plan:
        print(
            f"Plan: {entry['video_id']} | {entry['youtube_id']} | "
            f"[{entry['clip_start']}, {entry['clip_end']}] -> {entry['output_path']}"
        )

        if dry_run:
            continue

        if Path(entry["output_path"]).exists():
            summary["skipped_existing"] += 1
            print(f"Skipped existing: {entry['output_path']}")
            continue

        summary["attempted"] += 1
        try:
            status = download_planned_sample(entry)
        except Exception as exc:  # pragma: no cover - real network/ffmpeg path
            summary["failed"] += 1
            summary["failures"].append(
                {
                    "video_id": entry["video_id"],
                    "error": str(exc),
                }
            )
            print(f"Failed: {entry['video_id']} | {exc}")
            continue

        summary[status] += 1
        print(f"{status}: {entry['video_id']}")

    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a tiny QVHighlights video subset from a manifest."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-downloads", required=True, type=int)
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    summary = download_subset_videos(
        manifest=manifest,
        output_dir=args.output_dir,
        max_downloads=args.max_downloads,
        dry_run=args.dry_run,
    )

    print("Summary:")
    print(f"attempted: {summary['attempted']}")
    print(f"downloaded: {summary['downloaded']}")
    print(f"skipped_existing: {summary['skipped_existing']}")
    print(f"failed: {summary['failed']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
