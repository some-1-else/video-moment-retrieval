"""Extract a small set of Charades videos from a local zip archive."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence
import zipfile


DEFAULT_OUTPUT_DIR = Path("data/raw/charades/videos")
DEFAULT_MANIFEST = Path("data/processed/charades_zip_extract_manifest.csv")
MANIFEST_COLUMNS = ["video_id", "source_member", "output_path", "extracted", "error"]


def normalize_video_id_from_member(member_path: str) -> str | None:
    """Return a Charades video id from an mp4 zip member path."""

    path = Path(member_path)
    if path.suffix.lower() != ".mp4":
        return None

    video_id = path.stem.strip()
    return video_id or None


def load_video_ids(path: str | Path) -> set[str]:
    """Read video ids from a txt/csv file.

    The first comma-separated field is used for CSV-like rows. A header row named
    video_id, id, or vid is skipped.
    """

    video_ids: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue

        first_field = stripped_line.split(",", maxsplit=1)[0].strip()
        if first_field.lower() in {"video_id", "id", "vid"}:
            continue
        if first_field:
            video_ids.add(first_field)

    return video_ids


def list_video_members(
    zip_file: zipfile.ZipFile,
    video_ids: set[str] | None = None,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    """List selected mp4 members as (video_id, source_member)."""

    selected: list[tuple[str, str]] = []
    for member in zip_file.namelist():
        video_id = normalize_video_id_from_member(member)
        if video_id is None:
            continue
        if video_ids is not None and video_id not in video_ids:
            continue

        selected.append((video_id, member))
        if limit is not None and len(selected) >= limit:
            break

    return selected


def extract_charades_videos_from_zip(
    zip_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    video_ids: set[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Extract selected videos from a local Charades zip archive."""

    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as zip_file:
        for video_id, source_member in list_video_members(zip_file, video_ids, limit):
            output_path = output_dir / f"{video_id}.mp4"
            row = {
                "video_id": video_id,
                "source_member": source_member,
                "output_path": str(output_path),
                "extracted": False,
                "error": "",
            }

            try:
                with zip_file.open(source_member) as source_file:
                    output_path.write_bytes(source_file.read())
                row["extracted"] = True
            except Exception as exc:
                row["error"] = str(exc)

            rows.append(row)

    return rows


def write_manifest(rows: list[dict[str, object]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected Charades videos from a local zip archive."
    )
    parser.add_argument("--zip_path", required=True, type=Path)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--video_ids", default=None, type=Path)
    parser.add_argument("--limit", default=None, type=int)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer when provided.")

    video_ids = load_video_ids(args.video_ids) if args.video_ids else None
    rows = extract_charades_videos_from_zip(
        zip_path=args.zip_path,
        output_dir=args.output_dir,
        video_ids=video_ids,
        limit=args.limit,
    )
    write_manifest(rows, DEFAULT_MANIFEST)

    extracted = sum(1 for row in rows if row["extracted"] is True)
    failed = len(rows) - extracted
    print(f"Selected videos: {len(rows)}")
    print(f"Extracted videos: {extracted}")
    print(f"Failed videos: {failed}")
    print(f"Saved manifest to: {DEFAULT_MANIFEST}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
