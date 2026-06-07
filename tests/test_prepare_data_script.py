import csv
import zipfile
from pathlib import Path

from scripts.prepare_data import (
    extract_charades_videos_from_zip,
    list_video_members,
    load_video_ids,
    normalize_video_id_from_member,
    parse_args,
    write_manifest,
)


def test_normalize_video_id_from_zip_member_path() -> None:
    assert normalize_video_id_from_member("Charades_v1/ABC123.mp4") == "ABC123"
    assert normalize_video_id_from_member("nested/Charades_v1/DEF456.MP4") == "DEF456"
    assert normalize_video_id_from_member("Charades_v1/readme.txt") is None


def test_load_video_ids_reads_txt_and_csv_like_rows(tmp_path) -> None:
    ids_path = tmp_path / "video_ids.csv"
    ids_path.write_text("video_id,query\nABC123,a query\nDEF456\n", encoding="utf-8")

    assert load_video_ids(ids_path) == {"ABC123", "DEF456"}


def test_list_video_members_filters_by_video_ids_and_limit(tmp_path) -> None:
    zip_path = _make_zip(
        tmp_path,
        {
            "Charades_v1/ABC123.mp4": b"abc",
            "Charades_v1/DEF456.mp4": b"def",
            "Charades_v1/GHI789.mp4": b"ghi",
        },
    )

    with zipfile.ZipFile(zip_path) as zip_file:
        members = list_video_members(zip_file, video_ids={"DEF456", "GHI789"}, limit=1)

    assert members == [("DEF456", "Charades_v1/DEF456.mp4")]


def test_extract_charades_videos_from_small_temporary_zip(tmp_path) -> None:
    zip_path = _make_zip(
        tmp_path,
        {
            "Charades_v1/ABC123.mp4": b"abc",
            "Charades_v1/DEF456.mp4": b"def",
            "Charades_v1/readme.txt": b"ignored",
        },
    )
    output_dir = tmp_path / "videos"

    rows = extract_charades_videos_from_zip(
        zip_path=zip_path,
        output_dir=output_dir,
        video_ids={"DEF456"},
    )

    assert rows == [
        {
            "video_id": "DEF456",
            "source_member": "Charades_v1/DEF456.mp4",
            "output_path": str(output_dir / "DEF456.mp4"),
            "extracted": True,
            "error": "",
        }
    ]
    assert (output_dir / "DEF456.mp4").read_bytes() == b"def"
    assert not (output_dir / "ABC123.mp4").exists()


def test_write_manifest_uses_expected_columns(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(
        [
            {
                "video_id": "ABC123",
                "source_member": "Charades_v1/ABC123.mp4",
                "output_path": "data/raw/charades/videos/ABC123.mp4",
                "extracted": True,
                "error": "",
            }
        ],
        manifest_path,
    )

    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["video_id"] == "ABC123"
    assert rows[0]["extracted"] == "True"


def test_parse_args() -> None:
    args = parse_args(
        [
            "--zip_path",
            "/tmp/Charades_v1.zip",
            "--output_dir",
            "data/raw/charades/videos",
            "--video_ids",
            "ids.txt",
            "--limit",
            "20",
        ]
    )

    assert args.zip_path == Path("/tmp/Charades_v1.zip")
    assert args.output_dir == Path("data/raw/charades/videos")
    assert args.video_ids == Path("ids.txt")
    assert args.limit == 20


def _make_zip(tmp_path: Path, members: dict[str, bytes]) -> Path:
    zip_path = tmp_path / "Charades_v1.zip"
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        for member, content in members.items():
            zip_file.writestr(member, content)
    return zip_path
