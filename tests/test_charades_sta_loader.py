from src.data.charades_sta_loader import (
    load_charades_sta_annotations,
    parse_charades_sta_line,
)


def test_parse_charades_sta_line_parses_standard_format() -> None:
    parsed = parse_charades_sta_line("ABC123 12.3 18.7##person opens a door")

    assert parsed == {
        "video_id": "ABC123",
        "start_time": 12.3,
        "end_time": 18.7,
        "query": "person opens a door",
    }


def test_parse_charades_sta_line_preserves_query_with_spaces() -> None:
    parsed = parse_charades_sta_line(
        "ABC123 1.0 2.5##person slowly opens the front door"
    )

    assert parsed["query"] == "person slowly opens the front door"


def test_load_charades_sta_annotations_returns_expected_fields(tmp_path) -> None:
    annotations_path = tmp_path / "charades_sta_train.txt"
    annotations_path.write_text(
        "\n".join(
            [
                "ABC123 12.3 18.7##person opens a door",
                "DEF456 0.0 5.0##person sits down",
            ]
        ),
        encoding="utf-8",
    )
    metadata_path = tmp_path / "Charades_v1_train.csv"
    metadata_path.write_text(
        "id,duration\nABC123,30.0\nDEF456,28.5\n",
        encoding="utf-8",
    )

    samples = load_charades_sta_annotations(
        annotations_path,
        split="train",
        metadata_csv_path=metadata_path,
    )

    assert samples[0] == {
        "qid": "charades_sta_train_0",
        "vid": "ABC123",
        "query": "person opens a door",
        "duration": 30.0,
        "relevant_windows": [[12.3, 18.7]],
        "source_dataset": "charades_sta",
    }
    assert set(samples[1]) == {
        "qid",
        "vid",
        "query",
        "duration",
        "relevant_windows",
        "source_dataset",
    }


def test_load_charades_sta_annotations_relevant_windows_are_list_of_float_lists(
    tmp_path,
) -> None:
    annotations_path = tmp_path / "charades_sta_test.txt"
    annotations_path.write_text("ABC123 12.3 18.7##person opens a door", encoding="utf-8")

    samples = load_charades_sta_annotations(annotations_path, split="test")

    relevant_windows = samples[0]["relevant_windows"]
    assert isinstance(relevant_windows, list)
    assert isinstance(relevant_windows[0], list)
    assert all(isinstance(value, float) for value in relevant_windows[0])


def test_load_charades_sta_annotations_qids_are_unique(tmp_path) -> None:
    annotations_path = tmp_path / "charades_sta_train.txt"
    annotations_path.write_text(
        "\n".join(
            [
                "ABC123 12.3 18.7##person opens a door",
                "ABC123 1.0 3.0##person closes a door",
            ]
        ),
        encoding="utf-8",
    )

    samples = load_charades_sta_annotations(annotations_path, split="train")

    qids = [sample["qid"] for sample in samples]
    assert len(qids) == len(set(qids))
