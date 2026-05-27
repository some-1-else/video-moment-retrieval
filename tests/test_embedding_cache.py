import numpy as np

from src.models.embedding_cache import (
    estimate_cache_file_size,
    load_video_embeddings_cache,
    make_video_embedding_cache_key,
    save_video_embeddings_cache,
)


def test_make_video_embedding_cache_key_is_stable() -> None:
    first_key = make_video_embedding_cache_key("video_001", "ViT-B/32", 1.0)
    second_key = make_video_embedding_cache_key("video_001", "ViT-B/32", 1.0)

    assert first_key == second_key


def test_make_video_embedding_cache_key_sanitizes_model_name() -> None:
    key = make_video_embedding_cache_key(
        video_id="video id/with spaces",
        model_name="ViT-B/32",
        fps=1.0,
    )

    assert "/" not in key
    assert " " not in key
    assert ":" not in key
    assert "ViT-B_32" in key


def test_video_embeddings_cache_roundtrip(tmp_path) -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    timestamps = [0.0, 1.0]
    metadata = {
        "video_id": "sample_video",
        "model_name": "ViT-B/32",
        "fps": 1.0,
    }
    key = make_video_embedding_cache_key("sample_video", "ViT-B/32", 1.0)

    cache_path = save_video_embeddings_cache(
        tmp_path,
        key,
        frame_timestamps=timestamps,
        image_embeddings=embeddings,
        metadata=metadata,
    )
    loaded = load_video_embeddings_cache(tmp_path, key)

    assert loaded is not None
    assert loaded["frame_timestamps"] == timestamps
    assert np.allclose(loaded["image_embeddings"], embeddings)
    assert loaded["metadata"] == metadata
    assert loaded["path"] == cache_path


def test_missing_video_embeddings_cache_returns_none(tmp_path) -> None:
    assert load_video_embeddings_cache(tmp_path, "missing") is None


def test_estimate_cache_file_size_after_save(tmp_path) -> None:
    key = make_video_embedding_cache_key("sample_video", "ViT-B/32", 1.0)
    cache_path = save_video_embeddings_cache(
        tmp_path,
        key,
        frame_timestamps=[0.0],
        image_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        metadata={"video_id": "sample_video"},
    )

    assert estimate_cache_file_size(cache_path) > 0
