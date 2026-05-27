import importlib.util

import pytest

from src.models import clip_encoder
from src.models.clip_encoder import convert_image_to_pil, get_default_device


def test_get_default_device_can_be_called() -> None:
    assert get_default_device() in {"cuda", "mps", "cpu"}


def test_convert_image_to_pil_converts_opencv_bgr_numpy_frame() -> None:
    numpy = pytest.importorskip("numpy")

    bgr_frame = numpy.zeros((2, 2, 3), dtype=numpy.uint8)
    bgr_frame[:, :, 0] = 255

    image = convert_image_to_pil(bgr_frame)

    assert image.mode == "RGB"
    assert image.size == (2, 2)
    assert image.getpixel((0, 0)) == (0, 0, 255)


def test_compute_text_image_similarity_with_torch_tensors() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed in this environment")

    import torch

    text_embedding = torch.tensor([1.0, 0.0])
    image_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    similarities = clip_encoder.compute_text_image_similarity(
        text_embedding,
        image_embeddings,
    )

    assert similarities == pytest.approx([1.0, 0.0])


def test_import_clip_error_message_is_actionable(monkeypatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name):
        if name == "clip":
            raise ImportError("missing clip")
        return real_import_module(name)

    monkeypatch.setattr(clip_encoder.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="OpenAI CLIP package"):
        clip_encoder._import_clip()
