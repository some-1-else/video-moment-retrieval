"""Minimal CLIP encoder utilities."""

from __future__ import annotations

import importlib
from typing import Any


def get_default_device() -> str:
    """Return the best available torch device name."""

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def load_clip_model(model_name: str = "ViT-B/32", device: str = "auto"):
    """Load a pretrained CLIP model and preprocessing function."""

    clip = _import_clip()
    _import_torch()

    actual_device = get_default_device() if device == "auto" else device
    model, preprocess = clip.load(model_name, device=actual_device)
    model.eval()

    return model, preprocess, actual_device


def encode_text(model, text: str, device: str) -> Any:
    """Encode a text query into a normalized CLIP embedding."""

    clip = _import_clip()
    torch = _import_torch()

    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(tokens)
        text_embedding = _normalize(text_embedding)

    return text_embedding


def encode_images(
    model,
    preprocess,
    images,
    device: str,
    batch_size: int = 32,
) -> Any:
    """Encode PIL images or OpenCV numpy frames into normalized CLIP embeddings."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    torch = _import_torch()
    image_tensors = [
        preprocess(convert_image_to_pil(image))
        for image in images
    ]

    if not image_tensors:
        return torch.empty((0, 0), device=device)

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(image_tensors), batch_size):
            batch = torch.stack(image_tensors[start:start + batch_size]).to(device)
            image_embedding = model.encode_image(batch)
            embeddings.append(_normalize(image_embedding))

    return torch.cat(embeddings, dim=0)


def compute_text_image_similarity(text_embedding, image_embeddings) -> list[float]:
    """Compute cosine similarity between normalized text and image embeddings."""

    if text_embedding.dim() == 2:
        text_embedding = text_embedding.squeeze(0)

    similarities = image_embeddings @ text_embedding
    return [float(score) for score in similarities.detach().cpu().tolist()]


def convert_image_to_pil(image):
    """Convert a PIL image or an OpenCV-style numpy array to RGB PIL Image."""

    pil_image_module = _import_pil_image()

    if isinstance(image, pil_image_module.Image):
        return image.convert("RGB")

    numpy = _import_numpy()
    if isinstance(image, numpy.ndarray):
        if image.ndim == 2:
            return pil_image_module.fromarray(image).convert("RGB")

        if image.ndim == 3 and image.shape[2] >= 3:
            rgb_image = image[:, :, :3][:, :, ::-1]
            return pil_image_module.fromarray(rgb_image).convert("RGB")

    raise TypeError("images must be PIL images or numpy arrays.")


def _normalize(embedding):
    return embedding / embedding.norm(dim=-1, keepdim=True)


def _import_torch():
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise ImportError(
            "torch is required for CLIP encoding. Install project requirements, "
            "including torch and the OpenAI CLIP package."
        ) from error


def _import_clip():
    try:
        return importlib.import_module("clip")
    except ImportError as error:
        raise ImportError(
            "The OpenAI CLIP package is required. Install project requirements, "
            "including git+https://github.com/openai/CLIP.git."
        ) from error


def _import_pil_image():
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as error:
        raise ImportError(
            "Pillow is required for image preprocessing. It is installed as a "
            "dependency of the CLIP stack."
        ) from error


def _import_numpy():
    try:
        return importlib.import_module("numpy")
    except ImportError as error:
        raise ImportError("numpy is required for numpy frame conversion.") from error
