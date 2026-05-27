# CLIP Encoder

This document describes the minimal CLIP encoder layer planned for the CLIP-based Video Moment Retrieval baseline.

## Purpose

The project uses a pretrained CLIP model only as an encoder. The model is not trained or fine-tuned in this project step. The encoder layer is responsible for:

- loading a pretrained CLIP model;
- encoding a text query into a normalized text embedding;
- encoding local images or decoded video frames into normalized image embeddings;
- computing cosine similarity between the text embedding and image embeddings.

This is still not the full retrieval baseline. Dataset-level CLIP retrieval, video processing over QVHighlights, and embedding caching are separate later steps.

## Dependencies

The intended dependencies are:

```text
torch
torchvision
ftfy
regex
tqdm
git+https://github.com/openai/CLIP.git
```

The first CLIP run may download pretrained model weights. Unit tests should not call `clip.load()` and should not automatically download model weights.

## Device Selection

The helper `get_default_device()` selects:

```text
cuda -> if torch.cuda.is_available()
mps  -> if torch.backends.mps.is_available()
cpu  -> otherwise
```

`load_clip_model(..., device="auto")` uses this device selection automatically.

## Optional Smoke Test

After installing the CLIP dependencies, run:

```bash
.venv/bin/python scripts/inspect_clip.py \
  --model-name ViT-B/32 \
  --text "person opens a door"
```

The script loads CLIP, prints the selected device, encodes the text query, and prints the text embedding shape.

This command does not require videos or QVHighlights. It only checks that the local CLIP encoder layer works.

## Limitations

- This module does not run retrieval over QVHighlights.
- This module does not decode videos.
- This module does not cache embeddings yet.
- This module does not train or fine-tune CLIP.
