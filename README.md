# Video Moment Retrieval

**Английское название курсовой:** Search for video moments using text queries: implementations and experiments with video-language retrieval models.

## Цель

Сравнить подходы к `text-to-video moment retrieval` по качеству и вычислительной стоимости.

## Описание задачи

Проект посвящён поиску релевантных временных отрезков в видео по текстовому запросу. Практический фокус работы: собрать воспроизводимый baseline, провести серию простых экспериментов и оценить компромисс между качеством и стоимостью inference.

## Датасет

Основной датасет: `QVHighlights`.

For real QVHighlights annotation setup, see [docs/qvhighlights_data.md](docs/qvhighlights_data.md).

For a small local QVHighlights video subset workflow, see [docs/qvhighlights_video_subset.md](docs/qvhighlights_video_subset.md).

## Baseline

Основной baseline: `CLIP-based retrieval` с извлечением визуальных embedding-ов и сопоставлением с текстовым запросом.

CLIP baseline пока находится на стадии планирования. План реализации описан в [docs/clip_baseline_plan.md](docs/clip_baseline_plan.md). Текущий runnable baseline - это dummy retrieval pipeline без видео и CLIP.

Current implementation state is summarized in [docs/implementation_status.md](docs/implementation_status.md).

Local video frame extraction is documented in [docs/video_frame_extraction.md](docs/video_frame_extraction.md). This is an optional local smoke-test layer and does not download QVHighlights videos.

CLIP encoder setup is documented in [docs/clip_encoder.md](docs/clip_encoder.md). This layer only checks pretrained CLIP encoding utilities; it is not the full retrieval baseline.

## Метрики

- `Recall@1` при `IoU = 0.3 / 0.5 / 0.7`
- `mIoU`
- `inference time`
- число обработанных кадров
- размер сохранённых embedding-ов

## Quick smoke test

После установки зависимостей можно проверить текущий dummy pipeline на маленьком искусственном annotation-файле:

```bash
.venv/bin/python scripts/run_dummy_retrieval.py \
  --annotations data/sample/qvhighlights_sample.jsonl \
  --window-size 4 \
  --stride 2 \
  --output results/sample_dummy_retrieval.json
```

Это не `CLIP` baseline и не настоящий `QVHighlights`. Это sanity-check, который проверяет связку:

```text
annotations -> windows -> dummy scores -> prediction -> metrics
```

Можно также запустить frame-score rehearsal pipeline. Он имитирует будущую структуру CLIP baseline, но использует deterministic fake frame scores и не читает видео:

```bash
.venv/bin/python scripts/run_frame_score_retrieval.py \
  --annotations data/sample/qvhighlights_sample.jsonl \
  --fps 1 \
  --window-size 4 \
  --stride 2 \
  --aggregation mean \
  --smoothing-window 3 \
  --output results/sample_frame_score_retrieval.json
```

Это rehearsal pipeline, а не `CLIP` baseline.

После нескольких запусков можно собрать JSON results в одну CSV-таблицу:

```bash
.venv/bin/python scripts/collect_results.py \
  --results-dir results \
  --output results/summary.csv
```

`summary.csv` можно использовать для таблиц с метриками и вычислительной стоимостью в курсовой.

Для проверки experiment workflow можно запустить маленькую toy matrix на sample data:

```bash
bash scripts/run_sample_experiments.sh
```

Скрипт создаёт несколько JSON results в `results/sample_experiments/` и собирает `results/sample_experiments/summary.csv`. Это не `CLIP` baseline и не реальные результаты по `QVHighlights`, а только воспроизводимая проверка запуска экспериментов.

Если реальные QVHighlights annotations уже вручную лежат в `data/qvhighlights/highlight_val_release.jsonl`, можно проверить pipeline на первых 100 samples:

```bash
bash scripts/run_qvhighlights_annotation_experiments.sh
```

Эта команда также не читает видео и не запускает `CLIP`; она проверяет annotation loader, rehearsal retrieval и result collection на локальных annotations.

Optional local video smoke test:

```bash
.venv/bin/python scripts/inspect_video.py \
  --video path/to/video.mp4 \
  --fps 1
```

Эта команда проверяет чтение одного локального видеофайла и извлечение sampled frames. Она не запускает `CLIP` и не обрабатывает датасет.

Optional CLIP encoder smoke test:

```bash
.venv/bin/python scripts/inspect_clip.py \
  --model-name ViT-B/32 \
  --text "person opens a door"
```

Эта команда проверяет загрузку pretrained `CLIP` и encoding одного text query. Она не запускает retrieval baseline и не требует QVHighlights videos.

Optional one-video CLIP retrieval smoke test:

```bash
.venv/bin/python scripts/run_one_video_clip_retrieval.py \
  --video path/to/video.mp4 \
  --query "person opens a door" \
  --fps 1 \
  --window-size 8 \
  --stride 4 \
  --aggregation mean \
  --model-name ViT-B/32 \
  --output results/one_video_clip_retrieval.json
```

See [docs/one_video_clip_retrieval.md](docs/one_video_clip_retrieval.md). This is a local one-video smoke test, not dataset-level QVHighlights evaluation.

Tiny synthetic CLIP benchmark:

```bash
.venv/bin/python scripts/create_synthetic_clip_dataset.py

.venv/bin/python scripts/run_clip_dataset_retrieval.py \
  --annotations data/sample/synthetic_clip_annotations.jsonl \
  --videos-dir data/sample/synthetic_clip_videos \
  --fps 1 \
  --window-size 2 \
  --stride 1 \
  --aggregation mean \
  --model-name ViT-B/32 \
  --output results/synthetic_clip_dataset_retrieval.json
```

See [docs/synthetic_clip_benchmark.md](docs/synthetic_clip_benchmark.md). This uses real CLIP similarities on toy synthetic videos, not QVHighlights results.

The tiny synthetic CLIP benchmark has been executed successfully. Output JSON:

```text
results/synthetic_clip_dataset_retrieval.json
```

The reported synthetic metrics are not QVHighlights results.

Small QVHighlights video subset manifest:

```bash
.venv/bin/python scripts/create_qvhighlights_subset_manifest.py \
  --annotations data/qvhighlights/highlight_val_release.jsonl \
  --videos-dir data/qvhighlights/videos \
  --limit 20 \
  --output data/qvhighlights/qvhighlights_val_subset_manifest.json

.venv/bin/python scripts/inspect_qvhighlights_subset_manifest.py \
  --manifest data/qvhighlights/qvhighlights_val_subset_manifest.json
```

This does not download videos or run `CLIP`. It only checks which videos from a small annotation subset are already available locally.
The manifest also includes parsed `youtube_id`, `clip_start`, and `clip_end` fields when the QVHighlights `video_id` follows the observed `<youtube_id>_<clip_start>_<clip_end>` pattern.

To avoid selecting many clips from one unavailable YouTube source, create a diverse manifest with at most one sample per `youtube_id`:

```bash
.venv/bin/python scripts/create_qvhighlights_subset_manifest.py \
  --annotations data/qvhighlights/highlight_val_release.jsonl \
  --videos-dir data/qvhighlights/videos \
  --limit 20 \
  --strategy diverse-youtube \
  --output data/qvhighlights/qvhighlights_val_diverse_subset_manifest.json
```

Optional QVHighlights subset download dry run:

```bash
.venv/bin/python scripts/download_qvhighlights_subset_videos.py \
  --manifest data/qvhighlights/qvhighlights_val_subset_manifest.json \
  --output-dir data/qvhighlights/videos \
  --max-downloads 3 \
  --dry-run
```

See [docs/qvhighlights_video_download.md](docs/qvhighlights_video_download.md). Real video download is optional, limited to a tiny subset, and may fail if source YouTube videos are unavailable.

Run CLIP retrieval only on locally available QVHighlights subset videos:

```bash
.venv/bin/python scripts/run_qvhighlights_available_clip_retrieval.py \
  --manifest data/qvhighlights/qvhighlights_val_subset_manifest.json \
  --fps 1 \
  --window-size 8 \
  --stride 4 \
  --aggregation mean \
  --model-name ViT-B/32 \
  --output results/qvhighlights_available_clip_retrieval.json
```

See [docs/qvhighlights_clip_subset.md](docs/qvhighlights_clip_subset.md). This uses only videos marked as locally available in the manifest and is not a full QVHighlights validation result.

First QVHighlights subset CLIP run completed:

```text
results/qvhighlights_available_clip_retrieval_diverse.json
```

This run used 2 locally available validation clips and real CLIP similarities. It is a technical validation of the pipeline, not a full QVHighlights benchmark.

Validate local QVHighlights videos before running CLIP on a larger subset:

```bash
.venv/bin/python scripts/validate_qvhighlights_videos.py \
  --manifest data/qvhighlights/qvhighlights_val_diverse50_subset_manifest.json \
  --output data/qvhighlights/qvhighlights_val_diverse50_video_validation.json
```

Run CLIP retrieval using only videos marked as readable:

```bash
.venv/bin/python scripts/run_qvhighlights_available_clip_retrieval.py \
  --manifest data/qvhighlights/qvhighlights_val_diverse50_subset_manifest.json \
  --validation-report data/qvhighlights/qvhighlights_val_diverse50_video_validation.json \
  --fps 1 \
  --window-size 16 \
  --stride 8 \
  --aggregation mean \
  --model-name ViT-B/32 \
  --output results/qvhighlights_available_clip_retrieval_diverse50_w16_s8_mean.json
```

Run the same readable-subset CLIP retrieval with image embedding cache enabled:

```bash
.venv/bin/python scripts/run_qvhighlights_available_clip_retrieval.py \
  --manifest data/qvhighlights/qvhighlights_val_diverse50_subset_manifest.json \
  --validation-report data/qvhighlights/qvhighlights_val_diverse50_video_validation.json \
  --fps 1 \
  --window-size 16 \
  --stride 8 \
  --aggregation mean \
  --model-name ViT-B/32 \
  --use-cache \
  --embeddings-cache-dir results/embeddings/qvhighlights_readable \
  --output results/qvhighlights_available_clip_retrieval_cached.json
```

Run a preliminary parameter sweep on the locally available QVHighlights subset:

```bash
bash scripts/run_qvhighlights_available_sweep.sh
```

This writes JSON results and `results/qvhighlights_available_sweep/summary.csv`. It is an available-subset sweep, not a full validation benchmark.

The preliminary available-subset sweep has been completed. Summary:

```text
results/qvhighlights_available_sweep/summary.csv
```

The sweep used only 2 locally available validation clips, so it should not be treated as final QVHighlights benchmark performance.

Run a preliminary parameter sweep on QVHighlights videos that passed local readability validation:

```bash
bash scripts/run_qvhighlights_readable_sweep.sh
```

This uses `data/qvhighlights/qvhighlights_val_diverse50_video_validation.json` to filter unreadable files and writes `results/qvhighlights_readable_sweep/summary.csv`. It is still a readable-subset experiment, not a full validation benchmark.

Readable-subset sweep results have been drafted for the thesis in [thesis/experiments_draft.md](thesis/experiments_draft.md). The source summary table is:

```text
results/qvhighlights_readable_sweep/summary.csv
```

Embedding cache smoke experiment completed:

```text
results/qvhighlights_available_clip_retrieval_cached_first.json
results/qvhighlights_available_clip_retrieval_cached_second.json
```

The first run populated the image embedding cache, and the second run reused it. Inference time decreased from `40.3804s` to `2.8471s`, approximately `14.2x` faster, with identical metrics.

Run the readable-subset sweep with image embedding cache enabled:

```bash
bash scripts/run_qvhighlights_readable_sweep_cached.sh
```

This writes JSON results and `results/qvhighlights_readable_sweep_cached/summary.csv`, including cache hit/miss columns.

Cached readable-subset sweep completed:

```text
results/qvhighlights_readable_sweep_cached/summary.csv
```

Metrics matched the uncached sweep exactly. Total inference time decreased from `332.5960s` to `20.7162s`, about `16.05x` faster.

A consolidated results summary is available in [thesis/results_summary.md](thesis/results_summary.md).

The assembled coursework draft is available in [thesis/coursework_draft.md](thesis/coursework_draft.md).

## Текущий статус

Сейчас проект находится на стадии подготовки документации и плана работ. Следующий шаг: зафиксировать окружение, формат данных и протокол оценки, после чего перейти к воспроизводимому baseline.
