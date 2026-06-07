# Video Moment Retrieval Coursework

Курсовой проект посвящен поиску временного фрагмента видео по текстовому
запросу (`text-to-video moment retrieval`).

Английское название темы: **Search for video moments using text queries:
implementations and experiments with video-language retrieval models**.

## What this project does

Проект реализует и анализирует воспроизводимый baseline для Video Moment
Retrieval: по текстовому запросу система выбирает временное окно видео, которое
лучше всего соответствует описанию.

Основная экспериментальная линия:

- датасет: **Charades-STA** с локально подготовленными Charades videos;
- baseline: CLIP `ViT-B/32` без fine-tuning;
- представление видео: sampling кадров, CLIP image embeddings, fixed temporal
  windows;
- scoring: similarity между текстовым embedding и frame/window embeddings;
- оценка: `R@1` при `IoU = 0.3 / 0.5 / 0.7`, `mIoU`, runtime/cache metadata.

QVHighlights сохранен как preliminary направление: код, заметки и ранние
результаты есть в репозитории, но полноценный QVHighlights benchmark здесь не
заявляется, потому что raw videos зависят от внешних YouTube-ссылок.

## Current status

Реализовано:

- Charades-STA loader and dataset probes;
- CLIP-based raw-video retrieval baseline;
- window/stride/aggregation sweeps;
- optional smoothing experiments;
- evaluation metrics for temporal localization;
- isolated Moment-DETR raw-video feasibility probe;
- result tables and lightweight notebooks for coursework review.

Подробное состояние проекта и ограничения зафиксированы в
`PROJECT_STATUS.md`.

## Repository structure

```text
configs/      experiment configuration files
src/          implementation modules: data, video, retrieval, models, metrics
scripts/      reproducibility and experiment entry points
tests/        regression and smoke tests
notebooks/    readable coursework notebooks over saved results
data/         local data layout, sample annotations, ignored raw/cache data
results/      final coursework metrics, run configs, predictions, and figures
reports/      supervisor-facing result tables
thesis/       final coursework text, figures, tables, and bibliography
docs/         concise coursework documentation: data, CLIP, Moment-DETR, limits
archive/      old environment/history files not needed for the main path
external/     external Moment-DETR code/checkpoints used for the probe
```

## Recommended review path

Recommended route for a scientific supervisor or reviewer:

1. `thesis/coursework.md` - full coursework draft with motivation, method,
   experimental setup, results, limitations, and references.
2. `notebooks/06_results_summary.ipynb` - compact notebook with saved result
   tables and conclusions; it reads public `results/` files and does not run
   model inference.
3. `results/README.md` - map of saved experiment outputs used by the
   coursework.
4. `notebooks/03_clip_window_sweep_1000.ipynb`,
   `notebooks/04_smoothing_experiment.ipynb`, and
   `notebooks/05_clip_vs_moment_detr.ipynb` - detailed reader-facing notebooks
   for the main sweep, smoothing ablation, and Moment-DETR feasibility probe.
5. `PROJECT_STATUS.md` - concise Russian status summary.

Tests are optional for review. They are useful for engineering confidence, but
the supervisor does not need to run them to inspect the coursework results. The
latest full test check passed with `83 passed`.

## Reproducible experiments

The main reproducibility path is Charades-STA + CLIP baseline. Useful files:

- dependencies: `requirements.txt`;
- configuration: `configs/clip_baseline.yaml`;
- data notes: `data/README.md`, `docs/data_setup.md`;
- method notes: `docs/clip_baseline.md`, `docs/video_frame_extraction.md`,
  `docs/moment_detr.md`, `docs/qvhighlights_limitations.md`;
- core implementation: `src/`;
- final experiment runners:
  - `scripts/run_clip_sweep.py`;
  - `scripts/run_smoothing_sweep.py`;
  - `scripts/run_clip_vs_moment_detr.py`;
  - `scripts/run_moment_detr_probe.py`.

Saved `run_config.json`, `metrics.json`, `result.json`, and `summary.csv` files
are kept with the corresponding result folders so that reported numbers can be
checked without rerunning all inference.

All notebooks in `notebooks/` are designed as readable analysis notebooks over
saved final `results/*.json` and `results/*.csv` artifacts. They are not the
primary place where model logic lives.

## Results

Final coursework results are centered on Charades-STA:

- `results/charades_sta_sweep_1000/summary.csv` - CLIP window/stride/aggregation
  sweep on a fixed 1,000-query Charades-STA subset.
- `results/charades_sta_smoothing_1000/summary.csv` - smoothing sweep on the
  same 1,000-query subset.
- `results/clip_vs_moment_detr_50/comparison_summary.csv` and
  `results/moment_detr_charades_50/metrics.json` - CLIP vs Moment-DETR
  raw-video comparison on a fixed 50-query subset.
- `reports/results_tables.md` - compact result tables for coursework review.

Reported highlights, copied from `PROJECT_STATUS.md`:

- Best coarse CLIP result on the 1,000-query sweep:
  `clip_w16_s8_mean`, `R@1_IoU_0.3 = 0.625`.
- Best strict CLIP localization in that sweep:
  `clip_w8_s4_mean`, `R@1_IoU_0.5 = 0.393`,
  `R@1_IoU_0.7 = 0.181`.
- Best mIoU in the sweep:
  `clip_w16_s8_mean`, `mIoU = 0.34969020291901676`.
- On the fixed 50-query comparison subset, best CLIP strict result:
  `clip_w8_s4_mean`, `R@1_IoU_0.7 = 0.32`,
  `mIoU = 0.4393782459360136`.
- Moment-DETR raw-video probe on the same 50-query subset:
  `R@1_IoU_0.3 = 0.44`, `R@1_IoU_0.5 = 0.32`,
  `R@1_IoU_0.7 = 0.04`, `mIoU = 0.26340237468832634`.

Exploratory results are preserved locally, but they are not part of the public
coursework evidence line:

- earlier Charades-STA smoke/200-query runs;
- QVHighlights preliminary sweeps;
- synthetic/sample experiments;
- one-video and diagnostic probes.

They live under ignored `.agent_memory/results/`. Caches and embeddings,
especially `.npz` files, are kept only to speed up local reruns and should not
be read as standalone final results.

## Limitations

- The main CLIP sweep uses a fixed 1,000-query subset, not the full
  Charades-STA test split.
- Moment-DETR comparison uses only 50 examples and should be read as a
  feasibility probe, not a full official benchmark.
- All reported runs use CPU inference.
- No model is fine-tuned on Charades-STA.
- QVHighlights remains preliminary because raw video availability depends on
  external YouTube links.
- Moment-DETR checkpoint compatibility is sensitive to feature format: the
  raw-video-compatible checkpoint works for the probe, while the QVHighlights
  feature checkpoint is not directly compatible with the raw-video path.

## Thesis materials

Main coursework text:

```text
thesis/coursework.md
```

The final thesis folder is intentionally small:

```text
thesis/
├── coursework.md
├── figures/
├── tables/
└── references.bib
```

Intermediate writing notes were moved to local ignored agent memory and are not
part of the final coursework repository structure. The most useful
result-facing summary for review is `reports/results_tables.md`.

## Legacy and archive notes

`archive/` stores old environment/history files that are not needed for the main
Charades-STA reproduction path. Local ignored `.agent_memory/` stores
intermediate cleanup notes, deprecated planning documents, exploratory
QVHighlights/synthetic history, and other Codex/OpenCode working memory. These
files are preserved locally for transparency, but they are not required for
understanding or reproducing the public coursework results.
