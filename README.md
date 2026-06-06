# Video Moment Retrieval Coursework

Курсовой проект посвящен поиску временного фрагмента видео по текстовому
запросу (`text-to-video moment retrieval`). Практическая цель: собрать
воспроизводимый baseline, провести контролируемые эксперименты и сравнить
качество локализации с вычислительной стоимостью.

Английское название темы: **Search for video moments using text queries:
implementations and experiments with video-language retrieval models**.

## Dataset

Основной датасет для финальных экспериментов: **Charades-STA**. Он используется
с локально подготовленными Charades videos и подходит для воспроизводимого
raw-video baseline.

Предварительное направление: **QVHighlights**. Код, заметки и ранние результаты
по QVHighlights сохранены в репозитории, но raw videos зависят от YouTube и
оказались менее надежными для основной курсовой постановки.

Полезные заметки:

- Charades-STA setup: `docs/charades_sta_data_setup.md`
- QVHighlights data notes: `docs/qvhighlights_data.md`
- Moment-DETR probe log: `docs/moment_detr_probe_log.md`

## Main Experiments

Основной baseline: CLIP-based retrieval без fine-tuning. Видео сэмплируется по
кадрам, кадры и запрос кодируются CLIP `ViT-B/32`, frame-level similarities
агрегируются в fixed temporal windows, затем выбирается top-1 window.

Главные сохраненные эксперименты:

- `results/charades_sta_baseline_200/` - CLIP baseline на 200 Charades-STA
  queries.
- `results/charades_sta_sweep_1000/summary.csv` - sweep window/stride/aggregation
  на фиксированных 1,000 Charades-STA queries.
- `results/charades_sta_smoothing_1000/summary.csv` - smoothing sweep на тех же
  1,000 queries.
- `results/clip_vs_moment_detr_50/` и `results/moment_detr_charades_50/` -
  CLIP vs Moment-DETR raw-video probe на 50 queries.

Метрики: `R@1` при `IoU = 0.3 / 0.5 / 0.7`, `mIoU`, inference time и cache
statistics там, где они сохранены.

## How To Read Results

Для быстрого чтения результатов начните с:

- `reports/results_tables.md` - финальные таблицы для руководителя.
- `PROJECT_STATUS.md` - что реализовано, протестировано, какие есть ограничения.
- `notebooks/06_results_summary.ipynb` - lightweight summary notebook.

Все notebooks в `notebooks/` читают уже сохраненные `results/*.json` и
`results/*.csv`. Они не запускают CLIP или Moment-DETR заново.

## Draft

Основной черновик курсовой лежит в:

```text
thesis/coursework_draft_v2.md
```

Дополнительные черновики и обсуждение результатов находятся в `thesis/`.

## Repository Map

```text
notebooks/      readable coursework notebooks
reports/        final result tables for review
results/        saved metrics, configs, predictions, cache artifacts
thesis/         coursework drafts
docs/           setup notes and experiment logs
src/            implementation modules
scripts/        reproducibility and experiment entry points
tests/          regression tests
external/       external Moment-DETR code/checkpoint used for probe
```

Scripts and tests are kept for reproducibility, but the recommended reading
interface for coursework review is `README.md`, `PROJECT_STATUS.md`,
`reports/results_tables.md`, `notebooks/`, and `thesis/coursework_draft_v2.md`.
