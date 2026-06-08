# Курсовой проект: Video Moment Retrieval

Курсовой проект посвящен поиску временного фрагмента видео по текстовому
запросу (`text-to-video moment retrieval`).

Английское название темы: **Search for video moments using text queries:
implementations and experiments with video-language retrieval models**.

## Что делает проект

Проект реализует и анализирует воспроизводимый baseline для Video Moment
Retrieval: по текстовому запросу система выбирает временное окно видео, которое
лучше всего соответствует описанию.

Основная экспериментальная линия:

- датасет: **Charades-STA** с локально подготовленными Charades videos;
- baseline: CLIP `ViT-B/32` без fine-tuning;
- представление видео: sampling кадров, CLIP image embeddings, fixed temporal
  windows;
- scoring: similarity между текстовым embedding и frame/window embeddings;
- оценка: `R@1` при `IoU = 0.3 / 0.5 / 0.7`, `mIoU`,
  runtime/cache metadata.

QVHighlights был исходным направлением проекта и сохранен как preliminary
история, но полноценный QVHighlights benchmark здесь не заявляется: raw videos
зависят от внешних YouTube-ссылок, поэтому финальная оценка выполнена на
Charades-STA.

## Текущий статус

Реализовано:

- Charades-STA loader и проверки локального dataset setup;
- CLIP-based raw-video retrieval baseline;
- window/stride/aggregation sweeps;
- smoothing ablation;
- evaluation metrics для temporal localization;
- отдельный Moment-DETR raw-video feasibility probe;
- таблицы результатов и легкие notebooks для проверки курсовой.

Подробное состояние проекта и ограничения зафиксированы в `PROJECT_STATUS.md`.

## Структура репозитория

```text
configs/      конфигурации экспериментов
src/          implementation modules: data, video, retrieval, models, metrics
scripts/      public reproduction entrypoints
tests/        regression и smoke tests
notebooks/    reader-facing notebooks поверх сохраненных результатов
data/         local data layout, sample annotations, ignored raw/cache data
results/      финальные metrics, run configs, predictions и figures
reports/      таблицы результатов для быстрой проверки
thesis/       текст курсовой, figures, tables и bibliography
docs/         краткая документация: data, CLIP, Moment-DETR, limitations
archive/      старая история окружения, не нужная для основного маршрута
external/     внешний Moment-DETR код/checkpoints для probe
```

## Рекомендуемый маршрут просмотра

Рекомендуемый порядок для научного руководителя или проверяющего:

1. `thesis/coursework.md` - полный текст курсовой: постановка задачи,
   методика, результаты, ограничения и источники.
2. `notebooks/06_results_summary.ipynb` - компактный notebook с сохраненными
   таблицами результатов и выводами; он читает public `results/` files и не
   запускает model inference.
3. `results/README.md` - карта сохраненных experiment outputs, использованных
   в курсовой.
4. `notebooks/03_clip_window_sweep_1000.ipynb`,
   `notebooks/04_smoothing_experiment.ipynb`,
   `notebooks/05_clip_vs_moment_detr.ipynb` - подробные notebooks по main
   sweep, smoothing ablation и Moment-DETR feasibility probe.
5. `PROJECT_STATUS.md` - краткое резюме статуса проекта на русском.

Тесты запускать необязательно: они полезны для engineering confidence, но не
нужны для просмотра результатов курсовой. Последняя полная проверка прошла с
результатом `83 passed`.

## Воспроизводимые эксперименты

Основной маршрут воспроизведения: Charades-STA + CLIP baseline. Полезные файлы:

- зависимости: `requirements.txt`;
- конфигурация: `configs/clip_baseline.yaml`;
- подготовка данных: `data/README.md`, `docs/data_setup.md`;
- методика: `docs/clip_baseline.md`, `docs/video_frame_extraction.md`,
  `docs/moment_detr.md`, `docs/qvhighlights_limitations.md`;
- реализация: `src/`;
- public experiment runners:
  - `scripts/run_clip_sweep.py`;
  - `scripts/run_smoothing_sweep.py`;
  - `scripts/run_clip_vs_moment_detr.py`;
  - `scripts/run_moment_detr_probe.py`.

Файлы `run_config.json`, `metrics.json`, `result.json` и `summary.csv`
сохранены рядом с соответствующими result folders, чтобы reported numbers
можно было проверить без повторного запуска inference.

Notebooks в `notebooks/` являются reader-facing материалами поверх сохраненных
`results/*.json` и `results/*.csv`. Основная model logic находится не в
notebooks, а в `src/` и `scripts/`.

## Результаты

Финальные результаты курсовой сосредоточены на Charades-STA:

- `results/charades_sta_sweep_1000/summary.csv` - CLIP
  window/stride/aggregation sweep на фиксированной 1,000-query subset
  Charades-STA.
- `results/charades_sta_smoothing_1000/summary.csv` - smoothing ablation на той
  же 1,000-query subset.
- `results/clip_vs_moment_detr_50/comparison_summary.csv` и
  `results/moment_detr_charades_50/metrics.json` - CLIP vs Moment-DETR
  raw-video comparison на фиксированной 50-query subset.
- `reports/results_tables.md` - компактные таблицы результатов.

Ключевые числа:

- Лучший coarse CLIP result на 1,000-query sweep:
  `clip_w16_s8_mean`, `R@1_IoU_0.3 = 0.625`.
- Лучшая strict CLIP localization в этом sweep:
  `clip_w8_s4_mean`, `R@1_IoU_0.5 = 0.393`,
  `R@1_IoU_0.7 = 0.181`.
- Лучший `mIoU` в sweep:
  `clip_w16_s8_mean`, `mIoU = 0.34969020291901676`.
- На фиксированной 50-query comparison subset лучший CLIP strict result:
  `clip_w8_s4_mean`, `R@1_IoU_0.7 = 0.32`,
  `mIoU = 0.4393782459360136`.
- Moment-DETR raw-video probe на той же 50-query subset:
  `R@1_IoU_0.3 = 0.44`, `R@1_IoU_0.5 = 0.32`,
  `R@1_IoU_0.7 = 0.04`, `mIoU = 0.26340237468832634`.

Exploratory результаты сохранены локально, но не входят в public evidence line:

- ранние Charades-STA smoke/200-query runs;
- preliminary QVHighlights sweeps;
- synthetic/sample experiments;
- one-video и diagnostic probes.

Они находятся в ignored `.agent_memory/results/`. Caches и embeddings, особенно
`.npz` files, нужны только для ускорения локальных reruns и не являются
самостоятельными финальными результатами.

## Ограничения

- Основной CLIP sweep использует фиксированную 1,000-query subset, а не полный
  Charades-STA test split.
- Moment-DETR comparison использует только 50 examples и должен читаться как
  feasibility probe, а не как full official benchmark.
- Все reported runs выполнены на CPU.
- CLIP не fine-tuned на Charades-STA.
- QVHighlights остается preliminary направлением из-за зависимости raw videos
  от внешних YouTube-ссылок.
- Moment-DETR checkpoint compatibility зависит от feature format:
  raw-video-compatible checkpoint работает для probe, а QVHighlights feature
  checkpoint не подходит напрямую для raw-video path.

## Материалы курсовой

Основной текст:

```text
thesis/coursework.md
```

Финальная папка `thesis/` намеренно небольшая:

```text
thesis/
├── coursework.md
├── figures/
├── tables/
└── references.bib
```

Промежуточные writing notes перенесены в local ignored agent memory и не входят
в финальную структуру coursework repository. Самый удобный result-facing
summary для быстрой проверки: `reports/results_tables.md`.

## Legacy и archive

`archive/` хранит старые environment/history files, не нужные для основного
Charades-STA reproduction path. Local ignored `.agent_memory/` хранит
intermediate cleanup notes, deprecated planning documents, exploratory
QVHighlights/synthetic history и другую рабочую память Codex/OpenCode. Эти
файлы сохранены локально для traceability, но не нужны для понимания или
воспроизведения public coursework results.
