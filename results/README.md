# Результаты

Эта папка содержит финальные сохраненные результаты, использованные в курсовой.

## Публичные результаты курсовой

- `charades_sta_sweep_1000/` - финальный CLIP window/stride/aggregation sweep
  на фиксированной 1,000-query subset Charades-STA.
- `charades_sta_smoothing_1000/` - финальный smoothing ablation на той же
  1,000-query subset.
- `clip_vs_moment_detr_50/` - CLIP results для фиксированной 50-query
  comparison subset.
- `moment_detr_charades_50/` - Moment-DETR raw-video probe results на той же
  фиксированной 50-query subset.
- `figures/` - optional generated figures для reports или материалов курсовой.

Основные summary files:

- `charades_sta_sweep_1000/summary.csv`
- `charades_sta_smoothing_1000/summary.csv`
- `clip_vs_moment_detr_50/comparison_summary.csv`
- `moment_detr_charades_50/metrics.json`

## Локальная исследовательская история

Exploratory outputs, старые probe runs, preliminary QVHighlights results,
sample/synthetic demos и embedding caches сохранены локально в:

```text
.agent_memory/results/
```

Эта папка игнорируется git и не является частью public coursework results
surface. Файлы полезны для локальной traceability, но не входят в evidence path
для финальных reported numbers.

## Кэши

Embedding `.npz` files являются local cache artifacts. Они ускоряют reruns, но
не являются самостоятельными финальными результатами. Public result folders
хранят небольшие metrics, configs, summaries и prediction outputs, необходимые
для проверки reported experiments.
