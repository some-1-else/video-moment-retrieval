# Результаты

Эта папка содержит финальные сохраненные результаты, использованные в курсовой.

## Публичные результаты курсовой

- `charades_sta_sweep_1000/` - финальный CLIP window/stride/aggregation sweep
  на фиксированной 1,000-query subset Charades-STA.
- `charades_sta_smoothing_1000/` - финальный smoothing ablation на той же
  1,000-query subset.
- `charades_sta_full_test_clip/` - CLIP `clip_w16_s8_mean` result на полном
  Charades-STA test split.
- `moment_detr_charades_full_test/` - Moment-DETR raw-video result на том же
  полном Charades-STA test split.
- `clip_vs_moment_detr_full_test/` - финальная full-test comparison table.
- `clip_vs_moment_detr_50/` и `moment_detr_charades_50/` - historical
  50-query feasibility probe, предшествующий full-test evaluation.
- `figures/` - optional generated figures для reports или материалов курсовой.

Основные summary files:

- `charades_sta_sweep_1000/summary.csv`
- `charades_sta_smoothing_1000/summary.csv`
- `clip_vs_moment_detr_full_test/comparison_summary.csv`
- `charades_sta_full_test_clip/clip_w16_s8_mean/metrics.json`
- `moment_detr_charades_full_test/metrics.json`
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
