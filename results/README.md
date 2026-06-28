# Результаты

Папка содержит сохранённые результаты экспериментов, использованные в курсовой
работе. В неё входят небольшие файлы с метриками, конфигурациями запусков,
сводными таблицами и предсказаниями, необходимые для проверки чисел без
повторного запуска inference.

## Основные наборы результатов

- `charades_sta_sweep_1000/` - CLIP sweep по `window_size`, `stride` и
  `aggregation` на фиксированной выборке из 1,000 запросов Charades-STA.
- `charades_sta_smoothing_1000/` - ablation по smoothing на той же выборке.
- `charades_sta_full_test_clip/` - результат CLIP `clip_w16_s8_mean` на полном
  test split Charades-STA.
- `moment_detr_charades_full_test/` - результат Moment-DETR raw video на том же
  полном test split.
- `clip_vs_moment_detr_full_test/` - итоговая таблица сравнения CLIP и
  Moment-DETR.
- `clip_vs_moment_detr_50/` и `moment_detr_charades_50/` - контрольный запуск
  на 50 запросах, использованный перед полным сравнением.
- `figures/` - место для сгенерированных фигур, если они понадобятся.

## Важные файлы

- `charades_sta_sweep_1000/summary.csv`
- `charades_sta_smoothing_1000/summary.csv`
- `charades_sta_full_test_clip/clip_w16_s8_mean/metrics.json`
- `moment_detr_charades_full_test/metrics.json`
- `clip_vs_moment_detr_full_test/comparison_summary.csv`
- `clip_vs_moment_detr_50/comparison_summary.csv`
- `moment_detr_charades_50/metrics.json`

## Кэши

CLIP embedding files (`.npz`) являются локальными cache artifacts. Они ускоряют
повторные запуски, но не являются самостоятельными результатами, поэтому не
включаются в публичную версию репозитория. Для проверки итоговых чисел
достаточно сохранённых `metrics.json`, `summary.csv`, `comparison_summary.csv`
и `run_config.json`.
