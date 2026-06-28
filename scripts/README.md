# Скрипты

Папка содержит публичные CLI-скрипты для подготовки данных, запуска
экспериментов и сборки таблиц результатов.

## Подготовка данных

- `prepare_data.py` извлекает выбранные видео Charades из локального zip-архива
  в `data/raw/charades/videos/`.
- `prepare_charades_sta_full_test.py` формирует manifest полного test split
  Charades-STA и JSONL-файл для Moment-DETR без запуска моделей.

## Эксперименты

- `run_clip_sweep.py` запускает CLIP sweep по `window_size`, `stride` и
  `aggregation` на Charades-STA.
- `run_smoothing_sweep.py` запускает ablation по moving-average smoothing на
  той же фиксированной выборке Charades-STA.
- `run_clip_vs_moment_detr.py` воспроизводит сравнительный запуск CLIP для
  выборки из 50 запросов.
- `run_moment_detr_probe.py` запускает Moment-DETR raw-video inference по
  подготовленному JSONL-файлу Charades-STA. Для полного test split используется
  `--limit 3720`.

## Сбор результатов

- `collect_results.py` собирает JSON-файлы результатов в компактную CSV-таблицу.
- `build_clip_vs_moment_detr_comparison.py` строит итоговую таблицу
  CLIP vs Moment-DETR по сохранённым full-test metrics.

Скрипты для основной сдачи курсовой находятся в этой папке; локальные
исследовательские и временные утилиты в публичную структуру не включаются.
