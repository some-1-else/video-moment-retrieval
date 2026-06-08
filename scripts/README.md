# Скрипты

Эта папка содержит public reproduction entrypoints для курсовой.

## Подготовка данных

- `prepare_data.py` извлекает выбранные Charades videos из локального zip
  archive в ожидаемую raw-video структуру.

## Эксперименты

- `run_clip_sweep.py` запускает основной CLIP window/stride/aggregation sweep
  на Charades-STA.
- `run_smoothing_sweep.py` запускает CLIP smoothing ablation на той же
  Charades-STA setup.
- `run_clip_vs_moment_detr.py` запускает CLIP-часть фиксированного 50-query
  comparison и записывает combined comparison table.
- `run_moment_detr_probe.py` запускает Moment-DETR raw-video feasibility probe
  на фиксированной Charades-STA subset.

## Сбор результатов

- `collect_results.py` собирает JSON result files в компактную CSV summary.

Старые probe, synthetic, QVHighlights и debug scripts перенесены в локальную
ignored agent memory и не входят в public reproduction path.
