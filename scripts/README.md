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
- `run_clip_vs_moment_detr.py` запускает CLIP-часть historical 50-query
  comparison и записывает combined comparison table.
- `run_moment_detr_probe.py` запускает Moment-DETR raw-video inference на
  подготовленном Charades-STA JSONL; historical default остается 50-query
  subset, full-test запуск использует `--limit 3720`.

## Сбор результатов

- `collect_results.py` собирает JSON result files в компактную CSV summary.

## Full-test CLIP vs Moment-DETR

- `prepare_charades_sta_full_test.py` готовит общий Charades-STA test manifest
  и Moment-DETR JSONL без запуска моделей.
- `run_clip_sweep.py --config-name clip_w16_s8_mean` запускает выбранный CLIP
  full-test baseline на подготовленном manifest.
- `run_moment_detr_probe.py --limit 3720` запускает Moment-DETR full-test
  inference на подготовленном JSONL.
- `build_clip_vs_moment_detr_comparison.py` собирает comparison table из
  сохраненных CLIP и Moment-DETR metrics.

Старые probe, synthetic, QVHighlights и debug scripts перенесены в локальную
ignored agent memory и не входят в public reproduction path.
