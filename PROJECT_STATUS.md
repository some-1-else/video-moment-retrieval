# Статус проекта

Проект находится в финальной coursework-версии для показа научному
руководителю. Основная линия работы: **Video Moment Retrieval на Charades-STA**
с воспроизводимым baseline на базе CLIP и сравнением с Moment-DETR.

QVHighlights был исходным направлением, но не является финальным benchmark в
этом репозитории: raw videos зависят от внешних YouTube-ссылок, поэтому
финальная оценка перенесена на Charades-STA.

## Что реализовано

- Загрузка и проверка Charades-STA annotations.
- Подготовка локальных Charades videos из архива через public script
  `scripts/prepare_data.py`.
- CLIP-based raw-video retrieval baseline:
  - sampling кадров;
  - CLIP image/text embeddings;
  - temporal windows;
  - mean/max aggregation;
  - optional moving-average smoothing;
  - embedding cache для повторных sweep-запусков.
- Метрики temporal localization:
  - `R@1` при `IoU = 0.3 / 0.5 / 0.7`;
  - `mIoU`.
- Контролируемые CLIP experiments на фиксированной 1,000-query Charades-STA
  subset.
- Moment-DETR raw-video integration: сначала 50-query feasibility probe, затем
  full Charades-STA test evaluation.
- Reader-facing notebooks и таблицы результатов для проверки курсовой.

## Основные результаты

### CLIP sweep на 1,000 Charades-STA queries

Источник: `results/charades_sta_sweep_1000/summary.csv`.

- Лучший coarse result:
  `clip_w16_s8_mean`, `R@1_IoU_0.3 = 0.625`.
- Лучшая strict localization:
  `clip_w8_s4_mean`, `R@1_IoU_0.5 = 0.393`,
  `R@1_IoU_0.7 = 0.181`.
- Лучший `mIoU` в sweep:
  `clip_w16_s8_mean`, `mIoU = 0.34969020291901676`.

### Smoothing на той же 1,000-query subset

Источник: `results/charades_sta_smoothing_1000/summary.csv`.

Для `w8/s4/mean` moving average smoothing немного улучшает strict metrics:

- без smoothing: `R@1_IoU_0.7 = 0.181`,
  `mIoU = 0.34777747887511207`;
- moving average 5: `R@1_IoU_0.7 = 0.190`,
  `mIoU = 0.34942105981287236`.

Для `w16/s8/mean` smoothing не улучшает `mIoU` относительно варианта без
smoothing.

### CLIP vs Moment-DETR на полном Charades-STA test split

Источники:

- `results/charades_sta_full_test_clip/clip_w16_s8_mean/run_config.json`
- `results/moment_detr_charades_full_test/metrics.json`
- `results/clip_vs_moment_detr_full_test/comparison_summary.csv`

На полном Charades-STA test split: 3,720 queries / 1,334 videos.

- CLIP `clip_w16_s8_mean`: `R@1_IoU_0.3 = 0.5944`,
  `R@1_IoU_0.5 = 0.2368`, `R@1_IoU_0.7 = 0.0823`,
  `mIoU = 0.3348`, time = 1362.68 sec, output size = 78M.
- Moment-DETR `moment_detr_raw_video`: `R@1_IoU_0.3 = 0.4145`,
  `R@1_IoU_0.5 = 0.2570`, `R@1_IoU_0.7 = 0.0995`,
  `mIoU = 0.2662`, time = 2049.43 sec, output size = 3.0M.

Interpretation: CLIP выше по `R@1_IoU_0.3` и `mIoU`; Moment-DETR выше по
строгим `R@1_IoU_0.5` и `R@1_IoU_0.7`. CLIP быстрее в CPU experiment, но
сохраняет больший cache embeddings. Это coursework-scale reproducible
comparison, а не прямое SOTA comparison.

## Что является финальным результатом

Финальная evidence line для курсовой:

- `thesis/coursework.md` - основной текст курсовой;
- `reports/results_tables.md` - компактные таблицы результатов;
- `results/charades_sta_sweep_1000/` - главный CLIP sweep;
- `results/charades_sta_smoothing_1000/` - smoothing ablation;
- `results/charades_sta_full_test_clip/`,
  `results/moment_detr_charades_full_test/` и
  `results/clip_vs_moment_detr_full_test/` - full-test CLIP vs Moment-DETR
  comparison;
- `results/clip_vs_moment_detr_50/` и
  `results/moment_detr_charades_50/` - historical 50-query feasibility probe;
- `notebooks/06_results_summary.ipynb` - основной summary notebook.

## Что является дополнительным probe/history

- 50-query Moment-DETR result остается historical raw-video feasibility probe.
  Финальное CLIP vs Moment-DETR comparison теперь выполнено на полном
  Charades-STA test split.
- QVHighlights сохранен как preliminary/history direction, но full
  QVHighlights benchmark не заявляется.
- Exploratory/smoke/synthetic результаты сохранены локально в ignored
  `.agent_memory/` и не входят в публичную evidence line.

## Ограничения

- Основные CLIP sweep/ablation используют фиксированную 1,000-query subset;
  финальное CLIP vs Moment-DETR comparison использует полный Charades-STA test
  split.
- Все reported runs выполнены на CPU.
- CLIP используется без fine-tuning.
- Moment-DETR comparison использует raw-video-compatible checkpoint и не
  является official QVHighlights benchmark или fine-tuned Charades-STA model.
- QVHighlights не стал финальным benchmark из-за нестабильной доступности raw
  videos.
- Moment-DETR checkpoint compatibility зависит от feature format: raw-video
  checkpoint работает для probe, а QVHighlights feature checkpoint не подходит
  напрямую для raw-video path.

## Как смотреть репозиторий

Рекомендуемый порядок чтения:

1. `thesis/coursework.md` - полный текст курсовой.
2. `notebooks/06_results_summary.ipynb` - короткий notebook с таблицами и
   выводами.
3. `results/README.md` - карта сохраненных результатов.
4. `reports/results_tables.md` - markdown-таблицы для быстрой проверки чисел.
5. `notebooks/03_clip_window_sweep_1000.ipynb`,
   `notebooks/04_smoothing_experiment.ipynb`,
   `notebooks/05_clip_vs_moment_detr.ipynb` - подробные notebooks по
   отдельным экспериментам.

## Технические проверки

Последняя полная проверка тестов:

```text
83 passed
```

Тесты проверяют loaders, metrics, retrieval logic, video utilities, cache
interfaces и public reproduction scripts. Для чтения курсовой руководителю
запускать тесты необязательно.
