# Таблицы результатов

Таблицы ниже перенесены из сохранённых result artifacts без изменения
экспериментальных чисел.

## CLIP sweep на 1,000 запросов

Источник: `results/charades_sta_sweep_1000/summary.csv`.

| Configuration | Window | Stride | Aggregation | Queries | Unique Videos | R@1 IoU 0.3 | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU | Time (s) | Cache Hits | Cache Misses |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clip_w8_s4_mean | 8.0 | 4.0 | mean | 1000 | 363 | 0.509 | 0.393 | 0.181 | 0.34777747887511207 | 309.6866364169982 | 637 | 363 |
| clip_w16_s8_mean | 16.0 | 8.0 | mean | 1000 | 363 | 0.625 | 0.26 | 0.096 | 0.34969020291901676 | 22.592478417005623 | 1000 | 0 |
| clip_w32_s16_mean | 32.0 | 16.0 | mean | 1000 | 363 | 0.395 | 0.014 | 0.0 | 0.2814292262260163 | 21.506190500003868 | 1000 | 0 |
| clip_w16_s8_max | 16.0 | 8.0 | max | 1000 | 363 | 0.602 | 0.232 | 0.077 | 0.33861871441230174 | 20.922286917004385 | 1000 | 0 |

## Smoothing ablation

Источник: `results/charades_sta_smoothing_1000/summary.csv`.

| Configuration | Window | Stride | Aggregation | Smoothing | Queries | R@1 IoU 0.3 | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU | Time (s) |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| clip_w8_s4_mean_none | 8.0 | 4.0 | mean | none | 1000 | 0.509 | 0.393 | 0.181 | 0.34777747887511207 | 17.297376041009557 |
| clip_w8_s4_mean_moving_average_3 | 8.0 | 4.0 | mean | moving_average_3 | 1000 | 0.511 | 0.398 | 0.183 | 0.3485195232005171 | 15.721828207999351 |
| clip_w8_s4_mean_moving_average_5 | 8.0 | 4.0 | mean | moving_average_5 | 1000 | 0.509 | 0.398 | 0.19 | 0.34942105981287236 | 15.789070291997632 |
| clip_w16_s8_mean_none | 16.0 | 8.0 | mean | none | 1000 | 0.625 | 0.26 | 0.096 | 0.34969020291901676 | 14.78721945900179 |
| clip_w16_s8_mean_moving_average_3 | 16.0 | 8.0 | mean | moving_average_3 | 1000 | 0.62 | 0.257 | 0.098 | 0.3490817358808653 | 15.369382458011387 |
| clip_w16_s8_mean_moving_average_5 | 16.0 | 8.0 | mean | moving_average_5 | 1000 | 0.623 | 0.254 | 0.098 | 0.3480849408190796 | 15.99566145800054 |

## CLIP vs Moment-DETR на полном test split

Источник: `results/clip_vs_moment_detr_full_test/comparison_summary.csv`.

Полный Charades-STA test split: 3,720 queries / 1,334 videos.

| Model | Configuration | Queries | Failed | R@1 IoU 0.3 | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU | Time (s) | Output size |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP | clip_w16_s8_mean | 3720 | 0 | 0.5944 | 0.2368 | 0.0823 | 0.3348 | 1362.68 | 78M |
| Moment-DETR | moment_detr_raw_video | 3720 | 0 | 0.4145 | 0.2570 | 0.0995 | 0.2662 | 2049.43 | 3.0M |

Интерпретация: CLIP выше по `R@1 IoU 0.3` и `mIoU`. Moment-DETR выше по более
строгим `R@1 IoU 0.5` и `R@1 IoU 0.7`. CLIP быстрее в CPU experiment, но
сохраняет больший cache embeddings. Это воспроизводимое сравнение масштаба
курсовой работы, а не прямое SOTA comparison.

## CLIP vs Moment-DETR на 50 запросах

Источники:

- `results/clip_vs_moment_detr_50/*/run_config.json`
- `results/moment_detr_charades_50/metrics.json`

| Model | Configuration | Queries | R@1 IoU 0.3 | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| CLIP | clip_w8_s4_mean | 50 | 0.58 | 0.52 | 0.32 | 0.4393782459360136 | 14.223121250004624 |
| CLIP | clip_w16_s8_mean | 50 | 0.68 | 0.36 | 0.08 | 0.4007717686562073 | 2.116203374986071 |
| CLIP | clip_w32_s16_mean | 50 | 0.54 | 0.0 | 0.0 | 0.2854293692302304 | 2.0916127920063445 |
| CLIP | clip_w16_s8_max | 50 | 0.56 | 0.36 | 0.06 | 0.3374594801614168 | 2.0933764169894857 |
| Moment-DETR | raw-video compatible checkpoint | 50 | 0.44 | 0.32 | 0.04 | 0.26340237468832634 | 17.933695000014268 |
