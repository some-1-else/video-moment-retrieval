# Moment-DETR

Этот документ кратко описывает, что было сделано с Moment-DETR в курсовой и
какие ограничения были обнаружены. Moment-DETR используется как внешний
raw-video comparison: сначала как 50-query feasibility probe, затем как
full-test Charades-STA evaluation.

## Роль в проекте

Основная линия курсовой - Charades-STA + CLIP. Moment-DETR добавлен как
raw-video comparison, чтобы проверить, можно ли подключить специализированную
temporal localization model к тем же локальным Charades videos и оценить ее
теми же метриками.

Moment-DETR model не обучалась, не fine-tuned и не интегрировалась в основную
CLIP pipeline.

## Official repository

Официальный репозиторий:

```text
https://github.com/jayleicn/moment_detr
```

Он был клонирован в:

```text
external/moment_detr
```

Observed commit для локального probe:

```text
b7e553a
```

Официальный проект в первую очередь ориентирован на QVHighlights и содержит два
релевантных inference routes:

- dataset-level QVHighlights inference с pre-extracted features;
- raw-video prediction через demo `run_on_video`.

В курсовой используется raw-video route, потому что данные проекта - локальные
Charades `.mp4` files.

## Ограничения окружения

Официальный README Moment-DETR рекомендует отдельное окружение с Python 3.7 и
PyTorch 1.9.0. Текущий coursework `.venv` новее, поэтому integration сохраняет
Moment-DETR изолированным и не downgrade-ит основное окружение проекта.

Raw-video demo также зависит от packages вроде `ffmpeg-python`, `easydict` и
`tensorboard`. Изолированная wrapper-логика не меняет основное окружение и
использует OpenCV для frame extraction.

## Checkpoints

Проверялись два типа checkpoints:

- official release QVHighlights checkpoints, включая
  `ft_model_from_pt_model_e50.ckpt`;
- raw-video-compatible checkpoint из official repo:
  `external/moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt`.

Raw-video-compatible checkpoint работает для локального comparison. Released
QVHighlights feature checkpoint не совместим напрямую с raw-video path: он
ожидает feature dimension `2818`, а raw-video path создает CLIP image features
плюс temporal endpoint features, то есть `512 + 2 = 514`.

## Конвертация Charades-STA

Для локального comparison Charades-STA annotations были конвертированы в
QVHighlights-like JSONL format:

```json
{
  "qid": "charades_sta_test_0",
  "query": "person turn a light on.",
  "duration": 31.0,
  "vid": "3MSZA",
  "relevant_windows": [[24.3, 30.4]]
}
```

Historical fixed 50-query subset хранится здесь:

```text
data/processed/charades_sta_moment_detr_test_subset.jsonl
```

Full-test JSONL хранится здесь:

```text
data/processed/charades_sta_full_test_moment_detr.jsonl
```

Конвертация проверяет, что локальные videos существуют, открываются OpenCV и
имеют readable frames.

## One-video raw-video probe

Первый successful raw-video probe использовал:

```text
video: data/raw/charades/videos/3MSZA.mp4
query: person turn a light on.
checkpoint: external/moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt
output: .agent_memory/results/probes/moment_detr_probe/3MSZA_prediction.json
```

Top predicted window:

```json
[26.0624, 31.9724, 0.9685]
```

Это подтвердило, что Moment-DETR raw-video inference может построить temporal
prediction для локальной пары Charades video/query.

## 50-query feasibility probe

Historical probe использует одни и те же 50 Charades-STA examples для CLIP и
Moment-DETR. Results сохранены здесь:

```text
results/clip_vs_moment_detr_50/comparison_summary.csv
results/moment_detr_charades_50/metrics.json
```

Это feasibility probe. Он подтвердил raw-video path перед full-test запуском.

## Full-test comparison

Final coursework comparison использует полный Charades-STA test split:
3,720 queries / 1,334 videos. Results сохранены здесь:

```text
results/charades_sta_full_test_clip/clip_w16_s8_mean/metrics.json
results/moment_detr_charades_full_test/metrics.json
results/clip_vs_moment_detr_full_test/comparison_summary.csv
```

Это coursework-scale reproducible experiment. Это не full official
QVHighlights benchmark и не обученный Charades-STA Moment-DETR result.

## Ограничения

- Official dataset-level path ожидает QVHighlights-style pre-extracted features
  и не запускается напрямую на Charades-STA videos.
- Raw-video-compatible checkpoint usable, но released QVHighlights feature
  checkpoint не совместим с raw-video feature dimensions.
- Historical probe использовал 50 examples; final comparison использует полный
  Charades-STA test split.
- CPU inference медленный, если video encoding повторяется query by query.
- Rigorous Moment-DETR benchmark потребовал бы стабильного отдельного
  окружения, аккуратного feature setup, batching by video и, вероятно,
  fine-tuning или dataset-specific adaptation.

Вывод: Moment-DETR feasible как raw-video comparison на локальных Charades
videos. Финальный результат курсовой остается coursework-scale comparison на
Charades-STA, а не official QVHighlights/SOTA benchmark.
