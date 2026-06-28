# Moment-DETR

Документ кратко описывает использование Moment-DETR в курсовой. Moment-DETR
применяется как внешний raw-video comparison: сначала на выборке из 50
запросов, затем на полном test split Charades-STA.

## Роль в проекте

Основная линия курсовой - Charades-STA + CLIP. Moment-DETR добавлен для
сравнения с моделью, специально предназначенной для temporal localization.
Оценка выполняется на тех же локальных Charades videos и теми же метриками.

Moment-DETR model не обучалась, не fine-tuned и не включалась в основной
CLIP pipeline.

## Официальный репозиторий

Официальный репозиторий:

```text
https://github.com/jayleicn/moment_detr
```

Для локального запуска ожидается путь:

```text
external/moment_detr
```

Официальный проект в первую очередь ориентирован на QVHighlights и содержит два
релевантных inference routes:

- dataset-level QVHighlights inference с pre-extracted features;
- raw-video prediction через demo `run_on_video`.

В курсовой используется raw-video route, потому что данные проекта - локальные
Charades `.mp4` files.

## Ограничения окружения

Официальный README Moment-DETR рекомендует отдельное окружение с Python 3.7 и
PyTorch 1.9.0. Основное окружение проекта новее, поэтому Moment-DETR
предполагается запускать изолированно, без downgrade зависимостей основного
проекта.

Raw-video demo также зависит от packages вроде `ffmpeg-python`, `easydict` и
`tensorboard`. Wrapper-логика проекта не меняет основное окружение и использует
OpenCV для frame extraction.

## Checkpoints

Проверялись два типа checkpoints:

- official release QVHighlights checkpoints, включая
  `ft_model_from_pt_model_e50.ckpt`;
- raw-video-compatible checkpoint из official repo:
  `external/moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt`.

Raw-video-compatible checkpoint работает для локального comparison. Released
QVHighlights feature checkpoint не совместим напрямую с raw-video path: он
ожидает feature dimension `2818`, а raw-video path создаёт CLIP image features
плюс temporal endpoint features, то есть `512 + 2 = 514`.

## Конвертация Charades-STA

Для comparison Charades-STA annotations конвертируются в QVHighlights-like JSONL
format:

```json
{
  "qid": "charades_sta_test_0",
  "query": "person turn a light on.",
  "duration": 31.0,
  "vid": "3MSZA",
  "relevant_windows": [[24.3, 30.4]]
}
```

Full-test JSONL хранится здесь:

```text
data/processed/charades_sta_full_test_moment_detr.jsonl
```

Конвертация проверяет, что локальные videos существуют, открываются OpenCV и
имеют readable frames.

## Сравнение на 50 запросах

Первый контрольный запуск использует одни и те же 50 Charades-STA examples для
CLIP и Moment-DETR. Results сохранены здесь:

```text
results/clip_vs_moment_detr_50/comparison_summary.csv
results/moment_detr_charades_50/metrics.json
```

Этот запуск подтвердил работоспособность raw-video path перед полным
экспериментом.

## Full-test comparison

Финальное сравнение использует полный Charades-STA test split:
3,720 queries / 1,334 videos. Results сохранены здесь:

```text
results/charades_sta_full_test_clip/clip_w16_s8_mean/metrics.json
results/moment_detr_charades_full_test/metrics.json
results/clip_vs_moment_detr_full_test/comparison_summary.csv
```

Это воспроизводимый эксперимент масштаба курсовой работы. Он не является
официальным QVHighlights benchmark и не использует Moment-DETR, обученный
специально на Charades-STA.

## Ограничения

- Official dataset-level path ожидает QVHighlights-style pre-extracted features
  и не запускается напрямую на Charades-STA videos.
- Raw-video-compatible checkpoint usable, но released QVHighlights feature
  checkpoint не совместим с raw-video feature dimensions.
- CPU inference медленный, если video encoding повторяется query by query.
- Более строгий Moment-DETR benchmark потребовал бы отдельного окружения,
  аккуратного feature setup, batching by video и, вероятно, fine-tuning или
  dataset-specific adaptation.

Вывод: Moment-DETR можно использовать как raw-video comparison на локальных
Charades videos. Финальный результат курсовой остаётся сравнением на
Charades-STA, а не official QVHighlights/SOTA benchmark.
