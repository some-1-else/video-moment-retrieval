# Ограничения QVHighlights

QVHighlights рассматривался как возможный dataset для проекта по Video Moment
Retrieval, потому что это современный benchmark с natural-language queries,
temporal windows и связью с официальным release Moment-DETR.

Финальные эксперименты курсовой выполнены на Charades-STA, так как доступность
raw videos QVHighlights затрудняет воспроизводимый локальный benchmark.

## Почему рассматривался QVHighlights

QVHighlights предоставляет JSONL annotations с полями:

```text
qid
query
duration
vid
relevant_clip_ids
saliency_scores
relevant_windows
```

Локальный loader отображает эти поля в общий project format
`MomentRetrievalSample`. Это позволяет использовать один интерфейс для loading,
retrieval и evaluation.

## Что было проверено

В проекте проверялись:

- QVHighlights annotation parsing;
- чтение validation annotations;
- создание subset manifests для выбранных validation samples;
- parsing QVHighlights `video_id` values в YouTube id и clip timestamps;
- ограниченный CLIP retrieval на локально доступных videos;
- embedding cache behavior на небольшой читаемой выборке.

Эти проверки использовались только для валидации pipeline и не входят в
финальную линию результатов курсовой.

## Почему QVHighlights не выбран для финальной оценки

QVHighlights videos связаны с внешними web-hosted sources. Локальная
воспроизводимость зависит от того, доступны ли соответствующие YouTube videos,
можно ли их скачать, нет ли regional restrictions, и декодируются ли они
локальным OpenCV/FFmpeg setup.

Основные ограничения:

- многие selected videos отсутствовали локально;
- часть source videos была недоступна или заблокирована;
- некоторые downloaded files существовали на диске, но не читались OpenCV;
- full-dataset video download был бы большим, медленным и хрупким;
- official Moment-DETR dataset-level inference ожидает pre-extracted
  QVHighlights features, а не произвольные локальные raw videos.

Из-за этого небольшие QVHighlights experiments могли проверить pipeline, но не
могли служить надёжным benchmark claim.

## Статус результатов QVHighlights

Небольшие QVHighlights runs были полезны для проверки data loading, video
availability checks, CLIP scoring и embedding cache behavior, но они не
репрезентативны для full QVHighlights validation split.

Поэтому:

- QVHighlights numbers не используются как final benchmark performance;
- `results/` содержит финальные Charades-STA results;
- QVHighlights support сохраняется только как часть общего data loading слоя.

## Почему финальные эксперименты выполнены на Charades-STA

Charades-STA выбран для финальной coursework line, потому что даёт более ясный
local reproduction path:

- annotation format является простым text format;
- local videos можно подготовить из Charades archive;
- videos достаточно короткие для CPU CLIP experiments;
- fixed subsets можно проверить и переиспользовать между sweeps;
- CLIP и Moment-DETR comparison можно оценивать одними temporal localization
  metrics.

Такой выбор удерживает проект в воспроизводимой постановке Video Moment
Retrieval и не зависит от нестабильного скачивания внешних видео.
