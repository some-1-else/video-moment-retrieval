# Ограничения QVHighlights

QVHighlights был исходным dataset direction для проекта по Video Moment
Retrieval, потому что это современный benchmark с natural-language queries,
temporal windows и прямой связью с official Moment-DETR release.

Финальные coursework experiments были перенесены на Charades-STA, потому что
доступность QVHighlights raw videos затруднила воспроизводимый локальный
benchmark в рамках проекта.

## Почему рассматривался QVHighlights

QVHighlights предоставляет JSONL annotations с полями вроде:

```text
qid
query
duration
vid
relevant_clip_ids
saliency_scores
relevant_windows
```

Локальный loader отображает эти поля в project `MomentRetrievalSample` format.
Annotation-only checks подтвердили, что validation annotations читаются и
конвертируются в тот же retrieval/evaluation interface, который используется в
остальном проекте.

## Что удалось сделать

Локально были проверены следующие exploratory pieces:

- QVHighlights annotation parsing;
- небольшие annotation-only rehearsal runs;
- subset manifest creation для selected validation samples;
- parsing QVHighlights `video_id` values в YouTube id и clip timestamps;
- ограниченный CLIP retrieval на videos, которые уже были доступны и readable
  локально;
- embedding cache checks на tiny readable subset.

Эти эксперименты сохранены как research history в ignored `.agent_memory/`
paths. Они не входят в public final results.

## Почему QVHighlights оказался сложным для финальной оценки

QVHighlights videos связаны с внешними web-hosted sources. Локальная
воспроизводимость зависит от того, доступны ли соответствующие YouTube videos,
можно ли их скачать, нет ли regional restrictions, и декодируются ли они
локальным OpenCV/FFmpeg setup.

Наблюдавшиеся ограничения:

- многие selected videos отсутствовали локально;
- часть source videos была недоступна или заблокирована;
- некоторые downloaded files существовали на диске, но не читались OpenCV;
- full-dataset video download был бы большим, медленным и хрупким;
- official Moment-DETR dataset-level inference ожидает pre-extracted
  QVHighlights features, а не произвольные локальные raw videos.

Из-за этого небольшие QVHighlights experiments могли проверить pipeline, но не
могли служить надежным public benchmark claim.

## Preliminary results не являются основным benchmark

Exploratory QVHighlights runs использовали tiny locally available subsets,
включая настройки с 2 videos и 9 readable videos. Эти runs были полезны для
проверки data loading, video availability checks, CLIP scoring и embedding
cache behavior, но они не репрезентативны для full QVHighlights validation
split.

Поэтому:

- QVHighlights numbers не должны подаваться как final benchmark performance;
- QVHighlights outputs сохранены как local research history;
- public `results/` folder намеренно содержит только финальные Charades-STA
  results для курсовой.

## Почему финальные эксперименты выполнены на Charades-STA

Charades-STA выбран для финальной coursework line, потому что дает более ясный
local reproduction path:

- annotation format является простым text format;
- local videos можно подготовить из Charades archive;
- videos достаточно короткие для CPU CLIP experiments;
- fixed subsets можно проверить и переиспользовать между sweeps;
- CLIP и Moment-DETR probe results можно оценивать одними temporal localization
  metrics.

Такой переход удерживает проект на воспроизводимых Video Moment Retrieval
experiments, а не на хрупком external-video acquisition.

## Текущий статус

QVHighlights support остается в `src/data/qvhighlights.py`, потому что он
полезен для сохранения общего moment-retrieval sample format и для будущей
работы. Однако QVHighlights не является финальным benchmark path этой курсовой.
