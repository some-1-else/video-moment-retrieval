# Поиск временных фрагментов видео по текстовым запросам

## 1. Введение

Video Moment Retrieval - это задача локализации временного фрагмента видео,
соответствующего запросу на естественном языке. По видео и фразе вроде
`person turn a light on` система должна предсказать начало и конец релевантного
момента. Такая задача важна для поиска по видео, video question answering и
навигации по медиаконтенту, потому что пользователь обычно описывает событие
словами, а не номерами кадров.

Изначально в курсовой планировалось использовать QVHighlights как основной
датасет. QVHighlights связан с современной research line по moment retrieval и
используется в Moment-DETR. Однако raw videos в QVHighlights основаны на
YouTube clips, что привело к практическим проблемам воспроизводимости: часть
видео недоступна, удалена, закрыта или нестабильно скачивается локально. Для
курсового проекта это сделало full raw-video pipeline менее надежным.

Поэтому основной экспериментальный датасет был заменен на Charades-STA.
Charades-STA является стандартным temporal sentence grounding dataset,
построенным поверх Charades videos. В этом проекте Charades-STA используется
вместе с локально извлеченными Charades 480p videos. Это дает воспроизводимый
raw-video setup и сохраняет соответствие задаче Video Moment Retrieval.

Проект реализует и оценивает CLIP-based retrieval baseline. Baseline выбирает
кадры из видео, вычисляет image-text similarities через CLIP, преобразует
frame scores в temporal window scores и выбирает top temporal window. Также
реализованы temporal aggregation, optional similarity smoothing и embedding
cache для эффективных sweeps. Дополнительно проект содержит ограниченный
Moment-DETR raw-video feasibility probe, адаптированный из official external
repository. Moment-DETR comparison намеренно ограничен и не должен
интерпретироваться как full official benchmark.

## 2. Связанные работы

Temporal sentence grounding и Video Moment Retrieval изучают сопоставление
language query с временным сегментом в untrimmed video. Среди стандартных
датасетов важен Charades-STA, который содержит sentence annotations со start и
end timestamps поверх Charades videos. QVHighlights расширяет задачу в сторону
highlight-oriented setting и используется в Moment-DETR.

CLIP - vision-language model, обученная на image-text pairs. Хотя CLIP не
является temporal localization model, он дает сильный zero-shot image-text
similarity signal. Поэтому CLIP полезен как простой baseline для frame-level
video retrieval: sampled frames можно сравнить с query, а similarities
агрегировать по temporal windows.

Moment-DETR - transformer-based moment retrieval model, опубликованная вместе с
QVHighlights. Она специализированнее простого CLIP baseline, потому что
моделирует temporal proposals и relevance совместно. Однако official
dataset-level inference path ожидает pre-extracted QVHighlights-style features,
а raw-video demo path использует другой checkpoint и feature format. Поэтому в
этой курсовой Moment-DETR используется только как isolated raw-video probe, а
не как полноценное воспроизведение official benchmark.

## 3. Датасеты

### 3.1 QVHighlights как исходный план

QVHighlights был исходным target dataset, потому что он связан с Moment-DETR и
современной оценкой Video Moment Retrieval. В проекте сохранены QVHighlights
loader code, preliminary checks и ранние локальные эксперименты. Однако
QVHighlights raw videos зависят от YouTube availability, что сделало dataset
сложным для надежного воспроизведения в рамках курсовой.

QVHighlights code и preliminary results не удалены. Они остаются частью истории
проекта и могут быть использованы, если появится стабильная raw-video subset.
Тем не менее финальные экспериментальные результаты в этом тексте основаны на
Charades-STA.

### 3.2 Charades-STA как финальный экспериментальный датасет

Charades-STA был выбран как финальный экспериментальный датасет, потому что он
содержит temporal sentence grounding annotations поверх Charades videos, а raw
video archive можно подготовить локально. Локальная настройка проекта содержит:

- Charades 480p videos, извлеченные локально: 9,848 `.mp4` files в
  `data/raw/charades/videos/`.
- Charades-STA train annotations: 12,408 rows.
- Charades-STA test annotations: 3,720 rows.
- Annotation files в `data/charades_sta/`.

Charades-STA annotation format является line-based:

```text
video_id start_time end_time##sentence
```

Реализованный Charades-STA loader преобразует эти строки во внутренний
moment retrieval format:

```json
{
  "qid": "charades_sta_test_0",
  "vid": "3MSZA",
  "query": "person turn a light on.",
  "duration": 31.0,
  "relevant_windows": [[24.3, 30.4]],
  "source_dataset": "charades_sta"
}
```

Если duration не указана в annotations, она читается из metadata локального
video через OpenCV. Ground-truth windows, выходящие за decoded video duration,
обрезаются до valid interval `[0, duration]` в experiment runners, а количество
clipped windows логируется.

## 4. Методика

### 4.1 Реализованный CLIP-based retrieval

Основной метод проекта - CLIP-based frame и temporal-window retrieval baseline.
Модель не обучается и не fine-tuned. Используется pretrained CLIP ViT-B/32 для
вычисления visual и textual embeddings.

Для каждой пары video-query pipeline:

1. Samples video frames с фиксированной частотой; в reported experiments
   используется `fps=1`.
2. Кодирует sampled frames через CLIP image encoder.
3. Кодирует query через CLIP text encoder.
4. Вычисляет frame-level image-text similarity scores.
5. Опционально сглаживает similarity sequence.
6. Генерирует temporal candidate windows.
7. Агрегирует frame scores внутри каждого candidate window.
8. Выбирает highest-scoring window.
9. Оценивает prediction относительно ground-truth temporal window.

Retrieval и evaluation pipeline реализованы внутри проекта. Сохраненные
experiment results находятся в `results/`.

### 4.2 Temporal windows

Baseline преобразует frame-level similarities в temporal moment predictions
через fixed-size candidate windows. Каждая configuration определяется window
size и stride в секундах. Например, `w16/s8` означает 16-секундное window с
8-секундным stride.

Такой design позволяет контролируемо изучать компромисс между coarse и strict
localization. Shorter windows могут точнее локализовать actions, а longer
windows захватывают более широкий context и могут улучшать coarse overlap при
низких IoU thresholds.

### 4.3 Aggregation

Оцениваются две aggregation strategies:

- Mean aggregation: среднее CLIP frame similarities внутри window.
- Max aggregation: максимальная frame similarity внутри window.

Mean aggregation ожидаемо стабильнее, когда несколько frames в moment умеренно
релевантны. Max aggregation может помочь, если один frame особенно
диагностичен, но также может быть чувствительнее к noisy frame-level matches.

### 4.4 Similarity smoothing

Smoothing experiment применяет moving average к frame-level similarity scores
перед temporal window aggregation. Проверяются варианты:

- no smoothing;
- moving average с window size 3;
- moving average с window size 5.

Smoothing реализован как изолированное расширение CLIP retrieval runner. Он не
меняет общую evaluation logic. Цель - проверить, улучшает ли снижение
frame-score noise temporal localization на той же фиксированной 1,000-query
Charades-STA subset.

### 4.5 Embedding cache

CLIP image encoder - самая дорогая часть baseline. Поэтому проект использует
per-video embedding cache. Cache key зависит от video identifier, CLIP model
name и sampling FPS. Text embeddings считаются per query, а video image
embeddings переиспользуются между разными window sizes, strides, aggregation
methods и smoothing variants.

Cache особенно важен для controlled sweeps. В 1,000-query Charades-STA sweep
первая configuration заполнила cache и заняла 309.6866 seconds. Последующие
cached configurations заняли примерно 21-23 seconds в main sweep, а smoothing
runs переиспользовали тот же cache с 1,000 cache hits и 0 misses.

### 4.6 Moment-DETR raw-video probe

Moment-DETR не реализовывался с нуля. Проект использовал official external
repository в `external/moment_detr` и raw-video-compatible checkpoint:

```text
external/moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt
```

Изолированная wrapper-логика адаптирует official raw-video model path к
локальным Charades videos и оценивает top-1 predicted windows теми же метриками,
что и CLIP baseline. Эта wrapper-логика не является частью основной CLIP
pipeline.

Official QVHighlights release checkpoint также проверялся, но оказался
несовместим с raw-video path. Release checkpoint ожидает feature dimension
2818, а raw-video path создает CLIP image features плюс temporal endpoint
features, то есть `512 + 2 = 514`. Это описано в `docs/moment_detr.md`.

## 5. Экспериментальная настройка

### 5.1 Hardware и device

Все reported experiments в этом тексте используют CPU inference. GPU
acceleration не предполагается. Поэтому runtime numbers полезны для локальной
воспроизводимости, но не должны интерпретироваться как optimized inference
throughput.

### 5.2 Фиксированная 1,000-query Charades-STA subset

Основные CLIP experiments используют фиксированную 1,000-query subset из
Charades-STA test split. Subset содержит 363 unique videos. Все CLIP sweep и
smoothing configurations используют один и тот же selection manifest, поэтому
различия в reported metrics связаны с retrieval configuration, а не с
различиями data selection.

Main 1,000-query sweep сохранен здесь:

```text
results/charades_sta_sweep_1000/summary.csv
```

Smoothing experiment сохранен здесь:

```text
results/charades_sta_smoothing_1000/summary.csv
```

50-query CLIP vs Moment-DETR comparison сохранен здесь:

```text
results/clip_vs_moment_detr_50/comparison_summary.csv
results/moment_detr_charades_50/metrics.json
```

### 5.3 Метрики

Evaluation использует стандартные temporal localization metrics:

- `R@1 IoU 0.3`: доля queries, где top predicted window пересекается с
  ground-truth window с IoU не ниже 0.3.
- `R@1 IoU 0.5`: то же при IoU threshold 0.5.
- `R@1 IoU 0.7`: то же при IoU threshold 0.7.
- `mIoU`: mean temporal IoU между top prediction и ground-truth window.

Все методы оцениваются по top-1 temporal predictions.

### 5.4 CLIP configurations

Main 1,000-query CLIP sweep оценивает:

- `window_size=8`, `stride=4`, `aggregation=mean`;
- `window_size=16`, `stride=8`, `aggregation=mean`;
- `window_size=32`, `stride=16`, `aggregation=mean`;
- `window_size=16`, `stride=8`, `aggregation=max`.

Все configurations используют:

- `fps=1`;
- CLIP `ViT-B/32`;
- CPU;
- shared embedding cache.

Smoothing experiment оценивает две base configurations:

- `w8/s4/mean`;
- `w16/s8/mean`;

с smoothing variants `none`, `moving_average_3` и `moving_average_5`.

## 6. Результаты

### 6.1 CLIP 1,000-query sweep

Следующая таблица показывает main controlled sweep на фиксированной 1,000-query
Charades-STA test subset. Source file:
`results/charades_sta_sweep_1000/summary.csv`.

| Configuration | Window | Stride | Aggregation | Queries | Unique Videos | Frames | R@1 IoU 0.3 | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU | Time (s) | Cache Hits | Cache Misses |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP w8/s4/mean | 8 | 4 | mean | 1000 | 363 | 29994 | 0.509 | 0.393 | 0.181 | 0.3478 | 309.69 | 637 | 363 |
| CLIP w16/s8/mean | 16 | 8 | mean | 1000 | 363 | 29994 | 0.625 | 0.260 | 0.096 | 0.3497 | 22.59 | 1000 | 0 |
| CLIP w32/s16/mean | 32 | 16 | mean | 1000 | 363 | 29994 | 0.395 | 0.014 | 0.000 | 0.2814 | 21.51 | 1000 | 0 |
| CLIP w16/s8/max | 16 | 8 | max | 1000 | 363 | 29994 | 0.602 | 0.232 | 0.077 | 0.3386 | 20.92 | 1000 | 0 |

Лучший coarse retrieval result достигается `w16/s8/mean`, где R@1 IoU 0.3
равен 0.625. Лучшая strict localization достигается `w8/s4/mean`, где R@1 IoU
0.5 равен 0.393, а R@1 IoU 0.7 равен 0.181. Значения mIoU у `w16/s8/mean` и
`w8/s4/mean` близки: 0.3497 и 0.3478.

### 6.2 Smoothing experiment

Smoothing experiment использует ту же fixed 1,000-query subset и переиспользует
embedding cache из main sweep. Source file:
`results/charades_sta_smoothing_1000/summary.csv`.

| Configuration | Smoothing | R@1 IoU 0.3 | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU | Time (s) | Cache Hits | Cache Misses |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP w8/s4/mean | none | 0.509 | 0.393 | 0.181 | 0.3478 | 17.30 | 1000 | 0 |
| CLIP w8/s4/mean | moving_average_3 | 0.511 | 0.398 | 0.183 | 0.3485 | 15.72 | 1000 | 0 |
| CLIP w8/s4/mean | moving_average_5 | 0.509 | 0.398 | 0.190 | 0.3494 | 15.79 | 1000 | 0 |
| CLIP w16/s8/mean | none | 0.625 | 0.260 | 0.096 | 0.3497 | 14.79 | 1000 | 0 |
| CLIP w16/s8/mean | moving_average_3 | 0.620 | 0.257 | 0.098 | 0.3491 | 15.37 | 1000 | 0 |
| CLIP w16/s8/mean | moving_average_5 | 0.623 | 0.254 | 0.098 | 0.3481 | 16.00 | 1000 | 0 |

Smoothing немного улучшает strict localization metrics для короткой
configuration `w8/s4/mean`. Вариант `moving_average_5` достигает R@1 IoU 0.7 =
0.190 и mIoU = 0.3494. Для `w16/s8/mean` smoothing не улучшает main coarse
metric или mIoU; unsmoothed variant остается strongest по R@1 IoU 0.3 и mIoU.

### 6.3 CLIP vs Moment-DETR на одной 50-query subset

CLIP vs Moment-DETR comparison использует одни и те же 50 Charades-STA
query-video pairs из:

```text
data/processed/charades_sta_moment_detr_test_subset.jsonl
```

Moment-DETR result является raw-video feasibility probe с compatible raw-video
checkpoint из external repository. Это не full official Moment-DETR benchmark и
не fine-tuned Charades-STA model. Source files:
`results/clip_vs_moment_detr_50/comparison_summary.csv` и
`results/moment_detr_charades_50/metrics.json`.

| Model | Configuration | Queries | Failed | R@1 IoU 0.3 | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP | w8/s4/mean | 50 | 0 | 0.580 | 0.520 | 0.320 | 0.4394 | 14.22 |
| CLIP | w16/s8/mean | 50 | 0 | 0.680 | 0.360 | 0.080 | 0.4008 | 2.12 |
| CLIP | w32/s16/mean | 50 | 0 | 0.540 | 0.000 | 0.000 | 0.2854 | 2.09 |
| CLIP | w16/s8/max | 50 | 0 | 0.560 | 0.360 | 0.060 | 0.3375 | 2.09 |
| Moment-DETR | raw-video checkpoint | 50 | 0 | 0.440 | 0.320 | 0.040 | 0.2634 | 17.93 |

На этой ограниченной subset простой CLIP baseline показывает сильные результаты.
`w8/s4/mean` дает лучшие strict localization metrics, а `w16/s8/mean` дает
лучший coarse metric при IoU 0.3. Moment-DETR успешно строит predictions для
всех 50 examples, что подтверждает raw-video feasibility, но comparison слишком
мал и слишком зависит от setup, чтобы делать широкие выводы о превосходстве
моделей.

## 7. Обсуждение

Эксперименты показывают устойчивый компромисс между coarse retrieval и strict
localization. Medium 16-second windows лучше работают при IoU 0.3, где
достаточно approximate overlap. Shorter 8-second windows лучше работают при
строгих IoU thresholds, потому что они точнее align-ятся с Charades-STA moments.
Large 32-second windows теряют temporal precision и слабо работают при IoU 0.5
и 0.7.

Mean aggregation стабильнее max aggregation в проверенных settings. Для той же
`w16/s8` temporal setup mean aggregation улучшает все reported metrics по
сравнению с max aggregation в 1,000-query sweep. Это указывает, что average
visual-language relevance внутри window является более надежным сигналом, чем
один maximum-scoring frame.

Similarity smoothing дает умеренный эффект. Он немного улучшает strict
localization для короткого `w8/s4/mean`, особенно при IoU 0.7. Однако smoothing
не улучшает stronger coarse-retrieval configuration `w16/s8/mean`. Значит,
smoothing может быть полезен для коротких candidate windows, чувствительных к
local frame-score noise, но не является универсальным улучшением.

Embedding cache существенно важен для практических экспериментов. Без cache
каждая configuration заново запускала бы CLIP image encoder на тех же videos. С
cache дорогой video encoding step переиспользуется, что делает controlled
sweeps возможными на CPU.

Moment-DETR probe показывает, что интеграция specialized temporal localization
model технически возможна, но чувствительна к checkpoint и feature format.
Raw-video-compatible checkpoint работает с локальными Charades videos. Released
QVHighlights feature checkpoint несовместим с raw-video path из-за feature
dimension mismatch. Поэтому rigorous Moment-DETR benchmark потребовал бы более
аккуратного environment и feature setup, чем limited probe в этой курсовой.

## 8. Ограничения

Основной CLIP experiment использует fixed 1,000-query subset, а не полный
Charades-STA test split. Subset достаточно велика для контролируемого
coursework experiment, но ее нельзя подавать как full official Charades-STA
benchmark.

Все reported runs выполнены на CPU. Runtime может существенно отличаться на GPU
или при optimized batching.

CLIP baseline использует pretrained CLIP only. Модель не обучается и не
fine-tuned на Charades-STA. Метод также не моделирует temporal action dynamics
явно, кроме fixed windows, aggregation и optional smoothing.

Moment-DETR comparison использует только 50 examples и raw-video-compatible
checkpoint из external repository. Это preliminary feasibility comparison, а не
full official Moment-DETR evaluation. Full official benchmarking, feature
extraction и fine-tuning находятся вне scope этой курсовой.

QVHighlights остается preliminary направлением, потому что raw video
availability усложнила построение воспроизводимой локальной experiment pipeline
в доступное время.

## 9. Заключение

В курсовой реализована воспроизводимая raw-video Video Moment Retrieval
pipeline и проведена оценка на Charades-STA. Финальный dataset setup содержит
локальные Charades videos и standard Charades-STA annotations, что позволяет
избежать проблем raw video availability, возникших с QVHighlights.

Основной реализованный метод - CLIP-based temporal-window retrieval baseline.
На фиксированной 1,000-query Charades-STA subset `w16/s8/mean` достигает
лучшего coarse retrieval result с R@1 IoU 0.3 = 0.625, а `w8/s4/mean` достигает
лучших strict localization results с R@1 IoU 0.5 = 0.393 и R@1 IoU 0.7 =
0.181. Smoothing дает небольшие улучшения для short-window setup, но не
улучшает все configurations consistently.

Проект также демонстрирует isolated Moment-DETR raw-video probe на 50
Charades-STA examples. Это подтверждает, что specialized temporal model можно
подключить к локальным данным, но full official benchmark и fine-tuning
остаются за пределами scope.

В результате проект дает понятный baseline, reusable data preparation workflow,
controlled experiment outputs и практическую основу для будущей работы с
trained temporal localization models.

## 10. Источники и материалы проекта

Базовые источники проекта:

- Radford et al. представили CLIP как contrastive image-text pretraining model.
- Lei et al. представили Moment-DETR и QVHighlights benchmark для moment
  retrieval и highlight detection.
- Sigurdsson et al. представили Charades video dataset.
- Gao et al. представили Charades-STA для temporal activity localization via
  language query.

BibTeX entries собраны в `thesis/references.bib`.

Project evidence и result files, использованные в тексте:

- `docs/moment_detr.md`
- `results/charades_sta_sweep_1000/summary.csv`
- `results/charades_sta_smoothing_1000/summary.csv`
- `results/clip_vs_moment_detr_50/comparison_summary.csv`
- `results/moment_detr_charades_50/metrics.json`

Промежуточные writing notes перенесены в local ignored agent memory и не входят
в финальную структуру coursework repository.
