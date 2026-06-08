# CLIP baseline

Основной baseline курсовой использует pretrained CLIP как frozen image-text
encoder для raw-video moment retrieval на Charades-STA. Параметры CLIP не
обучаются и не fine-tuned.

## Pipeline

Baseline следует такой схеме:

```text
video + text query
-> sample frames at fixed FPS
-> encode frames with CLIP image encoder
-> encode query with CLIP text encoder
-> compute frame-level cosine similarity
-> aggregate scores over temporal windows
-> select the best window
-> evaluate with R@1 IoU 0.3/0.5/0.7 and mIoU
```

Основная логика находится в `src/`, а финальные experiment entrypoints
доступны через `scripts/run_clip_sweep.py` и
`scripts/run_smoothing_sweep.py`.

## CLIP encoder

Encoder layer отвечает за:

- загрузку pretrained CLIP model;
- кодирование text query в normalized text embedding;
- кодирование decoded video frames в normalized image embeddings;
- вычисление cosine similarity между text и image embeddings.

Основные зависимости:

```text
torch
torchvision
ftfy
regex
tqdm
git+https://github.com/openai/CLIP.git
```

`load_clip_model(..., device="auto")` выбирает `cuda`, затем `mps`, затем
`cpu` в зависимости от локальной доступности. Reported coursework runs
выполнены на CPU.

## Frame extraction

Кадры извлекаются из локальных видео по фиксированным timestamps. Детали
описаны в `docs/video_frame_extraction.md`. В финальных экспериментах
используется `fps=1`.

## Temporal windows

Frame scores агрегируются в candidate temporal windows. Main CLIP sweep
проверяет контролируемые комбинации:

- window size;
- stride;
- aggregation method.

Финальный 1,000-query Charades-STA sweep оценивает 8-, 16- и 32-секундные
windows с `mean` или `max` aggregation.

## Scoring и aggregation

Для каждой пары query-video:

1. CLIP один раз кодирует query.
2. CLIP image embeddings создаются или загружаются из cache для sampled frames.
3. Cosine similarity дает frame-level relevance score.
4. Scores внутри каждого candidate temporal window агрегируются через `mean`
   или `max`.
5. Window с максимальным score используется как top-1 prediction.

Результаты курсовой показывают, что `mean` aggregation стабильнее `max`
aggregation в проверенных настройках Charades-STA.

## Smoothing

Optional smoothing применяет moving average к frame-level similarity scores
перед window aggregation. Smoothing sweep сравнивает `none`,
`moving_average_3` и `moving_average_5` для выбранных CLIP configurations.

Smoothing немного улучшает strict localization для короткой конфигурации
`w8/s4/mean`, но не улучшает все метрики во всех настройках.

## Embedding cache

Image embedding extraction - самый дорогой повторяющийся шаг. Public CLIP
scripts могут переиспользовать cached image embeddings между parameter sweeps,
так что window size, stride, aggregation и smoothing можно менять без повторного
кодирования тех же video frames.

Embeddings и cache files являются local artifacts и не входят в финальную
public results surface.

## Scope

Baseline намеренно простой:

- нет model training;
- нет fine-tuning на Charades-STA;
- нет full QVHighlights benchmark;
- нет learned temporal dynamics за пределами fixed windows, aggregation и
  optional smoothing.
