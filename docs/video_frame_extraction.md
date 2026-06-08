# Извлечение кадров из видео

Frame extraction связывает локальные video files и CLIP-based baseline. Кадры
выбираются из видео по фиксированным timestamps, кодируются CLIP image encoder
и сравниваются с text query embedding.

## Текущий scope

Текущая реализация поддерживает extraction из локального video file. Она не
скачивает QVHighlights videos, не обрабатывает полный dataset и сама по себе не
запускает CLIP.

Этот слой является небольшим smoke-testable component:

```text
local video file -> sampled timestamps -> decoded frames
```

Модуль `src/video/frame_sampling.py` генерирует timestamps по `duration` и
`fps`. Frame extraction использует эти timestamps, чтобы читать frames из
локального video.

## Зависимость

Для чтения видео используется `opencv-python`:

```text
opencv-python
```

OpenCV импортируется lazy внутри frame extraction functions, поэтому большинству
tests и non-video workflows не требуется video decoding.

## Optional local smoke test

Если локальный video file доступен, можно проверить extraction через public
pipeline scripts или через unit tests для `src/video/`. Финальные notebooks не
запускают video decoding: они читают уже сохраненные results.

## Timestamp behavior

`extract_frames_at_timestamps(video_path, timestamps)` ожидает timestamps в
секундах. Negative timestamps отклоняются. Timestamps, которые больше или равны
оцененной video duration, вызывают `ValueError`.

Это сделано намеренно: silent skipping out-of-range timestamps может скрывать
ошибки в sampling logic.

## Ограничения

- QVHighlights videos не скачиваются этим project step.
- Dataset-level video processing не является частью этого документа.
- Разные video codecs могут вести себя по-разному в зависимости от локальной
  поддержки OpenCV/FFmpeg.
- Некоторым системам может потребоваться дополнительная FFmpeg support.
- Большие videos могут медленно seek-аться frame-by-frame.
- Этот модуль только подготавливает frames; он не вычисляет CLIP embeddings.
