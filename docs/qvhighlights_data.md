# QVHighlights Annotation Data

`QVHighlights` будет основным датасетом для экспериментов по Video Moment Retrieval.

На текущем этапе проекту нужны только annotation files: текстовые запросы, `video_id`, длительность видео и размеченные релевантные временные окна. Видео, кадры, CLIP embeddings и скачивание полного датасета пока не подключаются.

## Local Layout

Annotation-файлы нужно положить локально в:

```text
data/qvhighlights/
```

Ожидаемый вариант структуры:

```text
data/qvhighlights/
  train.jsonl
  val.jsonl
  test.jsonl
```

Если после ручного скачивания реальные файлы называются иначе, можно оставить их реальные имена и передавать нужный путь через `--annotations`.

## Manual annotation setup

На этом этапе нужны только annotation JSONL files QVHighlights. Видео, кадры и признаки пока не нужны.

После ручного скачивания annotations рекомендуется положить файлы так:

```text
data/qvhighlights/
  highlight_train_release.jsonl
  highlight_val_release.jsonl
  highlight_test_release.jsonl
```

Если реальные файлы называются иначе, например `train.jsonl`, `val.jsonl` или имеют другой release suffix, это нормально. Главное - передавать фактический путь к файлу через `--annotations`.

Проверить, что validation annotations читаются текущим loader-ом:

```bash
.venv/bin/python scripts/inspect_annotations.py \
  --annotations data/qvhighlights/highlight_val_release.jsonl \
  --limit 5
```

Запустить dummy retrieval на первых 100 validation samples:

```bash
.venv/bin/python scripts/run_dummy_retrieval.py \
  --annotations data/qvhighlights/highlight_val_release.jsonl \
  --window-size 8 \
  --stride 4 \
  --limit 100 \
  --output results/dummy_qvhighlights_val_100.json
```

## Annotation Rehearsal Experiments

После ручного добавления `data/qvhighlights/highlight_val_release.jsonl` можно запустить лёгкую матрицу rehearsal experiments:

```bash
bash scripts/run_qvhighlights_annotation_experiments.sh
```

Скрипт сначала проверяет, что annotation-файл существует. Если файла нет, он завершится с сообщением:

```text
Missing annotation file: data/qvhighlights/highlight_val_release.jsonl
Please add QVHighlights annotations manually. See docs/qvhighlights_data.md
```

Если файл найден, скрипт запускает `inspect_annotations.py`, затем несколько вариантов `run_frame_score_retrieval.py` с `--limit 100`, и собирает результаты в:

```text
results/qvhighlights_annotation_rehearsal/summary.csv
```

Это rehearsal experiments без `CLIP` и без видео. Они проверяют loader, annotations format, retrieval/evaluation wiring и result collection на реальных annotation samples.

## Validation annotation check

Validation annotation file был скачан локально из официального Moment-DETR repository:

```text
https://github.com/jayleicn/moment_detr
```

Локальный путь:

```text
data/qvhighlights/highlight_val_release.jsonl
```

Файл содержит `1549` JSONL records. Реальные поля в annotation-файле:

```text
qid
query
duration
vid
relevant_clip_ids
saliency_scores
relevant_windows
```

Текущий loader использует эти поля так:

```text
qid -> sample_id
vid -> video_id
query -> query
duration -> duration
relevant_windows -> relevant_windows
```

Поля `relevant_clip_ids` и `saliency_scores` пока не используются в текущем baseline skeleton. Они относятся к дополнительной clip/highlight разметке и могут понадобиться позже, но текущий moment retrieval pipeline использует только ground-truth temporal windows.

`scripts/inspect_annotations.py` успешно прочитал validation annotation file. Annotation-only rehearsal experiments на первых 100 samples также успешно прошли. Summary сохранён здесь:

```text
results/qvhighlights_annotation_rehearsal/summary.csv
```

Это не `CLIP` baseline и не финальная оценка качества. Это проверка annotation compatibility и pipeline workflow на реальных QVHighlights validation annotations без видео и без CLIP.

## Inspect Annotations

Перед запуском экспериментов удобно быстро проверить, что annotation-файл читается loader-ом:

```bash
.venv/bin/python scripts/inspect_annotations.py \
  --annotations data/qvhighlights/val.jsonl \
  --limit 5
```

Скрипт печатает количество загруженных samples, несколько первых примеров, минимальную и максимальную длительность, а также среднее число `relevant_windows` на sample.

## Dummy Retrieval Check

Текущий dummy CLI можно использовать для проверки формата annotations и dataset-level pipeline:

```bash
.venv/bin/python scripts/run_dummy_retrieval.py \
  --annotations data/qvhighlights/val.jsonl \
  --window-size 8 \
  --stride 4 \
  --limit 100 \
  --output results/dummy_qvhighlights_val_100.json
```

Этот запуск не является `CLIP` baseline. Он не читает видео и не извлекает признаки. Dummy scoring выбирает окно по близости центра окна к центру видео:

```text
score = -abs(window_center - video_center)
```

Цель запуска - проверить loader, формат annotations и связку:

```text
annotations -> temporal windows -> dummy scores -> predictions -> metrics
```

## Loader troubleshooting

`missing required field`: loader не нашел одно из обязательных полей, например `sample_id`, `video_id`, `query`, `duration` или `relevant_windows`. Нужно открыть несколько строк annotation-файла и проверить реальные имена полей. Если у QVHighlights release используется другое имя, добавьте его в соответствующий список aliases в `src/data/qvhighlights.py`.

`unknown field names`: если данные выглядят корректно, но loader все равно не видит нужные значения, скорее всего, реальные поля называются иначе. Проверьте названия в JSONL и адаптируйте aliases в `src/data/qvhighlights.py`.

`empty relevant_windows`: у sample нет размеченных релевантных окон. Нужно проверить, является ли это нормальным для конкретного split. Если такие samples должны пропускаться, это стоит явно добавить в loader или отдельный preprocessing step.

`invalid window format`: окно не выглядит как `[start, end]`. Проверьте, не хранится ли разметка в другом формате, например объектом с ключами `start` и `end`. Если формат отличается, адаптируйте парсинг окон в `src/data/qvhighlights.py`.

`end < start`: конец временного окна меньше начала. Это означает ошибку в данных или несовпадение формата. Проверьте единицы измерения и порядок значений в annotation-файле.

`duration missing or invalid`: отсутствует длительность видео или её нельзя привести к `float`. Проверьте, как в реальном файле называется поле длительности. Если используется другое имя, добавьте его в aliases для duration в `src/data/qvhighlights.py`.
