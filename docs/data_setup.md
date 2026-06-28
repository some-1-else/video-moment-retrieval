# Подготовка данных

Документ описывает подготовку данных для воспроизведения финальных
экспериментов: Charades-STA annotations и локальные Charades videos. Проект не
скачивает полный Charades или Charades-STA dataset автоматически.

## Ожидаемая структура

Charades-STA annotation files должны лежать здесь:

```text
data/charades_sta/
  charades_sta_train.txt
  charades_sta_test.txt
```

Локальные Charades videos должны лежать здесь:

```text
data/raw/charades/videos/
  <video_id>.mp4
```

Код также принимает `.avi` и `.mkv`, но `.mp4` является предпочтительным
локальным форматом. Каждая строка Charades-STA annotation ожидается в формате:

```text
<video_id> <start_time> <end_time>##<sentence>
```

Пример:

```text
ABC123 12.3 18.7##person opens a door
```

Соответствующий локальный video path:

```text
data/raw/charades/videos/ABC123.mp4
```

## Архив Charades videos

Raw Charades videos хранятся в официальном AI2 Charades archive. Скачанный
архив нужно держать вне git, например:

```text
data/raw/charades/archives/Charades_v1_480.zip
```

Для экспериментов курсовой достаточно 480p archive: он меньше, быстрее
обрабатывается локально и подходит для CLIP baseline.

## Извлечение видео

Для первичной проверки удобно начать с небольшой выборки:

```bash
python scripts/prepare_data.py \
  --zip_path data/raw/charades/archives/Charades_v1_480.zip \
  --limit 20
```

Скрипт извлекает выбранные `.mp4` files в:

```text
data/raw/charades/videos/
```

и записывает локальный extraction manifest:

```text
data/processed/charades_zip_extract_manifest.csv
```

Этот manifest является промежуточным файлом и не нужен для финальной проверки
репозитория.

Если нужные video ids уже известны, положите их в text или CSV file:

```text
data/charades_sta/smoke_video_ids.txt
```

Затем извлеките только эти видео:

```bash
python scripts/prepare_data.py \
  --zip_path data/raw/charades/archives/Charades_v1_480.zip \
  --video_ids data/charades_sta/smoke_video_ids.txt \
  --limit 20
```

Если одновременно переданы `--video_ids` и `--limit`, скрипт сначала фильтрует
по requested ids, а затем извлекает не больше первых `N` matching files.

## Проверка данных

Перед запуском CLIP experiments нужно убедиться, что annotations и локальные
videos согласованы:

- annotation files существуют в `data/charades_sta/`;
- нужные videos существуют в `data/raw/charades/videos/`;
- OpenCV может открыть видео и прочитать первый frame;
- размеченные windows лежат внутри decoded video duration.

Финальные experiment runners выполняют эти проверки и сохраняют `run_config.json`
рядом с результатами.

## Замечания по воспроизведению

- Raw archives, extracted videos, manifests, caches и embeddings не должны
  попадать в git, если это не специально подготовленные маленькие sample files.
- Перед масштабированием на фиксированную выборку из 1,000 запросов лучше
  проверить небольшой набор из 5-20 videos.
- Финальные папки результатов хранят `run_config.json`, `metrics.json`,
  `summary.csv` и prediction files, чтобы reported numbers можно было
  проверить без повторного inference.
