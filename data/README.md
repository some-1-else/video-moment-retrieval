# Данные

Папка описывает данные, необходимые для финальной версии проекта. Основная
экспериментальная линия: **Charades-STA + локальные Charades videos**.

## Что входит в git

- `charades_sta/charades_sta_train.txt` и `charades_sta_test.txt` - разметка
  Charades-STA.
- `processed/charades_sta_full_test_manifest.csv` - проверенный manifest
  полного test split Charades-STA.
- `processed/charades_sta_full_test_moment_detr.jsonl` - тот же test split в
  JSONL-формате для Moment-DETR raw-video evaluation.
- `sample/` - небольшие sample annotations для тестов.
- `.gitkeep` files - маркеры для локальных директорий с данными и cache.

## Что не входит в git

- `raw/` - локальные Charades videos и скачанные архивы.
- `embeddings/` - локальные embedding caches.
- промежуточные manifests и временные artifacts, кроме специально сохранённых
  файлов в `processed/`.
- локальные QVHighlights videos/manifests.

## Ожидаемая локальная структура

```text
data/
  charades_sta/
    charades_sta_train.txt
    charades_sta_test.txt
  raw/
    charades/
      videos/
        <video_id>.mp4
  processed/
    charades_sta_full_test_manifest.csv
    charades_sta_full_test_moment_detr.jsonl
  embeddings/
```

Подробная инструкция по подготовке Charades videos и проверке данных находится
в `docs/data_setup.md`.
