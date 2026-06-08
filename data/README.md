# Data

Эта папка описывает данные, используемые в финальной версии курсового проекта.
Основная экспериментальная линия сейчас: **Charades-STA + локальные Charades
videos**. QVHighlights сохранен только как preliminary/history direction и не
является финальным benchmark в репозитории.

## Что входит в git

- `charades_sta/charades_sta_train.txt` и `charades_sta_test.txt` - annotations
  Charades-STA.
- `processed/charades_sta_full_test_manifest.csv` - проверенный manifest полного
  Charades-STA test split.
- `processed/charades_sta_full_test_moment_detr.jsonl` - Charades-STA test split
  в JSONL format для Moment-DETR raw-video evaluation.
- `sample/` - маленькие sample annotations для тестов.
- `.gitkeep` files - placeholders для локальных raw/cache директорий.

## Что не входит в git

- `raw/` - локальные Charades videos и скачанные archives.
- `embeddings/` - локальные embedding caches.
- временные manifests и промежуточные artifacts, кроме специально сохраненных
  финальных files в `processed/`.
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
