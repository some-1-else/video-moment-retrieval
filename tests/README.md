# Тесты

Публичные тесты проверяют переиспользуемый код проекта и основные скрипты
воспроизведения:

- temporal metrics, scoring, aggregation и windowing;
- data loaders и sample data helpers;
- frame sampling/extraction и CLIP/cache interfaces;
- поведение CLIP retrieval pipeline;
- компиляцию публичных скриптов из `scripts/`.

Запуск:

```bash
pytest
```

Тесты не требуют полного набора raw videos Charades-STA. Для тяжёлых
экспериментов используются сохранённые результаты в `results/`.
