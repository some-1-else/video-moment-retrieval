# Тесты

Публичные тесты проверяют переиспользуемый код проекта и текущие reproduction
entrypoints:

- temporal metrics, scoring, aggregation и windowing;
- data loaders и sample data helpers;
- frame sampling/extraction и CLIP/cache interfaces;
- поведение CLIP retrieval pipeline;
- public scripts из `scripts/`.

Legacy tests для старых probe, synthetic, QVHighlights и ignored agent-memory
scripts перенесены в локальную `.agent_memory/tests/` вместе с соответствующими
скриптами.
