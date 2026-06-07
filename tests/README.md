# Tests

Public tests cover the reusable project code and the current reproduction
entrypoints:

- temporal metrics, scoring, aggregation, and windowing;
- data loaders and sample data helpers;
- frame sampling/extraction and CLIP/cache interfaces;
- CLIP retrieval pipeline behavior;
- public scripts under `scripts/`.

Legacy tests for old probe, synthetic, QVHighlights, and ignored agent-memory
scripts were moved to local `.agent_memory/tests/` together with those scripts.
