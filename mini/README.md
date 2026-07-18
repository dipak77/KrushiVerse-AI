# KrushiVerseAI Mini Factory

Lightweight agriculture ML factory that will produce the **~1M-parameter Mini LLM**, while sharing contracts with the platform RAG/agents stack.

| Field | Value |
|---|---|
| Sprint | **S1 — Taxonomy freeze** (S0 bootstrap complete) |
| Package | `mini/` |
| Schema | `1.0` (`StandardRecord`) |
| Taxonomy | **v1.0.0 frozen** |
| Plan | [`docs/KRUSHIVERSE_MINI_SPRINT_PLAN.md`](../docs/KRUSHIVERSE_MINI_SPRINT_PLAN.md) |

## CLI

```bash
# From repo root
python -m mini.orchestrator list-workers
python -m mini.orchestrator list-pipelines
python -m mini.orchestrator status
python -m mini.orchestrator init-lake
python -m mini.orchestrator run bootstrap --execute
python -m mini.orchestrator run dry-factory          # dry-run all workers
python -m mini.orchestrator run-worker W-BOOTSTRAP --dry-run
```

## Layout

```
mini/
  contracts.py      # StandardRecord, WorkerResult
  paths.py          # Lake + artifact paths
  taxonomy/         # Domain taxonomy draft
  workers/          # Automated worker modules
  orchestrator/     # DAG + CLI
  models/           # future checkpoints
  datasets/         # future manifests
  eval/             # future gold sets
  inference/        # future serve chain
```

Data lake (created by `init-lake` / `W-BOOTSTRAP`):

```
data/lake/
  raw/{domain}/
  processed/{domain}/
  training/
  validation/
  test/
  quarantine/
```

**Never train from `raw/`.**

## Sprint 0–1 acceptance

- [x] `python -m mini.orchestrator list-workers` works  
- [x] Lake dirs creatable via CLI  
- [x] Schema v1 `StandardRecord` defined  
- [x] Taxonomy **v1.0.0 frozen** + platform KB coverage  
- [x] `taxonomy-validate` / `taxonomy-summary` CLI  
- [x] Dry-run pipeline `dry-factory` / `sprint1` succeeds  
- [x] Tests: `tests/test_mini_sprint0.py`, `tests/test_mini_sprint1.py`  

```bash
python -m mini.orchestrator taxonomy-validate
python -m mini.orchestrator run sprint1 --execute
```

## Next

Sprint 2 — Real ingest worker + source registry + lake seed manifests.
