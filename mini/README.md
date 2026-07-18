# KrushiVerseAI Mini Factory

Lightweight agriculture ML factory that will produce the **~1M-parameter Mini LLM**, while sharing contracts with the platform RAG/agents stack.

| Field | Value |
|---|---|
| Sprint | **S4 — Standard records** (normalize / lang / export) |
| Package | `mini/` |
| Schema | `1.0` (`StandardRecord`) |
| Taxonomy | **v1.0.0 frozen** |
| Feature phase | **FP-2** (versioned train/val/test) |
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
python -m mini.orchestrator sources
python -m mini.orchestrator ingest --execute --skip-http
python -m mini.orchestrator quality --execute
python -m mini.orchestrator standardize --execute
python -m mini.orchestrator lake-status
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

## Sprint 0–4 acceptance

- [x] Factory + taxonomy + ingest + quality  
- [x] **W-NORMALIZE / W-LANGDETECT / W-STANDARD**  
- [x] Schema v1 JSONL + parquet train/val/test (≥90% lang+category)  
- [x] Tests: `test_mini_sprint0`–`4`  

```bash
python -m mini.orchestrator quality --execute
python -m mini.orchestrator standardize --execute
python -m mini.orchestrator run sprint4 --execute
```

## Next

Sprint 5 — `W-ANALYZE` coverage dashboard after each standardize cycle.
