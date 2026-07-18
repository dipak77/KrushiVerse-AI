# KrushiVerseAI Mini Factory

Lightweight agriculture ML factory that will produce the **~1M-parameter Mini LLM**, while sharing contracts with the platform RAG/agents stack.

| Field | Value |
|---|---|
| Sprint | **S3 — Quality pipeline** (validate / clean / dedup) |
| Package | `mini/` |
| Schema | `1.0` (`StandardRecord`) |
| Taxonomy | **v1.0.0 frozen** |
| Feature phase | **FP-2** (processed lake) |
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

## Sprint 0–3 acceptance

- [x] Factory CLI + lake layout + taxonomy freeze  
- [x] W-INGEST source registry (idempotent)  
- [x] **W-VALIDATE / W-CLEAN / W-DEDUP / W-QUALITY**  
- [x] Tests: `test_mini_sprint0`–`3`  

```bash
python -m mini.orchestrator ingest --execute --skip-http
python -m mini.orchestrator quality --execute
python -m mini.orchestrator run sprint3 --execute
```

## Next

Sprint 4 — Normalize + language detect + Schema v1 standard records → training/.
