# KrushiVerseAI Mini Factory

Lightweight agriculture ML factory that will produce the **~1M-parameter Mini LLM**, while sharing contracts with the platform RAG/agents stack.

| Field | Value |
|---|---|
| Sprint | **S6 — QA synthesis** (≥10k train / ≥1k val) |
| Package | `mini/` |
| Schema | `1.0` (`StandardRecord`) |
| Taxonomy | **v1.0.0 frozen** |
| Feature phase | **FP-3** |
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
python -m mini.orchestrator analyze --execute
python -m mini.orchestrator qasynth --execute --target 12000
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

## Sprint 0–6 acceptance

- [x] Factory through analyze dashboard  
- [x] **W-QASYNTH** multilingual packs ≥10k train / ≥1k val  
- [x] Human review queue CSV + no fact_key leakage  
- [x] Tests: `test_mini_sprint0`–`6`  

```bash
python -m mini.orchestrator qasynth --execute --target 12000
python -m mini.orchestrator run sprint6 --execute
```

## Next

Sprint 7 — Domain expert packs expansion toward ~50k.
