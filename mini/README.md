# KrushiVerseAI Mini Factory

Lightweight agriculture ML factory that will produce the **~1M-parameter Mini LLM**, while sharing contracts with the platform RAG/agents stack.

| Field | Value |
|---|---|
| Sprint | **S10 — Mini ~1M arch + train harness** |
| Package | `mini/` |
| Schema | `1.0` (`StandardRecord`) |
| Taxonomy | **v1.0.0 frozen** |
| Feature phase | **FP-5b** |
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
python -m mini.orchestrator qasynth --execute --target 62500
python -m mini.orchestrator kgbuild --execute
python -m mini.orchestrator token --execute --vocab-size 32000
python -m mini.orchestrator pretrain --execute --steps 50
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

## Sprint 0–10 acceptance

- [x] Factory through QA synth + KG + tokenizer  
- [x] **W-PRETRAIN** Mini ~1.36M arch (RoPE/RMSNorm/SwiGLU) + overfit smoke  
- [x] Tests: `test_mini_sprint0`–`10`  

**Local-only (do not push):** `data/lake/**`, `mini/datasets/**`, tokenizer binaries, model checkpoints.

```bash
python -m mini.orchestrator pretrain --execute --steps 50
python -m mini.orchestrator run sprint10 --execute
```

## Next

Sprint 11 — full domain pretraining on agri corpus.
