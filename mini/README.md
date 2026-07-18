# KrushiVerseAI Mini Factory

Lightweight agriculture ML factory that will produce the **~1M-parameter Mini LLM**, while sharing contracts with the platform RAG/agents stack.

| Field | Value |
|---|---|
| Sprint | **S13 — Evaluation harness (W-EVAL)** |
| Package | `mini/` |
| Schema | `1.0` (`StandardRecord`) |
| Taxonomy | **v1.0.0 frozen** |
| Feature phase | **E5-eval** |
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
python -m mini.orchestrator pretrain --execute --mode domain --steps 200 --seed 42
python -m mini.orchestrator sft --execute --steps-v03 120 --steps-v04 120 --seed 42
python -m mini.orchestrator eval --execute --version v0.4 --profile default
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
  models/           # train + SFT code; checkpoints local-only
  eval/             # gold sets, probes, gates, harness (reports local)
  datasets/         # future manifests
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

## Sprint 0–13 acceptance

- [x] Factory through QA synth + KG + tokenizer  
- [x] Mini ~1.36M arch + domain pretrain v0.2-base  
- [x] SFT v0.3-instruct + v0.4-agri-qa  
- [x] **W-EVAL scorecard (HTML/JSON) + gates (non-zero on fail)**  
- [x] Tests: `test_mini_sprint0`–`13`  

**Local-only (do not push):** `data/lake/**`, `mini/datasets/**`, tokenizer binaries, `mini/models/v0.*/**`, `mini/eval/EVAL_*.json|html`.

```bash
python -m mini.orchestrator eval --execute --version v0.4 --profile default
python -m mini.orchestrator run sprint13 --execute
```

## Next

Sprint 14 — `W-QUANT` INT8/INT4 + packaging; `W-DEPLOY` version registry.
