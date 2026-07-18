# KrushiVerseAI Mini Factory

Lightweight agriculture ML factory that will produce the **~1M-parameter Mini LLM**, while sharing contracts with the platform RAG/agents stack.

| Field | Value |
|---|---|
| Sprint | **S12 — Instruction + agri-QA SFT (v0.3 / v0.4)** |
| Package | `mini/` |
| Schema | `1.0` (`StandardRecord`) |
| Taxonomy | **v1.0.0 frozen** |
| Feature phase | **FP-7** |
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

## Sprint 0–12 acceptance

- [x] Factory through QA synth + KG + tokenizer  
- [x] Mini ~1.36M arch + domain pretrain v0.2-base  
- [x] **SFT v0.3-instruct + v0.4-agri-qa** (F1/loss beats base)  
- [x] Tests: `test_mini_sprint0`–`12`  

**Local-only (do not push):** `data/lake/**`, `mini/datasets/**`, tokenizer binaries, `mini/models/v0.2-base/**`, `mini/models/v0.3-instruct/**`, `mini/models/v0.4-agri-qa/**`.

```bash
python -m mini.orchestrator sft --execute --steps-v03 120 --steps-v04 120 --seed 42
python -m mini.orchestrator run sprint12 --execute
```

## Next

Sprint 13 — `W-EVAL` gold sets, gates, hallucination probes.
