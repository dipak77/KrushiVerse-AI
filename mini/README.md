# KrushiVerseAI Mini Factory

Lightweight agriculture ML factory that will produce the **~1M-parameter Mini LLM**, while sharing contracts with the platform RAG/agents stack.

| Field | Value |
|---|---|
| Sprint | **S14 — Quantize + version packaging** |
| Package | `mini/` |
| Schema | `1.0` (`StandardRecord`) |
| Taxonomy | **v1.0.0 frozen** |
| Feature phase | **E5-quant** |
| Plan | [`docs/KRUSHIVERSE_MINI_SPRINT_PLAN.md`](../docs/KRUSHIVERSE_MINI_SPRINT_PLAN.md) |

## CLI

```bash
# From repo root
python -m mini.orchestrator list-workers
python -m mini.orchestrator list-pipelines
python -m mini.orchestrator status
python -m mini.orchestrator init-lake
python -m mini.orchestrator run bootstrap --execute
python -m mini.orchestrator run dry-factory
python -m mini.orchestrator pretrain --execute --mode domain --steps 200 --seed 42
python -m mini.orchestrator sft --execute --steps-v03 120 --steps-v04 120 --seed 42
python -m mini.orchestrator eval --execute --version v0.4 --profile default
python -m mini.orchestrator quant --execute --version v0.4
python -m mini.orchestrator deploy --execute --version v0.4 --tag v0.5-quant --force
python -m mini.orchestrator lake-status
```

## Layout

```
mini/
  contracts.py
  paths.py
  taxonomy/
  workers/          # W-INGEST … W-QUANT, W-DEPLOY
  orchestrator/     # DAG + CLI
  models/           # train/SFT/quant/deploy code; checkpoints local-only
  eval/             # gold sets, probes, gates, harness
  inference/        # next: serve chain
```

**Never train from `raw/`.**

## Sprint 0–14 acceptance

- [x] Factory through QA synth + KG + tokenizer  
- [x] Mini ~1.36M + pretrain + SFT + eval gates  
- [x] **INT8/INT4 quant + serve package + version registry**  
- [x] Tests: `test_mini_sprint0`–`14`  

**Local-only:** `data/lake/**`, `mini/datasets/**`, `mini/models/v0.*/**`, `mini/models/serve/**`, eval/quant/deploy JSON dumps.

```bash
python -m mini.orchestrator quant --execute --version v0.4
python -m mini.orchestrator deploy --execute --tag v0.5-quant --force
python -m mini.orchestrator run sprint14 --execute
```

## Next

Sprint 15 — `W-INFER` + RAG wrap: intent → retrieve → Mini → validate.
