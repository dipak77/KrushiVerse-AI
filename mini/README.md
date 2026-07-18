# KrushiVerseAI Mini Factory

Lightweight agriculture ML factory for the **~1M-parameter Mini LLM**, sharing contracts with the platform RAG/agents stack.

| Field | Value |
|---|---|
| Sprint | **S15 — Inference pipeline (Mini + RAG)** |
| Package | `mini/` |
| Schema | `1.0` (`StandardRecord`) |
| Taxonomy | **v1.0.0 frozen** |
| Feature phase | **FP-8** |
| Plan | [`docs/KRUSHIVERSE_MINI_SPRINT_PLAN.md`](../docs/KRUSHIVERSE_MINI_SPRINT_PLAN.md) |

## CLI

```bash
python -m mini.orchestrator list-workers
python -m mini.orchestrator status
python -m mini.orchestrator pretrain --execute --mode domain --steps 200
python -m mini.orchestrator sft --execute --steps-v03 120 --steps-v04 120
python -m mini.orchestrator eval --execute --version v0.4
python -m mini.orchestrator quant --execute --version v0.4
python -m mini.orchestrator deploy --execute --tag v0.5-quant --force
python -m mini.orchestrator rag --execute --query "pink bollworm cotton IPM"
python -m mini.orchestrator infer --execute --query "How do I manage pink bollworm in cotton with IPM in Maharashtra?"
```

## Sprint 0–15 acceptance

- [x] Lake → QA → KG → tokenizer → pretrain → SFT → eval → quant/deploy  
- [x] **Grounded Mini+RAG inference with citations + validator + fallback**  
- [x] Tests: `test_mini_sprint0`–`15`  

**Local-only:** `data/lake/**`, checkpoints, `INFER_LATEST.json`, quant/serve dumps.

## Next

Sprint 16 — `/api/mini/chat`, Streamlit Mini panel, `USE_MINI_LLM` feature flag.
