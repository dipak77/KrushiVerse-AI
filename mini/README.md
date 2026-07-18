# KrushiVerseAI Mini Factory

| Field | Value |
|---|---|
| Sprint | **S16 — Platform Mini chat + USE_MINI_LLM** |
| Feature phase | **FP-9** |
| Plan | [`docs/KRUSHIVERSE_MINI_SPRINT_PLAN.md`](../docs/KRUSHIVERSE_MINI_SPRINT_PLAN.md) |

## Product endpoints

```bash
GET  /api/mini/status
POST /api/mini/chat   # always Mini+RAG (grounded by default)
POST /api/query       # planner; Mini synthesizer only if USE_MINI_LLM=true
```

## Feature flag

```bash
# Default: classic template synthesizer
USE_MINI_LLM=false

# Mini becomes planner brain (agents still run as specialists)
USE_MINI_LLM=true
```

## CLI (factory)

```bash
python -m mini.orchestrator infer --execute --query "pink bollworm cotton IPM"
python -m mini.orchestrator eval --execute --version v0.4
python -m mini.orchestrator quant --execute
python -m mini.orchestrator deploy --execute --force
```

## Sprint 0–16

- [x] Lake → train → SFT → eval → quant → infer  
- [x] **Platform `/api/mini/chat` + flag-gated planner Mini synthesizer**  
- [x] Streamlit Mini panel with citations  

## Next

Sprint 17 — production beta → Mini v1.0 checklist / hardening.
