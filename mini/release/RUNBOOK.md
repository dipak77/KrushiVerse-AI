# KrushiVerseAI Mini v1.0 Runbook

## Overview

Mini is a ~1M-parameter agriculture assistant integrated with platform RAG and agents.

| Flag | Default | Effect |
|---|---|---|
| `USE_MINI_LLM` | `false` | When `true`, planner uses Mini as synthesizer |
| `MINI_DEFAULT_MODE` | `grounded` | Requires sources for answers |
| `MINI_MODEL_VERSION` | `auto` | Prefer serve/v0.5-quant → v0.4 → v0.3 → v0.2 |

## Health checks

```bash
# API
curl http://127.0.0.1:8000/api/mini/status
curl -X POST http://127.0.0.1:8000/api/mini/chat -H "Content-Type: application/json" \
  -d "{\"query\":\"pink bollworm cotton IPM\",\"crop\":\"Cotton\",\"mode\":\"grounded\"}"

# Factory release gate
python -m mini.orchestrator release --execute
```

## Rollback

1. Set `USE_MINI_LLM=false` and restart API/Streamlit → classic template synthesizer.
2. Point `MINI_MODEL_VERSION` to a previous local tag (`v0.4`, `v0.3`).
3. Redeploy prior `mini/models/serve/<tag>` package if needed.

## Incident tips

- **Empty sources / grounded refusal:** check lake/KB and `W-RAG`; mini-local KB still serves cotton IPM demos.
- **Slow CPU latency:** reduce `MINI_MAX_NEW_TOKENS`; use INT8 package when available.
- **Unsafe advice:** validator + fallback templates; re-run `W-EVAL` probes.

## Artifacts (local-only weights)

- Checkpoints: `mini/models/v0.2-base` … `v0.5-quant`, `serve/`
- Reports: `EVAL_LATEST`, `INFER_LATEST`, `RELEASE_LATEST`, `CHECKLIST_SIGNED`

## Owners

See `SCALE_ROADMAP.json` owners and plan §8 deferred list.
