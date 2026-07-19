# KrushiVerseAI Mini Factory — **v1.0 shipped · v2 15M in progress**

| Field | Value |
|---|---|
| Version | **2.0.0-dev** (v1.0.0 still supported) |
| Sprint | **S18 — v2 15M foundation** |
| Feature phase | **v2-15M** |
| v1 plan | [`docs/KRUSHIVERSE_MINI_SPRINT_PLAN.md`](../docs/KRUSHIVERSE_MINI_SPRINT_PLAN.md) |
| **v2 15M plan** | **[`docs/Next 15M Gen Plan/IMPLEMENTATION_PLAN.md`](../docs/Next%2015M%20Gen%20Plan/IMPLEMENTATION_PLAN.md)** |
| Config | [`mini/models/config_v2_15M.json`](models/config_v2_15M.json) |
| Runbook | [`mini/release/RUNBOOK.md`](release/RUNBOOK.md) |
| **Model guide (v1)** | **[`docs/MINI_MODEL.md`](../docs/MINI_MODEL.md)** |

## Disk cleanup (local)

```bash
python -m mini.orchestrator cleanup              # dry-run
python -m mini.orchestrator cleanup --execute    # free GB from mini/datasets/versions
```

## v2 tokenizer (8k unigram)

```bash
python -m mini.orchestrator token --execute --version v2-8k --vocab-size 8192 --model-type unigram
```
## Product

```bash
GET  /api/mini/status
POST /api/mini/chat
GET  /api/mini/release
POST /api/mini/release?execute=true
POST /api/query          # USE_MINI_LLM=true → Mini synthesizer
```

## Release gate

```bash
python -m mini.orchestrator release --execute
python -m mini.orchestrator run sprint17 --execute
```

Writes (local): `RELEASE_LATEST.json`, `CHECKLIST_SIGNED.json`, `VERSION_MATRIX.json`, `SCALE_ROADMAP.json`, `RUNBOOK.md`.

## Factory CLI (selected)

```bash
python -m mini.orchestrator list-workers
python -m mini.orchestrator infer --execute --query "pink bollworm cotton IPM"
python -m mini.orchestrator eval --execute --version v0.4
python -m mini.orchestrator quant --execute
python -m mini.orchestrator deploy --execute --force
python -m mini.orchestrator release --execute
```

## Sprint 0–17 complete

- [x] Lake, QA, KG, tokenizer, ~1M pretrain, SFT, eval, quant, deploy  
- [x] Grounded Mini+RAG inference + platform chat + feature flag  
- [x] **v1.0 RC gate + checklist + load smoke + version matrix + scale decision**  

**Local-only:** data lake, model weights, eval/infer/release JSON dumps.

## Scale (post-v1.0)

**Decision:** proceed with same family — **10M → 50M → 100M → 300M → 1B**. See `mini/release/scale_roadmap.py`.
