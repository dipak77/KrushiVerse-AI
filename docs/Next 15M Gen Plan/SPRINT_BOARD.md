# Mini v2 (15M) — Sprint Board (S18 → S32)

**Program version target:** `2.0.0-15M`  
**Hardware:** RTX 2050 4GB + 16GB RAM (laptop recipe)  
**Status key:** `done` | `in_progress` | `next` | `planned`  
**Last update:** S20 start  

---

## How to use this board

1. Work **one sprint at a time**; exit gate must pass before promoting the next checkpoint.  
2. Keep **v1.0 weights** until S30 beta.  
3. After every heavy train day: `python -m mini.orchestrator cleanup --execute` if disk grows.  
4. Canonical config: `mini/models/config_v2_15M.json`.

---

## PHASE 1 — Foundation & fixes

| Sprint | Name | Status | Deliverables | Exit gate | Commands |
|---|---|---|---|---|---|
| **S18** | Tokenizer + data hygiene | **done** | Cleanup tool; 8k unigram train path; plan docs | Disk cleaned; fertility path works | `cleanup --execute` · `token --version v2-8k --vocab-size 8192` |
| **S19** | 15M infra & config | **done** | `config_v2_15M`, MiniConfig.v2, grad_ckpt, assistant-only SFT masks | Params ≈ **15.02M**; foundation tests green | `pytest tests/test_mini_v2_15m_foundation.py` |

---

## PHASE 2 — Pretrain

| Sprint | Name | Status | Deliverables | Exit gate | Commands |
|---|---|---|---|---|---|
| **S20** | v0.6-base pretrain | **in_progress / next** | Train loop: fp16, grad_accum, grad_ckpt, block 1024, out `v0.6-base/` | Smoke: loss drops; full: val PPL improves, ~10k–15k steps | See below |

### S20 laptop recipe (RTX 2050)
```bash
# Optional: train unified tokenizer first
python -m mini.orchestrator token --execute --version v2-8k --vocab-size 8192 --model-type unigram

# Smoke (CI / verify wiring) ~ few minutes CPU/GPU
python -m mini.orchestrator pretrain --execute --variant v2-15M --version v0.6-base \
  --steps 40 --batch-size 2 --grad-accum 2 --block-size 128 --fp16 --grad-checkpoint --seed 42

# Overnight full Phase A (GPU recommended)
python -m mini.orchestrator pretrain --execute --variant v2-15M --version v0.6-base \
  --steps 10000 --batch-size 4 --grad-accum 4 --block-size 1024 --fp16 --grad-checkpoint \
  --lr 3e-4 --seed 42 --eval-every 500
```

**Risk mitigations:** OOM → lower batch to 2; thermal → pause; power cut → save every 500 steps (built-in).

---

## PHASE 3 — SFT & safety

| Sprint | Name | Status | Deliverables | Exit gate | Commands (planned) |
|---|---|---|---|---|---|
| **S21** | v0.7-instruct | planned | Instruct + safety SFT from v0.6, assistant-only, 2k steps | Safety probes high; instruct F1↑ | `sft --stage instruct --init v0.6-base --steps 2000` |
| **S22** | v0.8-agri-qa | planned | RAG-context SFT 3k steps, citation boost | Grounding >0.75; primary ckpt | `sft --stage agri-qa --init v0.7-instruct --steps 3000` |
| **S23** | Guardrails v2 | planned | No-number rule, adversarial probes, SAFETY_REPORT | 100% on 50 hard probes | eval + validate_v2 |

---

## PHASE 4 — Serve & product

| Sprint | Name | Status | Deliverables | Exit gate |
|---|---|---|---|---|
| **S24** | Quant + ONNX | planned | INT8 + ONNX, latency report | p95 & size budgets |
| **S25** | RAG chat v2 | planned | 1024 context, citations stable on `/api/mini/chat` | n_sources≥2 on 80% |

---

## PHASE 5 — Multilingual & gold

| Sprint | Name | Status | Deliverables | Exit gate |
|---|---|---|---|---|
| **S26** | Marathi + translit | planned | Roman-MR pairs, MR eval | MR fluency ↑ |
| **S27** | Human gold 200 | planned | EVAL_MH_GOLD + scorecard | F1/EM targets |

---

## PHASE 6 — Prod hardening

| Sprint | Name | Status | Deliverables | Exit gate |
|---|---|---|---|---|
| **S28** | API/UI prod | planned | max_new_tokens 180, disclaimer, rate limit | UX checklist |
| **S29** | Observability | planned | RUNBOOK v2, metrics, rollback <2 min | Flag rollback works |
| **S30** | Beta release | planned | v2.0.0-prod tag, farmer beta log | RC checklist |

---

## PHASE 7 — Scale optional

| Sprint | Name | Status | Notes |
|---|---|---|---|
| **S31** | 300M-token cloud pretrain | planned | Optional; only if budget |
| **S32** | 50M plan + distill | planned | Same factory family |

---

## Checkpoint map

```text
v2-8k tokenizer  →  v0.6-base (S20)  →  v0.7-instruct (S21)
                                      →  v0.8-agri-qa (S22) ★ primary
                                      →  INT8/ONNX (S24) → beta (S30)
```

Rollback: `MINI_MODEL_VERSION=v0.4` / `USE_MINI_LLM` + v1.0 serve path.

---

## Immediate next (this session)

1. ✅ Write this sprint board  
2. **Implement S20** pretrain variant `v2-15M` → `v0.6-base`  
3. Run **smoke** pretrain + tests  
4. Document S20 notes  
5. (User overnight) full 10k-step GPU run  

---

## Definition of “S20 complete”

- [x] CLI: `--variant v2-15M --version v0.6-base --fp16 --grad-checkpoint --grad-accum`  
- [x] Worker passes kwargs through  
- [x] Checkpoint dir `mini/models/v0.6-base/` (local, gitignored)  
- [x] Smoke test: train loss decreases  
- [ ] Full overnight 10k steps on your RTX 2050 (user machine)  
- [ ] Val PPL gate for promotion to S21  
