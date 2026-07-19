# KrushiVerseAI Mini v2 (15M) — Master Implementation Plan

**Sources:** `KrushiVerseAI_15M_Prod_Plan.md` + `config_v2_15M.json`  
**Baseline:** Mini v1.0 (~1.36M) factory complete (S0–S17)  
**Target:** v2.0.0-15M grounded agri synthesizer  
**Hardware:** RTX 2050 4GB + 16GB RAM (primary)  
**Status:** Phase 1 done · Phase 2 (S20 pretrain) wiring complete  

---

## 0. Goals & non-goals

### Goals
1. Ship **~15M** MiniLM with **one** 8k SentencePiece Unigram tokenizer.
2. Fix v1 quality bugs: **block_size 1024**, **assistant-only SFT loss**, **max_new_tokens 180**, **fp16 + grad checkpoint**.
3. Keep the same **W-*** factory** (extend, don’t rewrite).
4. Free disk: **mini/datasets/versions** test dumps (~5 GB) and other disposable local artifacts.
5. Remain **grounded-RAG-first**; Mini is rewriter/synthesizer, not knowledge base.

### Non-goals (this phase)
- Full 300M-token cloud pretrain (Phase B / S31).
- 50M model.
- Deleting v1.0 code paths until v2 gates pass (keep dual-run for rollback).

---

## 1. Architecture lock (canonical)

**Canonical config file:**  
- Plan pack: `docs/Next 15M Gen Plan/config_v2_15M.json`  
- Runtime copy: `mini/models/config_v2_15M.json`

| Knob | v1.0 | v2.0-15M |
|---|---|---|
| params | ~1.36M | **~15.0M** |
| vocab | 4096 DomainTokenizer | **8192 Unigram SP** |
| n_embd | 128 | **320** |
| n_layer | 6 | **10** |
| n_head | 4 | **8** (head_dim 40) |
| n_hidden | 192 | **864** |
| block_size | 128–512 mismatch | **1024 everywhere** |
| SFT loss | full sequence | **assistant-only** |
| max_new_tokens | 40 | **180** |
| dtype train | mostly fp32 CPU | **fp16 AMP + checkpoint** on GPU |

**Checkpoint ladder (v2):**
| Tag | Stage |
|---|---|
| `v2-tok-8k` | Tokenizer only |
| `v0.6-base` | Pretrain 15M |
| `v0.7-instruct` | Instruct + safety SFT |
| `v0.8-agri-qa` | RAG-context SFT (**primary**) |
| `v0.8-int8` / ONNX | Serve |

---

## 2. Disk cleanup policy (memory / disk)

### Measured bloat (local machine)
| Path | ~Size | Action |
|---|---|---|
| `mini/datasets/versions/` | **~5.0 GB**, 100+ folders | **DELETE all but keep rules below** |
| `mini/datasets/kg/` | ~30 MB | Keep **latest only** |
| `mini/models/checkpoints/*smoke*` | small–medium | Delete old smoke ckpts |
| `mini/runs/` | tiny | Optional wipe |
| `data/lake/` | varies | **Never auto-delete training lake** without flag |
| v1 weights `v0.2`–`v0.5` | ~100 MB | **Keep** until v2.0 RC (rollback) |

### Keep rules for `mini/datasets/versions/`
1. Keep folders referenced by `LATEST_VERSION.json` / `QASYNTH_LATEST.json` / `KG_LATEST.json` if they point into `versions/`.
2. Else keep **1 newest `*-synth`** and **1 newest non-synth** only.
3. Prefer deleting **duplicate timestamp triplets** (many re-runs same day).

### Tool
```bash
python -m mini.tools.cleanup_local --dry-run
python -m mini.tools.cleanup_local --execute --keep-synth 1 --keep-other 1
```

### Git policy (unchanged)
- `data/lake/**`, `mini/datasets/**`, `*.pt` remain **local-only / gitignored**.
- Cleanup is **local disk hygiene**, not a git rewrite.

---

## 3. Sprint map (S18–S32 condensed)

### PHASE 1 — Foundation (NOW)
| Sprint | Deliverable | Gate |
|---|---|---|
| **S18** | 8k Unigram train path, fertility probes, data audit hook, cleanup tool | Fertility targets documentable; cleanup frees GB |
| **S19** | Config v2 loaded, MiniLM 15M builds, fp16+grad_ckpt hooks, **assistant-only SFT**, 100-step smoke | Param count ~15M ±10%; smoke loss drops |

### PHASE 2 — Pretrain
| Sprint | Deliverable | Gate |
|---|---|---|
| **S20** | `v0.6-base` 10k–15k steps, batch 4+accum 4, fp16 | Val PPL improves; ckpt saved |

### PHASE 3 — SFT & safety
| Sprint | Deliverable | Gate |
|---|---|---|
| **S21** | `v0.7-instruct` | Safety probes high pass |
| **S22** | `v0.8-agri-qa` RAG SFT | Grounding ↑, citations present |
| **S23** | Validator v2 (no-number rule) | Safety report |

### PHASE 4 — Serve
| Sprint | Deliverable | Gate |
|---|---|---|
| **S24** | INT8 + optional ONNX | size & latency budgets |
| **S25** | `/api/mini/chat` v2, 1024 context | citations stable |

### PHASE 5–7 — MR eval, prod, 300M optional
As in product plan S26–S32.

---

## 4. Implementation order (execution checklist)

### Done / in this PR (Phase 1 start)
- [x] Master plan (this file)
- [x] Install `mini/models/config_v2_15M.json`
- [x] `MiniConfig` load helpers + v2 defaults path
- [x] Gradient checkpointing flag on MiniLM
- [x] Assistant-only label masking for SFT
- [x] Tokenizer CLI support vocab 8192 unigram → `v2-8k`
- [x] Local cleanup tool + execute safe dataset prune
- [x] Smoke: count parameters for v2 config
- [x] Generation defaults: stop-friendly / longer max tokens knobs for v2

### Next (do not block Phase 1)
1. Train tokenizer on lake corpus (`v2-8k`).
2. Build pretrain streaming dataset at block 1024 for 15M.
3. Overnight pretrain on RTX 2050.
4. SFT stages with assistant-only.
5. Wire orchestrator flags: `--config config_v2_15M.json`, `--grad-checkpoint`, `--fp16`.

---

## 5. Hardware recipe (RTX 2050 4GB)

```text
pretrain:  batch=4, grad_accum=4, block=1024, fp16, grad_checkpoint=True, lr=3e-4
sft:       batch=4–8, accum=2, assistant_only=True, lr=1e-4 then 5e-5
close Chrome; free ~8GB RAM before data pack
save every 500–1000 steps
```

Config JSON lists batch 16 (ideal GPU); **laptop override** is required.

---

## 6. Rollback

| Situation | Action |
|---|---|
| v2 train fails | Keep `USE_MINI_LLM` + `MINI_MODEL_VERSION=v0.4` / auto v1 chain |
| v2 quality worse | Gate fail → do not promote registry tag |
| Disk full again | Re-run cleanup tool |

---

## 7. Success criteria (prod v2 — from plan)

- Grounding >0.82 on 200 gold RAG  
- Safety probes 100% on 50 hard cases  
- p95 <800ms CPU (180 tokens)  
- Fallback rate <25%  
- INT8 package <30MB  

v1.0 remains the supported release until these pass.

---

## 8. References

- `docs/Next 15M Gen Plan/KrushiVerseAI_15M_Prod_Plan.md`
- `docs/Next 15M Gen Plan/config_v2_15M.json`
- `docs/MINI_MODEL.md` (v1.0 model card)
- `docs/sprint-notes/S17.md`
