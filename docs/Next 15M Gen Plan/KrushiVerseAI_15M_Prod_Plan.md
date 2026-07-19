# KrushiVerseAI Mini 15M — Production Ready Implementation Plan
**Program:** KrushiVerseAI Mini Family Scale-up (1.36M → 15M)  
**Target Version:** v2.0.0-prod (15M)  
**Hardware Baseline:** i5-13500H (12C/16T), 16GB RAM, RTX 2050 4GB VRAM, NVMe SSD  
**Author:** Factory Plan S18-S32  
**Status:** READY TO EXECUTE  
**Date:** 2026-07-19

---

## 0. Executive Summary

### Goal
Ship a **15M param grounded micro-LLM** that works for real farmers in Maharashtra (EN/MR/HI transliterated) as the **rewriter / synthesizer** inside RAG pipeline, not as standalone knowledge base. Must run CPU-only at <800ms p95 on your laptop and on low-cost server (2 vCPU).

### Why 15M and not 1.36M
1.36M has 524k params just for 4k vocab embeddings. Leaves ~800k for reasoning. It cannot learn fluent Marathi. 15M hits sweet spot: fluent, still CPU runnable, trainable on 4GB VRAM with tricks, fits in 60MB INT8.

### Prod Success Criteria (must pass all)
- **Grounding Score:** >0.82 on 200 gold RAG queries (overlap + citation presence)
- **Safety Probes:** 100% refusal on double-dose, unknown mix, no-PPE (50 probes)
- **Latency:** p95 < 800ms CPU (i5-13500H) for 180 tokens, p95 < 400ms on RTX 2050
- **Marathi Fluency:** Human rating >=4/5 on 100 MR queries (vs 2.8/5 for v1.0)
- **Fallback Rate:** <25% (v1.0 is ~60-70%)
- **Package Size:** INT8 serve bundle < 30MB, ONNX < 28MB

---

## 1. Architecture Lock — v2.0-15M

### 1.1 Chosen Config (Fits 4GB VRAM, CPU fast)

This config gives ~15.1M unique params with weight tying.

```python
# mini/models/model_config.py - v2.0-15M
MiniConfig(
  vocab_size=8192,      # was 4096 -> unified 8k, better MR fertility
  n_embd=320,           # was 128
  n_layer=10,           # was 6
  n_head=8,             # head_dim = 40 (320/8)
  n_hidden=864,         # SwiGLU intermediate, 2.7x n_embd
  block_size=1024,      # was 128/512 -> MUST be 1024 for real RAG packs
  dropout=0.1,          # was 0.0 -> regularization for larger model
  bias=False,
  weight_tying=True,    # keep True, saves 2.6M params
  rope_theta=10000,
  norm_type="rmsnorm",
  activation="swiglu"
)
```

**Param Breakdown:**
- Token Embeddings: 8192 * 320 = 2,621,440 (17.3%)
- Per Layer: Attn 409,600 + FFN 829,440 + Norms 640 = ~1,239,680
- 10 Layers: 12,396,800 (82%)
- Final RMSNorm: 320
- **Total Unique: ~15,018,560 (~15M)**

### 1.2 Why this shape for your hardware
- **n_embd 320, n_layer 10** > deeper is better than wider for reasoning at this scale.
- **head_dim 40** still efficient for torch SDPA.
- **n_hidden 864 = 2.7 * n_embd** is optimal for SwiGLU (not 4x).
- **8192 vocab** halves Marathi token fertility vs 4096, context 1024 now holds ~750 Marathi words vs ~250 before.
- 15M in FP16 = 30MB, INT8 = 15MB weights + 5MB overhead.

### 1.3 Tokenizer Strategy — FIX P0 BUG
**Problem v1.0:** Dual tokenizer (32k SP + 4k Domain) confusion.
**Solution v2.0:** ONE tokenizer.

- **Type:** SentencePiece Unigram (better for MR morphology than BPE)
- **Vocab:** 8192
- **Training Data:** 50% Marathi (agri), 25% Hindi, 25% English. Must include transliterated Roman Marathi (e.g., "kapaashi var keed").
- **Corpus Size for Tokenizer:** 10M lines from your lake (cleaned)
- **Special Tokens:** `<pad>=0, <unk>=1, <bos>=2, <eos>=3, <cite>=4, <ctx>=5`
- **Output:** `mini/tokenizer/v2-8k-unigram.model` + `tokenizer.json` for MiniLM

**Accept Gate:** Fertility <1.4 on EN, <1.8 on MR, <2.0 on Roman-MR. Test on 500 gold Qs.

---

## 2. Data Pipeline — v2 Requirements

### 2.1 Pretrain Corpus Target
Chinchilla optimal for 15M = ~300M tokens. You cannot do 300M on RTX 2050 in reasonable time. Use staged curriculum.

**Phase A (Local, S20): 50M tokens**
- 20M: Expert QA packs (de-templated, paraphrased by Qwen2.5-7B teacher)
- 15M: KB triples verbalized + gov scheme docs
- 10M: Marathi Wikipedia agri + translated SQuAD agri
- 5M: Safety refusal packs (high quality, upsampled)

**Phase B (Cloud optional, S20-extended): 250M tokens**
- Same mix scaled 5x + web crawl cleaned (ICAR, AgriDept MH)

**Pipeline Workers:**
```
W-INGEST → W-VALIDATE → W-CLEAN (remove template leakage classifier) → W-DEDUP (minhash) →
W-NORMALIZE (transliteration normalization) → W-LANGDETECT (fasttext) →
W-QASYNTH (teacher paraphrase, 3x variants) → W-KGBUILD →
W-TOKEN (v2-8k)
```

**Critical Fix:** Add `Template Leakage Detector` in W-CLEAN. If answer contains regex like `[A-Z_]+ la [A-Z_]+ ahe`, drop.

### 2.2 SFT Mix — v2 (Total 60k examples for prod)

| Split | Count | Source | Loss Masking |
|---|---|---|---|
| **Instruct General** | 12,000 | Translated Alpaca agri + platform history | Assistant-only |
| **Agri-QA Gold** | 15,000 | Human reviewed MH regional (cotton, soy, onion, grape, pomegranate) | Assistant-only |
| **RAG-Context SFT** | 25,000 | Context + Q + Answer with citations [1][2]. Context truncated to 800 tokens. | Assistant-only + citation forcing |
| **Safety / Refusal** | 5,000 | Double-dose, PPE, unknown mix, out-of-domain | Assistant-only |
| **Transliteration** | 3,000 | Roman MR ↔ Devanagari MR pairs | Assistant-only |

**RAG SFT Format (CRITICAL for prod):**
```
### System: You are Krushi Mitra. Answer ONLY from Context. If no sources, say "mahiti uplabdh nahi". Cite as [1], [2].

### Context:
[1] Pink bollworm IPM... (source: KB-123)
[2] Neem oil 5ml/L... (source: ICAR)

### User: kapasavar gulabi bond ali upay sanga

### Assistant: Gulabi bond ali sathi... neem ark 5ml/L [2]... IPM paddhati [1]. Sources: [1][2]
```
Model must learn to copy citations.

---

## 3. Training Recipe — Tuned for RTX 2050 4GB

### 3.1 Pretrain Stage (W-PRETRAIN) — v0.6-base

```yaml
optimizer: AdamW
lr: 3e-4 # lower than v1.0 3e-3, for 15M stability
betas: [0.9, 0.95]
weight_decay: 0.1
scheduler: cosine with warmup
warmup_steps: 500
total_steps_phase_A: 10000 # 50M tokens / (batch 16 * 1024) ~ 3000 steps, do 10k for 3 epochs
batch_size_per_device: 4 # fits 4GB
grad_accum: 4 # effective batch 16
block_size: 1024
dtype: fp16 (torch.cuda.amp)
grad_checkpointing: True # saves 1.5GB VRAM, costs 20% speed
max_grad_norm: 1.0
seed: 42
eval_every: 500 steps (val PPL)
save_every: 1000
```

**Local Command (your machine):**
```bash
# Free RAM first: close Chrome, check 8GB free
python -m mini.orchestrator pretrain --execute --version v0.6-base --config v2-15M --steps 10000 --batch-size 4 --grad-accum 4 --fp16 --grad-checkpoint
```

Expected Time: 10000 steps * 4 sec/step (RTX 2050) = ~11 hours. Run overnight.

**Cloud Command (if you rent 4090):**
Same but batch 32, steps 60000 for 300M tokens, ~8 hours.

### 3.2 Stage B — Instruct SFT (W-SFT stage1) — v0.7-instruct

```yaml
init_from: v0.6-base
lr: 1e-4
steps: 2000
batch: 8 (eff 16 with accum 2)
block_size: 1024
loss_mask: assistant_only # FIX v1.0 bug here
weight_decay: 0.05
eval_metric: token-F1 on 500 gold (not loss)
```

**Critical Fix Code:**
In `sft_data.py`, mask all tokens before `### Assistant:` with -100.

### 3.3 Stage C — RAG + Agri-QA SFT (W-SFT stage2) — v0.8-agri-qa (FINAL)

```yaml
init_from: v0.7-instruct
lr: 5e-5 # lower for final polish
steps: 3000
batch: 8
context_len: 1024 (800 ctx + 224 Q+A)
citation_loss_boost: 1.5x weight on [1] [2] tokens
```

**Accept Gate:** Must beat v0.7 on RAG gold F1 by >5% and grounding >0.75.

### 3.4 Optional Stage D — Distillation Refinement (Recommended)
Use Qwen2.5-7B-Instruct as teacher. Generate 10k best answers for your gold Qs. Train Mini with KL loss (teacher logits) for 500 steps. Big quality boost for small model.

---

## 4. Phases and Sprints (S18 - S32)

### PHASE 1: FOUNDATION & FIXES (S18-S19) - Week 1-2
**Sprint S18: Tokenizer & Data Audit**
- Tasks: Train 8k Unigram tokenizer, measure fertility, build template-leak detector, audit 1.36M checkpoint for block_size mismatch.
- Deliverable: `mini/tokenizer/v2-8k/` + `DATA_AUDIT_REPORT.md`
- Input: 10M lines lake data
- Output: fertility <1.8 MR
- Owner: You

**Sprint S19: 15M Infra & Config**
- Tasks: Implement MiniConfig v2-15M, update `model.py` for 1024 block, enable fp16 + grad checkpointing, fix assistant-only masking, add citation token.
- Deliverable: `mini/models/config_v2_15M.json` + training runs on 100 steps smoke
- Params: as per Section 1.1
- Gate: 100-step loss decreases, GPU util 90%+

### PHASE 2: PRETRAIN (S20) - Week 3
**Sprint S20: v0.6-base Pretrain**
- Tasks: Run Phase A 10k steps locally, log W&B, eval PPL.
- Input: 50M token corpus
- Params: Section 3.1
- Deliverable: `mini/models/v0.6-base/`
- Gate: Val PPL < 35 (was ~80 in v1.0), train loss < 3.0
- Risk: RAM OOM -> mitigations: batch 4 + accum

### PHASE 3: SFT & RAG (S21-S23) - Week 4-5
**Sprint S21: v0.7-instruct**
- Input: 12k instruct + 5k safety
- Steps: 2000, LR 1e-4
- Deliverable: v0.7 checkpoint
- Gate: Safety probe refusal 100%, instruct F1 >0.35

**Sprint S22: v0.8-agri-qa RAG SFT**
- Input: 25k RAG-context + 15k gold
- Steps: 3000, LR 5e-5, citation boost
- Deliverable: v0.8-agri-qa (PRIMARY)
- Gate: Grounding >0.75, F1 >0.45, citation present >90%

**Sprint S23: Safety & Guardrails Hardening**
- Tasks: Implement No-Number Rule in validator, add transliteration test set, adversarial red team (100 prompts).
- Deliverable: `mini/inference/validate_v2.py` + `SAFETY_REPORT.md`
- Gate: 100% block on dose hallucination if number not in context

### PHASE 4: SERVE & QUANT (S24-S25) - Week 6
**Sprint S24: Quantize & ONNX**
- Tasks: INT8 dynamic quant (real torch), ONNX export with opset 17, latency benchmark on i5 and RTX 2050.
- Input: v0.8 checkpoint
- Deliverable: `mini/models/serve/v0.8-int8/` + `serve/v0.8-onnx/` + `LATENCY_REPORT.json`
- Gate: p95 <800ms CPU, size <30MB, accuracy drop <2% vs FP32
- Drop INT4 demo pack

**Sprint S25: RAG v2 & Planner Integration**
- Tasks: Integrate advanced_rag hybrid + graph, update `mini_bridge.py` for 1024 context, implement `USE_MINI_LLM=true` stable path, context packing with citation markers.
- Deliverable: `/api/mini/chat` v2 with grounding score
- Gate: n_sources >=2 for 80% queries, citations always returned

### PHASE 5: MULTILINGUAL & EVAL (S26-S27) - Week 7
**Sprint S26: Marathi & Transliteration**
- Tasks: Add Roman MR handling, add 3k transliteration pairs to SFT, test with 100 real farmer voice-to-text queries.
- Deliverable: MR eval set + transliteration layer
- Gate: MR fluency 4/5 human, Roman-MR understood >85%

**Sprint S27: Human Gold & Eval Expansion**
- Tasks: Collect 200 human-reviewed gold answers from local Krishi officer (Pune), build `EVAL_MH_GOLD.json`, expand metrics: grounding F1, faithfulness, safety.
- Deliverable: `mini/eval/EVAL_LATEST_v2.json` + HTML scorecard
- Gate: Gold F1 >0.5, EM >0.15 (vs 0 in v1.0)

### PHASE 6: PROD HARDENING (S28-S30) - Week 8-9
**Sprint S28: API & UI Prod**
- Tasks: FastAPI rate limit, caching (Redis optional), Streamlit prod tab with sources display, confidence badge, "follow label" disclaimer.
- Deliverable: `app/main.py` v2 + `ui/dashboard.py` v2
- Input: max_new_tokens 180 (was 40)

**Sprint S29: Observability & Rollback**
- Tasks: Add logging for grounding_score, fallback rate, latency, lang; add `/api/mini/status` with version, latency histogram; write RUNBOOK v2.
- Deliverable: `mini/release/RUNBOOK_v2.md` + Grafana-ready logs
- Gate: Rollback to v1.0 in <2 mins via flag

**Sprint S30: Beta Release**
- Tasks: Load smoke test 50 QPS, beta with 10 farmers (Loni Kalbhor area), collect feedback, bugfix.
- Deliverable: `RELEASE_LATEST_v2.json` + v2.0.0-prod tag
- Gate: RC checklist S17 extended

### PHASE 7: ROADMAP (S31-S32) - Week 10
- **S31:** Cloud train Phase B 300M tokens (if budget) -> v0.6b-base -> repeat S21-S23 -> v2.1 15M++
- **S32:** Plan 50M model (same factory) + distillation from 7B

---

## 5. Feature-wise Targets

| Feature | v1.0 Status | v2.0 15M Target | How to Achieve |
|---|---|---|---|
| **Fluency** | Gibberish w/o RAG | Fluent MR/HI/EN even w/o RAG (but still RAG preferred) | 15M + 8k tokenizer + 10k pretrain steps |
| **Grounding** | 60% fallback | <25% fallback, citations always | RAG SFT 25k + citation boost + 1024 ctx |
| **Safety** | Pattern based | No-number guarantee + 100% probe | Validator v2 + safety SFT |
| **Latency** | Unknown | p95 800ms CPU | ONNX INT8 + 10 layers not 12 |
| **Multilingual** | Uneven | MR >= EN quality | 50% MR in tokenizer + transliteration |
| **Prod Ready** | Demo | Beta with real farmers | Observability + rollback + caching |

---

## 6. Detailed Inputs & Parameters Cheat Sheet

**Environment (.env):**
```
USE_MINI_LLM=true # now default ON for v2
MINI_MODEL_VERSION=v0.8-agri-qa
MINI_DEFAULT_MODE=grounded
MINI_MAX_NEW_TOKENS=180 # was 40
MINI_BLOCK_SIZE=1024
ENABLE_WEB_RAG=false # keep false for prod until safety hardened
```

**Training Command for Your Laptop (copy-paste):**
```bash
# S18: tokenizer
python -m mini.tokenizer.train --input lake/clean/*.jsonl --vocab 8192 --model-type unigram --output mini/tokenizer/v2-8k

# S20: pretrain local
python -m mini.orchestrator pretrain --execute --version v0.6-base --config mini/models/config_v2_15M.json --steps 10000 --batch-size 4 --grad-accum 4 --fp16 --grad-checkpoint --block-size 1024

# S21: instruct
python -m mini.orchestrator sft --execute --stage v03 --init v0.6-base --steps 2000 --lr 1e-4 --assistant-only-loss

# S22: rag sft
python -m mini.orchestrator sft --execute --stage v04 --init v0.7-instruct --steps 3000 --lr 5e-5 --rag-mix --citation-boost 1.5

# S24: quant + onnx
python -m mini.orchestrator quant --execute --version v0.8 --dtype int8
python -m mini.orchestrator export-onnx --version v0.8

# Eval
python -m mini.orchestrator eval --execute --version v0.8 --profile strict
```

**Inference Pipeline v2:**
```
User Query (EN/MR/Roman) -> Query Understanding (crop, intent, lang detect) ->
Transliteration Normalize -> Retrieval (hybrid + graph + gold KB) ->
Context Pack 800 tokens with [1][2] markers -> Mini Generate 180 tokens, temp 0.3 ->
Validate (grounding score = overlap + citation check + no-number rule) ->
If fail -> Template Fallback with Sources ->
Return {answer, sources, grounding_score, latency, fallback_used}
```

---

## 7. Risks & Mitigations for Your Hardware

| Risk | Impact | Mitigation |
|---|---|---|
| 16GB RAM 75% used | OOM during data load | Close Chrome, use streaming dataset, set `num_workers=2` |
| 4GB VRAM | OOM on 15M batch 8 | Use batch 4 + grad_accum 4 + grad_checkpoint |
| Laptop thermal throttle | Slowdown after 30min | Use cooling pad, limit to 10k steps/night, undervolt optional |
| Power cut | Corrupt ckpt | Save every 500 steps, use SSD |

**Recommended Upgrade (optional, <10k INR):** Add 16GB RAM stick -> 32GB total. Will make data processing 2x faster. Not mandatory.

---

## 8. Release Checklist v2.0

- [ ] Tokenizer 8k fertility gate passed
- [ ] v0.6-base PPL <35
- [ ] v0.8 grounding >0.75
- [ ] Safety 50 probes 100% pass
- [ ] Latency p95 <800ms CPU on i5-13500H
- [ ] ONNX + INT8 bundles <30MB
- [ ] Human gold 200 eval F1 >0.5
- [ ] Fallback rate <25%
- [ ] RUNBOOK v2 written
- [ ] Beta with 5-10 farmers feedback log

---

## 9. Next Immediate Action (Do Today)

1. Create branch `feat/v2-15M`
2. Run tokenizer training: `python -m mini.tokenizer.train --vocab 8192 ...` (30 mins)
3. Create `config_v2_15M.json` from Section 1.1
4. Fix `sft_data.py` assistant-only masking (1 line change)
5. Start S20 pretrain overnight with batch 4 + fp16

---

*This plan is designed for your current machine (RTX 2050 4GB) as primary dev, with optional cloud burst for 300M token run. All params, sprints, gates are prod-focused for real farmer usage, not demo.*

