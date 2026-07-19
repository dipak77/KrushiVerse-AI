# KrushiVerseAI Mini — Complete Model Guide

**Product name:** KrushiVerseAI Mini (Krushi Mitra text brain)  
**Program version:** **1.0.0** (Sprint 17 / FP-10)  
**Role:** ~1.4M-parameter agriculture language model for Maharashtra-focused farming Q&A, designed to run **with retrieval (RAG)** and platform specialist agents — not as a standalone “ChatGPT replacement.”

This document describes **what Mini is**, **how it was built**, **how it works**, **how to use it**, **technical specs**, **strengths / weaknesses**, and **what to improve next**.

---

## 1. What is this model?

**Mini** is a **tiny decoder-only Transformer** trained inside the KrushiVerseAI **Mini factory** (`mini/`). It is the text generation component of an agriculture assistant stack that also includes:

| Layer | What it does |
|---|---|
| **Data lake** | Ingest → validate → clean → dedup → schema v1 train/val/test |
| **QA synth + KG** | Expert-style QA packs + knowledge graph |
| **Tokenizer** | Domain SentencePiece (30–50k) + small DomainTokenizer for Mini (4k) |
| **Mini LM** | ~1.36M params — pretrain → instruction SFT → agri-QA SFT |
| **Eval / quant / deploy** | Gates, INT8/INT4 packages, local serve registry |
| **Inference** | Intent → retrieve → Mini generate → validate → optional template fallback |
| **Platform** | FastAPI `/api/mini/chat`, optional planner synthesizer via `USE_MINI_LLM` |

### One-sentence pitch

> Mini is a **grounded, factory-built micro-LLM** for agri Q&A that **prefers retrieved context and safety rules** over free-form guessing.

### What it is *not*

- Not a general-purpose chatbot (world knowledge, coding, long creative writing).
- Not a licensed agronomist or a substitute for **pesticide labels / soil tests / local officers**.
- Not a vision-language model (vision stays in separate platform agents).
- Not production “frontier” quality — by design it is **small, CPU-runnable, and RAG-coupled**.

---

## 2. Identity & version ladder

| Tag | Meaning | Sprint |
|---|---|---|
| **v0.1** | Domain tokenizer family | S9 |
| **v0.2-base** | Domain language-model pretrain | S11 |
| **v0.3-instruct** | Instruction + safety SFT | S12 |
| **v0.4-agri-qa** | Agri-QA + RAG-context SFT (primary QA checkpoint) | S12 |
| **v0.5-quant** | INT8 / INT4 packages + serve bundle | S14 |
| **v0.5-reasoning-lite** | Optional short-rationale packaging tag | S14 |
| **v0.6+** | RAG-coupled serving path (code + infer) | S15 |
| **v0.8** | Multi-agent: Mini as planner synthesizer (flag) | S16 |
| **v0.9 / v1.0** | Prod beta tooling + program release | S17 |

**Recommended runtime weights:** `v0.4-agri-qa` or packaged `serve/v0.5-quant` (auto-pick tries serve → v0.4 → v0.3 → v0.2).

Checkpoints live under `mini/models/` and are **local-only** (gitignored). Source code and factory workers are in git.

---

## 3. Architecture (technical)

### 3.1 Model class

- **Class name:** `MiniLM` (`mini/models/model.py`)
- **Type:** Causal (decoder-only) language model
- **Objective:** Next-token prediction (cross-entropy; pad ignored)

### 3.2 Hyperparameters (v1.0 lock)

| Hyperparameter | Value | Notes |
|---|---|---|
| Approx. parameters | **~1.36M** (`unique_params` 1,361,536) | Target band 0.8–1.5M |
| Vocab size (model) | **4096** | DomainTokenizer for LM; separate 32k SP for factory |
| Embedding dim `n_embd` | **128** | |
| Layers `n_layer` | **6** | |
| Attention heads `n_head` | **4** | head_dim = 32 |
| FFN intermediate `n_hidden` | **192** | SwiGLU |
| Context `block_size` | **512** default; many trained ckpts use **128** | Always clamp gen/prompt to loaded config |
| Dropout | 0.0 | |
| Bias on Linear | False | |
| Weight tying | **True** (tok emb ↔ lm_head) | Critical for ~1M budget |
| Positional encoding | **RoPE** (θ=10000) | |
| Normalization | **RMSNorm** | |
| Activation (FFN) | **SwiGLU** | |
| Attention | Causal scaled-dot-product | |

### 3.3 Design choices (why this shape)

1. **~1M params** — fits research/demo on CPU; fast iteration in a monorepo factory.  
2. **Vocab 4096 for the LM** — embeddings dominate param count; full 32k SP is kept for the *data* tokenizer track without exploding Mini size.  
3. **RoPE + RMSNorm + SwiGLU + tying** — modern “small LLM” defaults without extra libraries.  
4. **RAG-first product design** — tiny models hallucinate; v1.0 product path is **grounded mode** (answer from retrieved context + citations).

### 3.4 Tokenization

| Component | Role |
|---|---|
| **SentencePiece BPE ~32k** (`mini/tokenizer/`) | Domain corpus tokenizer (W-TOKEN, S9) |
| **DomainTokenizer vocab 4096** (`mini/models/corpus.py`) | Word/piece-style fit for Mini pretrain/SFT/infer |

Inference encoding for Mini uses the **checkpoint’s** `tokenizer.json` (DomainTokenizer saved with the model).

Special IDs (default): `pad=0`, `unk=1`, `bos=2`, `eos=3`.

### 3.5 Training prompt format (SFT / infer)

```text
### System:
<system instructions>

### User:
<question or Context: ... Question: ...>

### Assistant:
<answer>
```

**Systems in use:**

- **Instruct:** Krushi Mitra helper; IPM / soil-test preference.  
- **RAG:** Answer **only** from context; refuse to invent chemicals.  
- **Safety:** Refuse double-dosing, unknown mixes, skipping PPE, etc.

---

## 4. How it was trained (end-to-end factory)

Training is **not** a single Hugging Face script. It is a **worker pipeline** (`W-*`) orchestrated by `python -m mini.orchestrator`.

### 4.1 Data pipeline (before any gradient step)

```text
W-INGEST → W-VALIDATE → W-CLEAN → W-DEDUP
  → W-NORMALIZE / W-LANGDETECT / W-STANDARD
  → W-ANALYZE
  → W-QASYNTH   (≥50k train-scale QA target)
  → W-KGBUILD   (≥200 nodes / ≥400 edges target)
  → W-TOKEN
```

Rules:

- **Never train from `raw/`** — only versioned standard records under training/validation/test (and factory datasets).  
- Schema: **StandardRecord** v1 (`mini/contracts.py`) — category, crop, region, language (mr/hi/en), question, answer, provenance, split.

Data sources (conceptual mix):

- Open / platform KB seeds, government-style scheme/market patterns, disease/pest/crop packs  
- Synthetic expert QA expansion (templates + lake facts)  
- Curated **gold** sets for eval (disease, fertilizer, schemes, market, MH regional)  
- Safety / hard-negative style refusal pairs  

Languages: **English, Marathi, Hindi** (quality is uneven at Mini scale).

### 4.2 Stage A — Domain pretrain (v0.2-base)

| Item | Detail |
|---|---|
| Worker | `W-PRETRAIN` |
| Code | `mini/models/pretrain.py`, `corpus.py` |
| Data | Packed agri text blocks from lake QA, KB, tokenizer corpus, KG triples |
| Objective | Causal LM (next token) |
| Accept | Val PPL improves (or strong train-loss drop); seed reproducible |
| Output | `mini/models/v0.2-base/` |

Typical short training knobs (factory defaults / CI-scale): AdamW, LR ~3e-3, batch 4–8, hundreds of steps in smoke; longer runs locally as needed.

### 4.3 Stage B — Instruction SFT (v0.3-instruct)

| Item | Detail |
|---|---|
| Worker | `W-SFT` (stage 1) |
| Code | `mini/models/sft.py`, `sft_data.py` |
| Mix | Multilingual QA + **upsampled safety** refusals |
| Start | Load v0.2-base when present |
| Output | `mini/models/v0.3-instruct/` |

### 4.4 Stage C — Agri-QA + RAG-context SFT (v0.4-agri-qa)

| Item | Detail |
|---|---|
| Worker | `W-SFT` (stage 2) |
| Mix | Full train mix + **RAG-context** examples (“answer from context only”) |
| Accept | Val token-F1 / loss **beats base** (v0.2) under factory rules |
| Output | `mini/models/v0.4-agri-qa/` + `SFT_LATEST.json` |

SFT uses teacher-forced LM loss on full formatted dialogues; evaluation also runs **greedy/short generation** for token-F1 / exact match.

### 4.5 Stage D — Evaluation (W-EVAL)

| Metric family | Examples |
|---|---|
| Generation quality | token-F1, EM, ROUGE-L, keyword hit on gold |
| LM | teacher-forced loss / PPL on gold-style sequences |
| Ops | latency p95, optional RSS |
| Safety | hallucination / safety probes (dose doubling, unknown mixes, fact conflict) |
| Gates | `default` (CI-soft) vs `strict` (promotion-hard) |

Reports: `mini/eval/EVAL_LATEST.json` + HTML scorecard.

### 4.6 Stage E — Quantize & package (W-QUANT / W-DEPLOY)

| Artifact | Meaning |
|---|---|
| FP32 snapshot | Baseline size/latency |
| **INT8** dynamic quant (Linear) | Deploy-oriented; budget ~4 MiB weights |
| **INT4** packed weights | Size demo / research pack (not full inference engine) |
| Serve package | `mini/models/serve/<tag>/` + model card, license, provenance |
| Registry | `VERSION_REGISTRY.json` |

### 4.7 Stage F — Product integration & release

| Sprint | Deliverable |
|---|---|
| S15 | Grounded infer chain + citations + validator + template fallback |
| S16 | `/api/mini/chat`, `USE_MINI_LLM` planner synthesizer, Streamlit Mini panel |
| S17 | RC gate, checklist sign-off, load smoke, version matrix, scale roadmap, **v1.0** |

---

## 5. How it works at runtime (inference)

### 5.1 Grounded chain (default product path)

```text
User query
   │
   ├─► Query understanding (crop, intent, language hint)
   ├─► Retrieval
   │      • Platform advanced_rag (hybrid + graph + tools ± web)
   │      • Mini-local lexical KB / gold facts (always available offline)
   ├─► Context pack with citation markers [1], [2], …
   ├─► Optional agent notes (disease / weather / fertilizer specialists)
   ├─► Mini generate (SYSTEM_RAG prompt)
   ├─► Validate
   │      • grounding score (overlap + citation markers)
   │      • banned unsafe advice patterns
   └─► If fail → template fallback from sources
              → Always attach **Sources** in grounded mode
```

**Hard rule (grounded mode):**  
If there are **no sources**, Mini **must not** invent an answer — it refuses and asks for better retrieval/query.

### 5.2 Open mode

Less strict about sources; still not a general LLM. Prefer grounded for any farmer-facing demo.

### 5.3 Planner integration

| Mode | Behavior |
|---|---|
| `USE_MINI_LLM=false` (**default**) | Classic multi-agent + **template** synthesizer (old platform behavior) |
| `USE_MINI_LLM=true` | Agents still run as **tools**; **Mini** synthesizes the final answer via Mini bridge |
| `POST /api/mini/chat` | **Always** Mini chain (not gated by the flag) |

---

## 6. How to use

### 6.1 Prerequisites

- Python venv with project deps (PyTorch, FastAPI, etc.)
- Local checkpoints if you want real Mini weights (train or copy `v0.2`/`v0.4` under `mini/models/`)
- Without weights, factory falls back to random-init Mini for some paths — **quality will be noise**; train first for demos

### 6.2 Factory CLI

```bash
# From repo root
python -m mini.orchestrator list-workers
python -m mini.orchestrator status

# Train (examples — long on CPU)
python -m mini.orchestrator pretrain --execute --mode domain --steps 200 --seed 42
python -m mini.orchestrator sft --execute --steps-v03 120 --steps-v04 120 --seed 42

# Evaluate & release
python -m mini.orchestrator eval --execute --version v0.4 --profile default
python -m mini.orchestrator quant --execute --version v0.4
python -m mini.orchestrator deploy --execute --tag v0.5-quant --force
python -m mini.orchestrator release --execute

# Inference
python -m mini.orchestrator rag --execute --query "pink bollworm cotton IPM"
python -m mini.orchestrator infer --execute \
  --query "How do I manage pink bollworm in cotton with IPM in Maharashtra?"
```

### 6.3 HTTP API

```bash
# Status (flags + version)
GET /api/mini/status

# Mini chat (always Mini+RAG)
POST /api/mini/chat
Content-Type: application/json

{
  "query": "How do I manage pink bollworm in cotton with IPM?",
  "crop": "Cotton",
  "location": "Pune",
  "mode": "grounded",
  "language": "en",
  "enable_web": false,
  "enable_agents": true,
  "max_new_tokens": 40
}

# Full multi-agent planner (Mini only if USE_MINI_LLM=true)
POST /api/query
{ "query": "...", "farm_id": "FARM_101", "language": "en" }

# Release gate
POST /api/mini/release?execute=true
```

### 6.4 Streamlit

```bash
streamlit run ui/dashboard.py
```

Open the **“Mini LLM + Citations”** tab. Sidebar shows `USE_MINI_LLM` flag state.

### 6.5 Environment flags

| Variable | Default | Meaning |
|---|---|---|
| `USE_MINI_LLM` | `false` | Mini as planner synthesizer |
| `MINI_DEFAULT_MODE` | `grounded` | Prefer cited answers |
| `MINI_MAX_NEW_TOKENS` | `40` | Generation length budget |
| `MINI_MODEL_VERSION` | `auto` | Checkpoint selection |
| `ENABLE_WEB_RAG` | `true` | Web in platform RAG (Mini chat often forces off for demos) |

### 6.6 Rollback

1. Set `USE_MINI_LLM=false` → classic synthesizer.  
2. Point `MINI_MODEL_VERSION` to an older local tag.  
3. Redeploy prior `mini/models/serve/<tag>` if packaged.

---

## 7. Strengths (what is good)

1. **Right-sized for the problem class** — agri micro-assistant demos on CPU without cloud GPU.  
2. **Factory reproducibility** — every stage is a worker + tests + sprint notes (S0–S17).  
3. **RAG-native product design** — grounded mode + citations reduce open-ended hallucination for farmer UX.  
4. **Safety scaffolding** — refusal templates, banned-pattern validator, eval probes for double-dose / PPE / unknown mixes.  
5. **Maharashtra agri domain focus** — crops, pests, schemes, mandi language appear in data/taxonomy.  
6. **Multilingual intent** — EN/MR/HI present in packs (even if generation quality is limited).  
7. **Full MLOps-lite loop** — pretrain → SFT → eval gates → quant → package → chat API → release checklist.  
8. **Non-breaking platform integration** — flag default **off** preserves existing planner behavior.  
9. **Clear scale path** — same workers for 10M+ later (`proceed_with_family_scale`).  
10. **Observable** — `EVAL_LATEST`, `INFER_LATEST`, `SFT_LATEST`, `RELEASE_LATEST`, HTML scorecard.

---

## 8. Weaknesses (what is bad / limited)

1. **Capacity wall** — ~1.36M params cannot hold broad factual knowledge; answers without RAG are often **gibberish or generic loops**.  
2. **Generation quality** — even after SFT, free generation F1 on gold is **low**; demos often use **template fallback** after validation.  
3. **Exact match ~0** on gold — model rarely reproduces full expert answers verbatim.  
4. **Context length** — many local ckpts trained at **block_size 128**; long RAG prompts are truncated.  
5. **Multilingual unevenness** — Devanagari answers weaker than English scaffolding.  
6. **Tokenizer split** — 32k SP vs 4k DomainTokenizer adds complexity; fertility on rare agri terms still limited.  
7. **Training scale** — factory steps are **research/CI-scale**, not multi-epoch GPU pretrain on millions of tokens.  
8. **INT4 path** is a **weight packing demo**, not a complete optimized runtime.  
9. **Not human-verified at 1M-QA scale** — synth packs dominate; residual errors and template-y answers.  
10. **No true reasoning / tool-use inside the LM** — tools and agents are **outside** the model.  
11. **Safety is pattern-based**, not guaranteed against all adversarial prompts.  
12. **Weights not shipped in git** — each environment must train or supply checkpoints.

---

## 9. Evaluation snapshot (how we judge quality)

| Layer | Signals |
|---|---|
| Pretrain | Train loss ↓, val PPL ↓, seed match |
| SFT | token-F1 / loss vs base; demos in `SFT_LATEST` |
| W-EVAL | Gold F1, ROUGE-L, keyword hit, probes, latency p95, gates |
| Infer | `ok`, `n_sources`, `used_fallback`, citations present |
| Release | Checklist §8, eval gate, load smoke p95 |

**Interpretation for stakeholders:**  
At v1.0, success means **the system** (RAG + Mini + validator + fallback) can answer **grounded agri demos with sources**, not that Mini alone matches a 7B model on open QA.

---

## 10. Repository map (where code lives)

```text
mini/
  models/          MiniConfig, MiniLM, pretrain, sft, quantize, deploy
  tokenizer/       SentencePiece train (domain)
  eval/            gold, metrics, probes, gates, harness
  inference/       RAG wrap, validate, fallback, pipeline
  release/         v1.0 checklist, matrix, smoke, runbook
  workers/         W-INGEST … W-RELEASE
  orchestrator/    CLI + DAG
  lake/            ingest/quality/standardize/qa_synth/kg_build
  taxonomy/        frozen domain taxonomy
app/
  llm/mini_bridge.py   platform ↔ Mini
  llm/generator.py     template + optional Mini synthesizer
  agents/planner.py    multi-agent; USE_MINI_LLM
  knowledge/           advanced_rag, query_understanding
  main.py              /api/mini/chat, /api/mini/status, /api/mini/release
ui/dashboard.py        Streamlit Mini panel
docs/
  KRUSHIVERSE_MINI_SPRINT_PLAN.md
  sprint-notes/S00…S17.md
  MINI_MODEL.md         ← this file
```

---

## 11. Security, safety & compliance notes

- Prefer **grounded** answers with citations for user-facing content.  
- Never treat Mini output as a **pesticide prescription**; always “follow the label / local officer.”  
- Validator blocks common unsafe phrasings; still review high-risk chemical advice.  
- Web RAG can pull untrusted text — keep `enable_web=false` for locked demos.  
- Training data may include synthetic content — disclose in model cards (`serve/*/MODEL_CARD.json`).

---

## 12. Next improvements (post-v1.0 roadmap)

### Near-term (quality at same ~1M–10M scale)

1. **Longer, cleaner pretrain** on packed agri corpus (more steps, curriculum).  
2. **Larger SFT mix** with human-reviewed MR/HI slices; reduce pure template leakage.  
3. **Assistant-only loss masking** (train on answer tokens, not full prompt) for better SFT.  
4. **Longer context** (train at 256–512 consistently) for real RAG packs.  
5. **Unified tokenizer** strategy (resize emb carefully or distill SP→Domain).  
6. **Stronger grounding training** (more RAG-context SFT + citation forcing).  
7. **Better decoding** (constrained decode, stop sequences, lower temperature for facts).  
8. **Eval expansion** — regional MH gold, scheme accuracy, adversarial safety suite.  
9. **Serve path** — real INT8 runtime latency on reference CPU; optional ONNX.  
10. **Product UX** — always show sources, confidence, and “needs more info” states.

### Scale ladder (program decision)

Same factory family (do not fork pipelines):

| Target | Focus |
|---|---|
| **10M** | Wider emb/layers; same data/workers |
| **50M** | Longer context; stronger MR/HI; optional LoRA |
| **100M+** | Multi-GPU; optional distillation |
| **300M–1B** | Separate infra/funding track |

### Explicitly deferred (not v1.0)

- Full 1M human-verified QA  
- Neo4j production cluster  
- Vision-language Mini  
- Multi-GPU 100M+ training now  
- Kubernetes HA serve  

---

## 13. Quick FAQ

**Q: Can farmers rely only on Mini’s free text?**  
**A:** No. Use **grounded mode + sources**, and follow labels/officers for chemicals.

**Q: Why is the answer sometimes a structured template?**  
**A:** Validator rejected low-grounding or unsafe Mini text; **fallback synthesizer** rebuilt a safe answer from retrieved snippets.

**Q: Where are the weight files?**  
**A:** Local under `mini/models/` (gitignored). Train with orchestrator or restore from your artifact store.

**Q: How do I turn Mini on for the main assistant?**  
**A:** `USE_MINI_LLM=true` and restart API/Streamlit. Or call `/api/mini/chat` directly.

**Q: Is this open-source ready as a “model release”?**  
**A:** Code and factory yes; weights are environment-local. Ship a model card + license with any public weight dump.

---

## 14. Summary table

| Dimension | Mini v1.0 |
|---|---|
| Size | ~**1.36M** parameters |
| Type | Decoder-only Transformer (RoPE, RMSNorm, SwiGLU, tied emb) |
| Domain | Maharashtra / India agri assistant |
| Best used as | **RAG + tools + validators**, not open chat |
| Primary ckpt | **v0.4-agri-qa** (or quant serve package) |
| Product API | `POST /api/mini/chat` |
| Flag | `USE_MINI_LLM` (default off) |
| Program status | **Released 1.0.0** after S0–S17 factory |

---

## 15. Related docs

| Doc | Purpose |
|---|---|
| [`docs/KRUSHIVERSE_MINI_SPRINT_PLAN.md`](KRUSHIVERSE_MINI_SPRINT_PLAN.md) | Full program plan & success criteria |
| [`docs/sprint-notes/`](sprint-notes/) | Per-sprint delivery notes S00–S17 |
| [`mini/README.md`](../mini/README.md) | Factory CLI quickstart |
| [`mini/release/RUNBOOK.md`](../mini/release/RUNBOOK.md) | Ops / rollback (after `release --execute`) |
| Serve `MODEL_CARD.json` | Per-package card when deployed locally |

---

*Document version: aligned with Mini program **1.0.0** (Sprint 17). Update this file when architecture, training recipe, or product flags change.*
