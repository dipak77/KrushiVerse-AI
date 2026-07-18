# KrushiVerseAI Mini (1M) — Sprint & Feature Phase Implementation Plan

**Document version:** 1.0  
**Platform version baseline:** KrushiVerse-AI v10.2 (agents + multi-source RAG + GraphRAG + open data)  
**Goal:** Build a **reusable agriculture AI factory** (data → train → eval → serve) whose first product is a **~1M-parameter domain Mini LLM**, designed to scale to 10M → 100M → 1B without rewiring the platform.  
**Principle:** The Mini model is **one worker** in an automated multi-worker system — never the sole intelligence.

---

## 0. Executive summary

| Item | Decision |
|---|---|
| North star | Farmer-usable agricultural answers in **Marathi / Hindi / English**, grounded by RAG + KG + tools |
| First shippable model | **KrushiVerseAI Mini ~1M params** (decoder-only) + **mandatory RAG** |
| Org model | **Automated Worker Modules** that hand off artifacts via a shared data contract |
| Delivery style | **6 Program Epics → 18 Sprints (2 weeks each) ≈ 9 months** to v1.0 Mini, with earlier demos every 4–6 weeks |
| What we already have | Multi-agent planner, hybrid+dense RAG, GraphRAG seed, open data, Streamlit, FastAPI, 213 KB docs |
| What we build next | Data lake, workers, tokenizer, train/eval pipelines, Mini model, inference chain, versioning |

### Critical design rule

```
1M Mini alone  →  high hallucination risk
1M Mini + Hybrid RAG + Graph + Tools + Validator  →  useful product
```

Every sprint must leave the system **more automatable**, not only “more accurate once.”

---

## 1. Current baseline vs target

### 1.1 Already in repo (reuse — do not rebuild)

| Capability | Location | Reuse as |
|---|---|---|
| Multi-agent planner | `app/agents/` | Expert Agent workers + routing |
| Hybrid + dense RAG | `app/knowledge/` | Retrieval Layer (Phase 14) |
| GraphRAG seed | `app/knowledge/graph_rag.py` + `data/knowledge_graph.json` | KG starter |
| Open data / Agmarknet | `app/live_feeds/opendata_client.py` | Market Engine worker |
| Template synthesizer | `app/llm/generator.py` | Interim “brain” until Mini is ready |
| FastAPI + Streamlit | `app/main.py`, `ui/dashboard.py` | Inference & ops UI |
| Farm memory | `app/memory/` | Memory worker seed |
| Predictive calculators | `app/predictive/` | Recommendation Engine helpers |

### 1.2 Greenfield modules (new package tree)

```
mini/                          # KrushiVerseAI Mini program root
├── taxonomy/                  # Phase 1 domain ontology
├── lake/                      # Phase 3 data lake layout + contracts
├── workers/                   # Automated worker modules (core of this plan)
│   ├── ingest/
│   ├── clean/
│   ├── standardize/
│   ├── analyze/
│   ├── qa_synth/
│   ├── kg_builder/
│   ├── tokenizer/
│   ├── train/
│   ├── eval/
│   ├── quantize/
│   └── deploy/
├── models/                    # model configs, checkpoints refs
├── tokenizer/                 # trained tokenizer artifacts
├── datasets/                  # versioned HF/parquet manifests (not raw dumps)
├── eval/                      # gold sets, reports
├── inference/                 # intent → retrieve → Mini → validate
└── orchestrator/              # worker DAG (Prefect/Airflow-compatible)
```

Platform packaging stays under `app/` for serving; `mini/` is the **ML factory**.

---

## 2. Automated Worker Architecture

Workers are **small, single-responsibility jobs** that read/write versioned artifacts.  
They communicate only through the **Artifact Bus** (filesystem + optional DB + event log).

```
                    ┌─────────────────────────────┐
                    │     Orchestrator (DAG)      │
                    │  Prefect / Airflow / CLI     │
                    └──────────────┬──────────────┘
                                   │ schedules / triggers
     ┌──────────┬──────────┬───────┼────────┬──────────┬──────────┐
     ▼          ▼          ▼       ▼        ▼          ▼          ▼
  Ingest     Clean    Standardize Analyze  QASynth   KGBuild   Tokenizer
     │          │          │       │        │          │          │
     └──────────┴──────────┴───────┴────────┴──────────┴──────────┘
                                   │
                          Artifact Bus (Parquet/JSON/SQLite)
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
                 Train          Eval          Quantize
                    │              │              │
                    └──────────────┴──────────────┘
                                   │
                              Deploy Worker
                                   │
                    ┌──────────────▼──────────────┐
                    │  Serving (FastAPI + RAG)     │
                    │  Mini + Retriever + Agents   │
                    └─────────────────────────────┘
```

### 2.1 Worker catalog (contracts)

| Worker ID | Input | Output | Owner epic | SLO |
|---|---|---|---|---|
| `W-INGEST` | Source registry URL/path | `lake/raw/{domain}/{date}/` + manifest | E1 | Idempotent, retryable |
| `W-VALIDATE` | Raw batch | Validation report + quarantine list | E1 | Fail batch if critical schema break |
| `W-CLEAN` | Raw + rules | `lake/processed/` cleaned rows | E1 | Deterministic given seed |
| `W-DEDUP` | Cleaned | Deduped + near-dup clusters | E1 | MinHash/LSH |
| `W-NORMALIZE` | Deduped | Canonical fields, units, crop IDs | E1 | Uses taxonomy |
| `W-LANGDETECT` | Text fields | `language` tags (mr/hi/en/mixed) | E1 | ≥95% on gold sample |
| `W-STANDARD` | Normalized | Standard QA/record schema | E2 | Schema v1 locked |
| `W-ANALYZE` | Standard dataset | Coverage/quality dashboard JSON | E2 | Runs every ingest cycle |
| `W-QASYNTH` | Structured facts + templates | Expert QA pairs (by domain) | E2 | Human-review sample queue |
| `W-KGBUILD` | Standard entities/relations | Graph export (NetworkX → Neo4j later) | E3 | Diff-based updates |
| `W-TOKEN` | Training text corpus | Tokenizer model + vocab stats | E4 | 30–50k vocab |
| `W-PRETRAIN` | Tokenized corpus | Base checkpoint v0.2 | E4 | Reproducible seed |
| `W-SFT` | Instruction/QA sets | Instruction model v0.3–v0.4 | E4 | Loss + eval gates |
| `W-EVAL` | Model + gold sets | Metrics report + gate pass/fail | E5 | Blocks bad deploys |
| `W-QUANT` | FP model | INT8/INT4 artifacts | E5 | Size/latency budgets |
| `W-DEPLOY` | Artifact + config | Served endpoint + version tag | E5 | Blue/green or file swap |
| `W-INFER` | User query | Validated answer + citations | E6 | p95 latency target |
| `W-RAG` | Query + filters | Context pack (reuse v10.2) | E6 | Existing advanced RAG |
| `W-AGENT` | Intent | Specialist agent outputs | E6 | Existing planner agents |

### 2.2 Shared data contract (Phase 5 — enforced early)

Every training/eval record **must** eventually match:

```json
{
  "id": "uuid",
  "category": "disease|pest|soil|weather|crop|fertilizer|irrigation|scheme|market|finance|machinery",
  "subcategory": "string",
  "crop": "Cotton|...|null",
  "region": {"state": "Maharashtra", "district": "Akola", "zone": "Vidarbha"},
  "language": "mr|hi|en|mixed",
  "question": "string",
  "answer": "string",
  "source": "string",
  "source_url": "string|null",
  "verified": false,
  "confidence": 0.0,
  "updated_at": "ISO-8601",
  "license": "string",
  "split": "train|val|test|holdout",
  "schema_version": "1.0"
}
```

**Rule:** No worker may train on records missing `schema_version` or `split`.

### 2.3 Orchestrator interfaces

| Interface | Purpose |
|---|---|
| `mini/orchestrator/dag.py` | Defines worker DAG (local runnable without Airflow) |
| `mini/orchestrator/cli.py` | `python -m mini.orchestrator run --pipeline ingest|train|eval|full` |
| `mini/orchestrator/events.jsonl` | Append-only run log |
| Optional later | Prefect/Airflow deploy of the same DAG functions |

---

## 3. Program structure: Epics → Phases → Sprints

Map original Phases 0–17 into **6 Epics** and **18 sprints** (2 weeks each).

```
Epic E0  Foundation & Taxonomy          Sprints 0–1     (Weeks 0–4)
Epic E1  Data Lake & Engineering        Sprints 2–4     (Weeks 5–10)
Epic E2  Expert Datasets & Analysis     Sprints 5–7     (Weeks 11–16)
Epic E3  Knowledge Graph Factory        Sprint  8       (Weeks 17–18)
Epic E4  Tokenizer + Mini Model Train   Sprints 9–12    (Weeks 19–26)
Epic E5  Eval, Quant, Version Gates     Sprints 13–14   (Weeks 27–30)
Epic E6  Inference Product Integration  Sprints 15–17   (Weeks 31–36)
```

Buffer / polish / scale planning: continuous + Sprint 17 closeout.

---

## 4. Feature phases (product capability targets)

Independent of sprint numbers — these are **user-visible feature gates**.

| Feature phase | Name | User-visible capability | Gate |
|---|---|---|---|
| **FP-0** | Scaffold | `mini/` package, CLI, empty lake, taxonomy stubs | Repo compiles, workers list |
| **FP-1** | Ingest-1 | Ingest open ICAR/gov/local JSON into lake | ≥5 domains raw batches |
| **FP-2** | Clean-1 | Validated, deduped, normalized standard records | Schema v1 locked |
| **FP-3** | QA-1 | First expert QA slices (synthetic + curated) | 10k train / 1k val |
| **FP-4** | KG-1 | Expanded graph from standard entities | ≥200 nodes, ≥400 edges |
| **FP-5** | Tok-1 | Domain tokenizer v0.1 | Vocab 30k+, fertility on agri terms |
| **FP-6** | Base-1 | Mini base model v0.2 trained | ~1M params, loss curves logged |
| **FP-7** | SFT-1 | Instruction/QA model v0.3–v0.4 | Gold QA EM/F1 improves vs base |
| **FP-8** | RAG+Mini | Mini only answers with retrieved context | Hallucination rate ↓ vs no-RAG |
| **FP-9** | Prod-Beta | FastAPI `/api/mini/chat` + Streamlit panel | Latency + safety checks |
| **FP-10** | Mini v1.0 | Full inference chain + versioned release | Success criteria checklist |

**Scale path (post v1.0, not in first 18 sprints as commit):** 10M → 50M → 100M → 300M → 1B using same workers/datasets/tokenizer family.

---

## 5. Sprint-by-sprint plan

Each sprint: **Goal · Workers · Stories · Acceptance · Depends on · Demo**.

---

### Epic E0 — Foundation & Taxonomy

#### Sprint 0 — Program bootstrap (Week 1–2)

| | |
|---|---|
| **Goal** | Create Mini factory skeleton + vision alignment with existing platform |
| **Feature phase** | FP-0 |
| **Workers** | Scaffold only (`W-INGEST` stub …) |
| **Stories** | (1) Create `mini/` package layout (2) Artifact path conventions (3) `schema_version` pydantic models (4) CLI skeleton (5) Link plan from README |
| **Acceptance** | `python -m mini.orchestrator list-workers` works; lake dirs exist; CI still green |
| **Demo** | Architecture walkthrough + empty pipeline dry-run |
| **Maps to** | Phase 0 Vision, Phase 3 lake skeleton |

#### Sprint 1 — Domain taxonomy freeze (Week 3–4)

| | |
|---|---|
| **Goal** | Lock agriculture taxonomy used by all later workers |
| **Workers** | `W-NORMALIZE` dependency (taxonomy service) |
| **Stories** | (1) Taxonomy JSON for Soil/Weather/Crops/Stages/Disease/Pest/Fertilizer/Irrigation/Schemes/Market/Finance/Machinery (2) Crop alias table EN/MR/HI (extend `query_understanding.CROP_ALIASES`) (3) Region hierarchy MH + India stubs (4) Unit vocabulary (kg/ha, mm, °C) |
| **Acceptance** | Taxonomy validator; every existing `data/*.json` entity maps to a category |
| **Demo** | Taxonomy browser in Streamlit sidebar (read-only) |
| **Maps to** | Phase 1 Domain Planning |

---

### Epic E1 — Data Lake & Engineering

#### Sprint 2 — Lake layout + source registry (Week 5–6)

| | |
|---|---|
| **Goal** | Production-shaped data lake; never train from `raw/` |
| **Workers** | `W-INGEST` v1 |
| **Stories** | (1) Create `data/lake/{raw,processed,training,validation,test}/…` domains (2) Source registry (`sources.yaml`: ICAR, data.gov.in, local JSON, Wikipedia agri, etc.) (3) Ingest existing `data/*.json` into lake raw with manifests (4) Checksums + run IDs |
| **Acceptance** | Ingesting twice is idempotent; manifest lists file hashes |
| **Demo** | Show lake tree + ingest report |
| **Maps to** | Phase 2–3 |

#### Sprint 3 — Validation → cleaning pipeline (Week 7–8)

| | |
|---|---|
| **Goal** | Automated quality gate on every batch |
| **Workers** | `W-VALIDATE`, `W-CLEAN`, `W-DEDUP` |
| **Stories** | (1) Schema/type checks (2) Strip HTML/boilerplate (3) Near-duplicate detection (4) Quarantine folder for rejects (5) Quality report JSON |
| **Acceptance** | Synthetic dirty batch is cleaned; dups removed; report metrics recorded |
| **Demo** | Before/after sample + quarantine list |
| **Maps to** | Phase 4 Data Engineering |

#### Sprint 4 — Normalize + language + versioning (Week 9–10)

| | |
|---|---|
| **Goal** | Canonical records + language tags + dataset versioning |
| **Workers** | `W-NORMALIZE`, `W-LANGDETECT`, `W-STANDARD` (start) |
| **Stories** | (1) Map to standard record fields (2) Crop/region IDs from taxonomy (3) Language detection (4) DVC or simple `datasets/versions/vYYYYMMDD/` manifests (5) Parquet export |
| **Acceptance** | ≥90% of processed rows have language + category; train split exportable |
| **Demo** | Parquet sample in notebook/CLI |
| **Maps to** | Phase 4–5 |
| **Feature phase** | **FP-1, FP-2** |

---

### Epic E2 — Expert Datasets & Analysis

#### Sprint 5 — Analysis dashboard worker (Week 11–12)

| | |
|---|---|
| **Goal** | Every ingest cycle produces coverage intelligence |
| **Workers** | `W-ANALYZE` |
| **Stories** | (1) Missingness, dups, length histograms (2) Crop/state/language balance (3) Coverage gaps vs taxonomy (4) Streamlit “Data Factory” panel or static HTML report |
| **Acceptance** | Report regenerates from CLI; gaps list non-empty initially |
| **Demo** | Dashboard after ingest of current KB |
| **Maps to** | Phase 7 |

#### Sprint 6 — QA synthesis factory (Week 13–14)

| | |
|---|---|
| **Goal** | Generate large expert-style QA from structured facts (bootstrap toward 1M pairs) |
| **Workers** | `W-QASYNTH` |
| **Stories** | (1) Template + controlled generation from crops/diseases/schemes/fertilizer (2) Multilingual prompt variants (EN/MR/HI) (3) Confidence scoring heuristics (4) Human review queue CSV for verified=true (5) Initial slices: Disease, Crop, Fertilizer, Scheme |
| **Acceptance** | **≥10k** train QA + **≥1k** val with schema; no train/test leakage |
| **Demo** | Sample bilingual QA + counts by category |
| **Maps to** | Phase 6 (start), Phase 12 (templates) |
| **Feature phase** | **FP-3** |

#### Sprint 7 — Domain expert packs expansion (Week 15–16)

| | |
|---|---|
| **Goal** | Scale QA packs toward Phase 6 targets (still realistic intermediate volumes) |
| **Workers** | `W-QASYNTH` v2, curators |
| **Stories** | (1) Soil + Weather + Pest + Irrigation + Market + Finance packs (2) Hard negatives / “I don’t know / need more info” answers (3) Safety refusals for dangerous pesticide misuse (4) Intermediate target **≥50k** train pairs |
| **Acceptance** | Category coverage ≥8 domains; language mix ≥20% non-English if MR templates exist |
| **Demo** | Coverage chart vs Phase 6 long-term targets |
| **Maps to** | Phase 6 progress |

**Long-term Phase 6 volume (not a single-sprint commitment):**

| Pack | Target (over time) | Sprint 7 interim |
|---|---|---|
| Disease | 200k | 8k |
| Crop management | 200k | 8k |
| Pest | 150k | 5k |
| Soil | 100k | 5k |
| Fertilizer | 100k | 5k |
| Weather | 80k | 4k |
| Government | 50k | 5k |
| Finance | 50k | 3k |
| Machinery | 50k | 2k |
| **Total** | **~1M** | **~50k** |

---

### Epic E3 — Knowledge Graph Factory

#### Sprint 8 — KG builder worker (Week 17–18)

| | |
|---|---|
| **Goal** | Automated graph construction from standard records + existing graph |
| **Workers** | `W-KGBUILD` |
| **Stories** | (1) Entity extraction from standard records (2) Relation templates (AFFECTED_BY, TREATED_BY, GROWS_IN, COVERED_BY, …) (3) Merge with `data/knowledge_graph.json` (4) Export NetworkX + optional Neo4j loader stub (5) Graph QA triples as training text |
| **Acceptance** | ≥200 nodes, ≥400 edges; GraphRAG APIs still pass tests |
| **Demo** | Graph growth before/after |
| **Maps to** | Phase 8 |
| **Feature phase** | **FP-4** |

---

### Epic E4 — Tokenizer + Mini Model Training

#### Sprint 9 — Corpus + tokenizer training (Week 19–20)

| | |
|---|---|
| **Goal** | Agriculture-aware tokenizer v0.1 |
| **Workers** | `W-TOKEN` |
| **Stories** | (1) Build domain corpus from processed lake + QA (2) Train BPE/Unigram 30–50k (sentencepiece) (3) Force-include crop/pest/fertilizer/district tokens (4) Fertility tests for Marathi agri terms (5) Artifact versioning `tokenizer/v0.1` |
| **Acceptance** | Lower fertility on agri terms vs generic baseline; vocab size in range |
| **Demo** | Tokenization of sample MR/EN farm sentences |
| **Maps to** | Phase 9, Model v0.1 |
| **Feature phase** | **FP-5** |

#### Sprint 10 — Mini architecture + train harness (Week 21–22)

| | |
|---|---|
| **Goal** | Implement ~1M decoder-only model + training loop (no full pretrain yet) |
| **Workers** | `W-PRETRAIN` skeleton |
| **Stories** | (1) Config: emb 128, layers 6, heads 4, hidden 256, seq 512 (2) RoPE, RMSNorm, SwiGLU, weight tying (3) PyTorch module + parameter count assert (~0.8–1.5M) (4) Lightning/plain trainer, AMP, checkpointing (5) MLflow or local run logs |
| **Acceptance** | Overfit 32-batch smoke; param count report committed |
| **Demo** | Loss drops on tiny corpus |
| **Maps to** | Phase 10–11 stage scaffolding |

#### Sprint 11 — Domain pretraining (Week 23–24)

| | |
|---|---|
| **Goal** | Base model v0.2 on agriculture text |
| **Workers** | `W-PRETRAIN` |
| **Stories** | (1) Stream tokenized corpus (2) Context packing (3) Eval perplexity on val agri text (4) Checkpoint `models/mini/v0.2-base` (5) Resume/repro seeds |
| **Acceptance** | Val PPL improves vs random init; training reproducible with seed |
| **Demo** | Training curves + sample completions (raw) |
| **Maps to** | Phase 11 Stage 2, Model v0.2 |
| **Feature phase** | **FP-6** |

#### Sprint 12 — Instruction + QA SFT (Week 25–26)

| | |
|---|---|
| **Goal** | Instruction-tuned Mini v0.3 / agri-QA v0.4 |
| **Workers** | `W-SFT` |
| **Stories** | (1) Instruction format (system/user/assistant) (2) Multilingual SFT mix (3) RAG-context SFT examples (answer from context only) (4) Safety/refusal set (5) Checkpoints v0.3, v0.4 |
| **Acceptance** | Gold val F1/EM beats base; RAG-conditioned answers cite context more often |
| **Demo** | Side-by-side base vs SFT answers |
| **Maps to** | Phase 11 Stage 3–4, Phase 12, Models v0.3–v0.4 |
| **Feature phase** | **FP-7** |

---

### Epic E5 — Evaluation, Quantization, Gates

#### Sprint 13 — Evaluation harness (Week 27–28)

| | |
|---|---|
| **Goal** | Block bad models from deploy automatically |
| **Workers** | `W-EVAL` |
| **Stories** | (1) Gold sets: disease, fertilizer, schemes, market (2) Metrics: PPL, EM, F1, ROUGE-L, latency, memory (3) Hallucination probes (fact conflict) (4) Regional correctness (MH crops) (5) Gate config: min scores to promote version |
| **Acceptance** | Eval CLI produces HTML/JSON report; failing gate exits non-zero |
| **Demo** | Scorecard for v0.4 |
| **Maps to** | Phase 13 |

#### Sprint 14 — Quantize + version packaging (Week 29–30)

| | |
|---|---|
| **Goal** | Deployable Mini artifacts with size budgets |
| **Workers** | `W-QUANT`, `W-DEPLOY` (package only) |
| **Stories** | (1) INT8 (and optional INT4) export (2) Size & CPU latency benchmarks (3) Model card + license + data provenance (4) Version registry `v0.5-reasoning-lite` optional small chain-of-thought traces (5) Semantic versioning policy |
| **Acceptance** | Quantized model ≤ target disk budget; latency p95 logged on reference machine |
| **Demo** | Size comparison FP32 vs INT8 |
| **Maps to** | Phase 11 Stage 6, Phase 16 partial |

---

### Epic E6 — Inference product integration

#### Sprint 15 — Inference pipeline (Week 31–32)

| | |
|---|---|
| **Goal** | Full chain: intent → entity → retrieve → Mini → validate |
| **Workers** | `W-INFER`, `W-RAG` (wrap existing), `W-AGENT` |
| **Stories** | (1) Reuse `query_understanding` + `advanced_rag` (2) Context builder with citation markers (3) Mini generate with context (4) Answer validator (grounding score, banned advice patterns) (5) Fallback to template synthesizer if low confidence |
| **Acceptance** | End-to-end unit/integration tests; no answer without sources when mode=grounded |
| **Demo** | Cotton disease query with citations |
| **Maps to** | Phase 14–15 |
| **Feature phase** | **FP-8** |

#### Sprint 16 — Platform integration & multi-agent (Week 33–34)

| | |
|---|---|
| **Goal** | Mini becomes default brain inside planner; agents still tool-specialists |
| **Workers** | `W-DEPLOY`, serving |
| **Stories** | (1) FastAPI `POST /api/mini/chat` (2) Replace/augment `app/llm/generator.py` (3) Streamlit Mini panel + citations (4) Agent outputs as optional context channels (5) Feature flag `USE_MINI_LLM` |
| **Acceptance** | Existing 42+ tests still pass; new Mini tests green; flag off = old behavior |
| **Demo** | Live assistant using Mini+RAG |
| **Maps to** | Phase 16 v0.6–v0.8 |
| **Feature phase** | **FP-9** |

#### Sprint 17 — Production beta → Mini v1.0 (Week 35–36)

| | |
|---|---|
| **Goal** | Release KrushiVerseAI Mini v1.0 as platform component |
| **Stories** | (1) Hardening, docs, runbooks (2) Eval gate on release candidate (3) Load smoke (4) Model version matrix published (5) Scale roadmap decision for 10M (6) Success criteria checklist sign-off |
| **Acceptance** | All success criteria (Section 8) met or explicitly deferred with owners |
| **Demo** | Release demo day |
| **Maps to** | Phase 16 v0.9–v1.0, Phase 17 planning |
| **Feature phase** | **FP-10** |

---

## 6. Sprint calendar overview

```
Month 1   S0–S1   Taxonomy + factory scaffold
Month 2–3 S2–S4   Lake + engineering workers
Month 3–4 S5–S7   Analysis + QA packs (50k)
Month 5   S8      KG factory
Month 5–7 S9–S12  Tokenizer + train + SFT
Month 7–8 S13–S14 Eval + quant
Month 8–9 S15–S17 Inference + product + v1.0
```

**Early demos (every 4–6 weeks):** S1 taxonomy · S4 lake · S7 QA · S11 base model · S15 RAG+Mini · S17 release.

---

## 7. How workers collaborate (example end-to-end run)

```text
1. W-INGEST     pulls data.gov.in prices + local disease JSON
2. W-VALIDATE   rejects broken rows → quarantine
3. W-CLEAN      normalizes text encoding
4. W-DEDUP      removes near-duplicates
5. W-NORMALIZE  maps crop aliases via taxonomy
6. W-LANGDETECT tags mr/en
7. W-STANDARD   emits schema v1 parquet
8. W-ANALYZE    updates coverage dashboard
9. W-QASYNTH    expands facts → QA pairs
10. W-KGBUILD   updates graph edges (Cotton→Pink Bollworm→Trap)
11. W-TOKEN     (weekly) refreshes tokenizer if corpus shifted
12. W-PRETRAIN / W-SFT  nightly or on-demand GPU job
13. W-EVAL      gates checkpoint
14. W-QUANT     produces serveable weights
15. W-DEPLOY    publishes version tag
16. W-INFER     online path uses W-RAG + Mini + validator
```

**Parallelism:** Ingest domains can run in parallel; train waits on standard dataset version; deploy waits on eval gate.

---

## 8. Success criteria (v1.0 Mini program)

### Must-have (release blockers)

- [ ] Standardized data lake with raw ≠ training separation  
- [ ] Schema v1 standard records + versioned dataset manifests  
- [ ] Automated ingest → clean → standardize pipeline (CLI/DAG)  
- [ ] Domain tokenizer v0.1 (30–50k) with agri term coverage  
- [ ] Mini model **~1M parameters** (±50%) trained and versioned  
- [ ] Instruction/QA SFT checkpoint with eval report  
- [ ] Hybrid RAG + Mini inference path with citations  
- [ ] Eval gates (at least PPL + QA F1 + latency + grounding heuristic)  
- [ ] FastAPI + Streamlit integration behind feature flag  
- [ ] Documented scale path to larger models reusing artifacts  

### Should-have

- [ ] ≥50k training QA pairs across ≥8 domains  
- [ ] KG ≥200 nodes automated builder  
- [ ] INT8 quantized serve artifact  
- [ ] Marathi + English answer quality on gold set  

### Defer post-v1.0 (explicit)

- Full 1M curated human-verified QA  
- Neo4j production cluster  
- Vision-language Mini  
- Multi-GPU 100M+ training  
- Kubernetes production HA  

---

## 9. Model version ladder (Phase 16) mapped to sprints

| Version | Meaning | Sprint |
|---|---|---|
| v0.1 Tokenizer | Domain vocab | S9 |
| v0.2 Base | Pretrained Mini | S11 |
| v0.3 Instruction | General instruction SFT | S12 |
| v0.4 Agri QA | Domain QA SFT | S12 |
| v0.5 Reasoning-lite | Optional short rationales | S14 |
| v0.6 RAG-coupled | Trained/serving with context | S15 |
| v0.7 Vision hook | API-ready multimodal stub | post / optional S16 |
| v0.8 Multi-agent | Planner uses Mini as synthesizer | S16 |
| v0.9 Prod beta | Hardened serve | S17 |
| v1.0 Mini | Release | S17 |

---

## 10. Technology choices (pragmatic for this repo)

| Layer | Choice now | Later |
|---|---|---|
| Training | PyTorch | Lightning if complexity grows |
| Tokenizer | SentencePiece | Same family for larger models |
| Datasets | Parquet + JSONL + manifests | HF Datasets hub private |
| Vector | Existing Qdrant/local dense | Same |
| Graph | NetworkX → optional Neo4j | Neo4j when multi-user KG |
| Orchestration | Python DAG CLI first | Prefect/Airflow |
| Tracking | Local JSON/MLflow lite | W&B |
| Serve | FastAPI (existing) | K8s |
| Eval | Custom harness + optional Ragas | DeepEval |

**Do not** block Sprint 0–8 on GPU cloud. Training sprints (S10–S12) need a single GPU (or CPU overfit for CI smoke only).

---

## 11. Mini model config lock (Phase 10)

| Hyperparameter | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Approx params | ~1.0M (assert 0.8–1.5M) |
| Embedding dim | 128 |
| Layers | 6 |
| Heads | 4 |
| Hidden / FFN | 256 / ~682 (SwiGLU) |
| Seq length | 512 |
| Positional | RoPE |
| Norm | RMSNorm |
| Attention | SDPA / Flash if available |
| Weight tying | Yes |
| Precision | AMP fp16/bf16 when GPU |

This config is **locked until v1.0** unless param count assert fails.

---

## 12. Risk register & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 1M model too weak alone | High | Mandatory RAG + validator; fallback synthesizer |
| Data quality / hallucination in synth QA | High | Confidence scores, human review sample, verified flag |
| Marathi tokenization poor | High | Domain tokenizer + MR fertility tests |
| Scope explosion to 1M QA | Medium | Interim 50k gate; long-term backlog |
| GPU unavailable | Medium | CPU smoke tests in CI; cloud GPU only for S11–S12 |
| Breaking existing platform | High | Feature flag; keep template path |
| License/provenance issues | High | Source + license fields mandatory |

---

## 13. Team / RACI (even if solo — wear hats)

| Role | Responsibilities |
|---|---|
| Data Engineer | Workers W-INGEST…W-STANDARD |
| Agronomy curator | Taxonomy, verified QA, safety |
| ML Engineer | Tokenizer, Mini train/eval |
| Platform Engineer | FastAPI/Streamlit integration |
| QA | Eval gold sets, gate thresholds |

Solo mode: execute **one worker vertical slice** each sprint; do not parallelize all domains early.

---

## 14. Definition of Done (every sprint)

1. Code merged + tests for new workers  
2. Artifact written under versioned path  
3. CLI command documented in `mini/README.md`  
4. Report or demo note in `docs/sprint-notes/SXX.md`  
5. No regression on existing platform test suite  
6. Feature-phase checklist updated  

---

## 15. Immediate next actions (start Sprint 0)

| # | Action | Output |
|---|---|---|
| 1 | Create `mini/` package skeleton + orchestrator CLI | FP-0 |
| 2 | Add `mini/taxonomy/` draft from current crops/diseases/schemes | Sprint 1 start |
| 3 | Mirror current `data/*.json` into `data/lake/raw/seed/` with manifest | Sprint 2 start |
| 4 | Define pydantic `StandardRecord` shared by all workers | Contract freeze |
| 5 | Add Streamlit “Factory status” stub (worker list + last run) | Visibility |
| 6 | Keep serving path on template+RAG until Mini v0.4 | Risk control |

---

## 16. Traceability: original phases → plan

| Original phase | Covered by |
|---|---|
| 0 Vision | This doc §0–2, Sprint 0 |
| 1 Domain | Sprint 1 |
| 2 Collection | Sprint 2, source registry |
| 3 Data lake | Sprint 2 |
| 4 Engineering | Sprints 3–4 |
| 5 Standardization | Sprint 4, §2.2 |
| 6 Expert datasets | Sprints 6–7 (+ continuous) |
| 7 Analysis | Sprint 5 |
| 8 Knowledge graph | Sprint 8 |
| 9 Tokenizer | Sprint 9 |
| 10 Mini LM | Sprint 10 |
| 11 Training pipeline | Sprints 10–12, 14 |
| 12 Instruction data | Sprints 6–7, 12 |
| 13 Evaluation | Sprint 13 |
| 14 Retrieval | Existing RAG + Sprint 15 |
| 15 Inference | Sprint 15–16 |
| 16 Model versions | §9 + S9–S17 |
| 17 Scaling | Post-v1.0 program (same workers) |

---

## 17. Scaling program (post Mini v1.0)

Same **workers, schema, tokenizer family, eval gates**:

```
1M Mini v1.0
  → 10M (wider/deeper config only)
  → 50M
  → 100M
  → 300M
  → 1B foundation (multi-node train)
```

**Rule:** Scaling changes **model config + compute**, not data contracts or worker IDs.

---

## 18. Document control

| Field | Value |
|---|---|
| Owner | KrushiVerse-AI engineering |
| Status | Sprint 0 (FP-0) **implemented & tested** (58/58); pushed to `arena/019f74ea-krushiverse-ai` |
| Related | `ARCHITECTURE.md`, `README.md`, platform v10.2 RAG stack |
| Next doc | `docs/sprint-notes/S00.md` after Sprint 0 execution |

---

*End of plan. Implement Sprint 0 first; do not start training before FP-2 (standard records) exists.*
