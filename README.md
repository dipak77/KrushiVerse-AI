# 🌾 KrushiVerse-AI — Autonomous AI Agriculture Platform (AI Krushi Mitra)

**Generation 10 Autonomous Agriculture Operating System** combining RAG, GraphRAG, AI Agents, Computer Vision, Live Feeds, IoT, Satellite Intelligence, Predictive Models, Workflow Automation, and Farm Memory.

Designed specifically to deliver actionable agricultural solutions in **Marathi (मराठी)** and **English**.

---

## 🌟 Architecture Highlights

```
                    Farmer Query
                        │
                        ▼
               AI Krushi Assistant
                        │
                        ▼
                  Planner Agent
                        │
 ┌──────────────────────┼──────────────────────┐
 │                      │                      │
 Weather Agent     Crop Agent          Disease Agent
 │                      │                      │
 Market Agent      Soil Agent        Fertilizer Agent
 │                      │                      │
 Government Agent  Vision Agent       Finance Agent
                        │
                        ▼
                 Knowledge Layer
       (GraphRAG + Hybrid Search + Vector Search)
                        │
                        ▼
               Unified Knowledge Platform
                        │
                        ▼
               Marathi LLM Synthesizer
                        │
                        ▼
                 Marathi Answer
```

---

## 🚀 Key Features

### 1. Agentic Multi-Agent System
- **Planner Agent:** Central coordinator that deconstructs complex farmer questions and delegates sub-tasks to specialized agents.
- **9 Specialized Agents:** Weather, Crop, Disease, Market, Soil, Fertilizer, Government, Vision, and Finance Agents.

### 2. Multi-Tiered Advanced RAG Knowledge Layer (v10.1)
- **Hybrid Search RAG:** Lexical BM25 search + Cosine Vector Semantic search fused via Reciprocal Rank Fusion (RRF).
- **GraphRAG Engine:** Knowledge Graph mapping relationships between Crops, Diseases, Soil, Weather, Fertilizers, and Government Schemes.
- **Expanded open-source KB:** 20+ crops, 20 pests/diseases, 15 schemes, 20 mandis, 75+ advisories, seed varieties, irrigation practices, climate zones (ICAR/SAU/gov-style open compilations).
- **Dense embeddings + Qdrant:** Hash / MiniLM / OpenAI-compatible embeddings; Qdrant when `QDRANT_URL` is set, otherwise local numpy dense index (disk-cached under `.cache/`).
- **data.gov.in / Agmarknet:** Live commodity prices with API key; automatic local market KB fallback.
- **Web RAG:** DuckDuckGo Instant Answers + Wikipedia (+ offline open-catalog fallbacks) for latest public knowledge.
- **Tool-augmented RAG:** Open-Meteo weather, mandi + Agmarknet tools, scheme lookup, crop PoP tools, open-source catalog — fused with local docs via cross-source RRF.
- **Streamlit “Advanced RAG & Sources” tab:** Fused docs, citations, web hits, tools, Agmarknet explorer.
- **Query understanding:** Crop entity extraction (EN/MR), intent tags, multi-query expansion.
- **Memory RAG:** Stores personalized farmer profiles, past crop rotation, soil test history, and previous interventions.
- **Live RAG:** Weather (Open-Meteo + regional feed), Agmarknet-style mandi prices, IoT sensor telemetry, and Sentinel-2 style NDVI metrics.

### 3. Computer Vision & OCR
- **Plant Leaf Disease Diagnostic Classifier:** Identifies crop diseases and provides instant organic/chemical treatment plans.
- **Soil Health Card OCR Parser:** Extracts pH, EC, Organic Carbon, Nitrogen, Phosphorus, and Potassium from scanned report text.

### 4. Predictive AI Engines
- **Yield Forecasting:** Predicts total output based on soil health, irrigation quality, and crop stage.
- **Pest Outbreak Risk Model:** Calculates outbreak probabilities using real-time humidity, temperature, and rainfall.
- **Smart Irrigation Schedule:** Calculates daily crop evapotranspiration and drip runtime hours.
- **Target Fertilizer Calculator:** Computes exact bags of Urea, DAP, and MOP required per acre.

---

## 🛠️ Quickstart Guide

### Prerequisites
- Python 3.11+
- Virtualenv / PIP

### Setup & Installation
```bash
# Clone the repository
git clone https://github.com/dipak77/KrushiVerse-AI.git
cd KrushiVerse-AI

# Execute convenience script to set up environment and run tests
./run.sh test
```

### Running the Services

#### 1. Start FastAPI Backend API:
```bash
./run.sh api
# Server starts at http://localhost:8000
```

#### 2. Start Streamlit Interactive UI Dashboard:
```bash
./run.sh ui
# Dashboard opens at http://localhost:8501
```

#### Advanced RAG API (multi-source)
```bash
# Knowledge base stats
curl http://localhost:8000/api/knowledge/stats

# Advanced retrieval: local + tools + web
curl -X POST http://localhost:8000/api/rag/advanced \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"latest cotton pest advisory Maharashtra\",\"force_web\":true,\"top_k\":8}"
```

Environment flags (see `.env.example`):
- `ENABLE_WEB_RAG=true|false` (default true)
- `ENABLE_TOOL_RAG=true|false` (default true)
- `ENABLE_DENSE_RAG=true|false` (default true)
- `EMBEDDING_BACKEND=auto|hash|minilm|openai`
- `QDRANT_URL=http://127.0.0.1:6333` (optional; local dense index if unset)
- `DATA_GOV_IN_API_KEY=` free key from https://data.gov.in for live Agmarknet
- `WEB_CACHE_TTL_SEC=300`
- `RAG_TOP_K=8`

#### Open data + embeddings endpoints
```bash
curl http://localhost:8000/api/rag/backends
curl "http://localhost:8000/api/opendata/agmarknet?commodity=Cotton&state=Maharashtra"
```

---

## 🧪 Running Unit & Integration Tests

```bash
./venv/bin/pytest -v
```
The test suite validates:
- Multi-Agent execution & Planner routing
- Hybrid RAG & GraphRAG traversal
- Vision diagnostic & Soil OCR parsing
- Live weather, market, IoT, and satellite feeds
- Predictive AI yield, irrigation, and fertilizer models
- FastAPI REST endpoints

---

## 📁 Repository Directory Structure

```
KrushiVerse-AI/
├── app/
│   ├── config.py                 # Platform settings & configuration
│   ├── main.py                   # FastAPI application entrypoint & REST API
│   ├── agents/                   # Planner Agent & 9 Specialized Sub-Agents
│   ├── knowledge/                # Hybrid Search (BM25+Vector), GraphRAG & KB Loader
│   ├── memory/                   # Personalized Farm Memory Store
│   ├── live_feeds/               # IMD Weather, APMC Mandi, IoT & Sentinel Satellite Feeds
│   ├── vision/                   # Leaf Disease Vision Classifier & Soil Card OCR
│   ├── predictive/               # Yield, Pest Risk, Irrigation & Fertilizer Models
│   ├── workflows/                # Automated Telemetry Audits & Alert Dispatch
│   └── llm/                      # Marathi & English Response Synthesizer
├── data/                         # Built-in Datasets (ICAR Advisories, Mandis, Schemes, Graph)
├── ui/                           # Streamlit Web UI Dashboard
├── tests/                        # Comprehensive PyTest Suite
├── requirements.txt
├── run.sh                        # Service Launcher Script
├── ARCHITECTURE.md               # Full Architecture Specification
└── README.md
```

---

## 🤝 License
Released under the MIT License.

## 📐 Production architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the complete 2026 nine-layer Agentic GraphRAG blueprint, multilingual voice flow, safety/governance requirements, evaluation plan, observability, and staged production roadmap. Its latency, accuracy, scale, and cost figures are explicitly treated as targets until validated by benchmark and production evidence.

## 🧠 KrushiVerseAI Mini (1M) — Sprint plan

End-to-end implementation roadmap for the ~1M-parameter agriculture Mini LLM, data lake, automated workers, training/eval, and RAG-coupled inference:

→ **[`docs/KRUSHIVERSE_MINI_SPRINT_PLAN.md`](docs/KRUSHIVERSE_MINI_SPRINT_PLAN.md)**  
→ Factory package: [`mini/README.md`](mini/README.md)  
→ Sprint 0 notes: [`docs/sprint-notes/S00.md`](docs/sprint-notes/S00.md)

Summary: **6 epics · 18 two-week sprints · automated worker DAG** (ingest → clean → QA → KG → tokenizer → train → eval → deploy), reusing the existing v10.2 multi-source RAG platform. The Mini model is one worker component, not the only intelligence.

### Sprint 0–10 (factory through Mini ~1M harness) — done

```bash
python -m mini.orchestrator token --execute --vocab-size 32000
python -m mini.orchestrator pretrain --execute --steps 50
python -m mini.orchestrator run sprint10 --execute
```

API: `/api/taxonomy` · `/api/lake/*` · `/api/lake/tokenizer` · `/api/lake/pretrain`

### React OS UI (reference design)

```bash
cd ui/web
npm install
npm run build
# then open http://127.0.0.1:8000/ui  (or / while FastAPI serves dist/)
# dev: npm run dev  → http://127.0.0.1:5173  (API proxy to :8000)
```

**Do not push:** `data/lake/**`, `mini/datasets/**`, tokenizer model binaries, `mini/models/checkpoints/**`, `ui/web/node_modules/**`.
