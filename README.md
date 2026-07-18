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

### 2. Multi-Tiered RAG Knowledge Layer
- **Hybrid Search RAG:** Lexical BM25 search + Cosine Vector Semantic search fused via Reciprocal Rank Fusion (RRF).
- **GraphRAG Engine:** Knowledge Graph mapping relationships between Crops, Diseases, Soil, Weather, Fertilizers, and Government Schemes.
- **Memory RAG:** Stores personalized farmer profiles, past crop rotation, soil test history, and previous interventions.
- **Live RAG:** Streams real-time IMD weather warnings, Agmarknet mandi price feeds, IoT sensor telemetry, and Sentinel-2 satellite NDVI canopy metrics.

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
