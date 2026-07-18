# Generation 10 — Autonomous AI Agriculture Platform Architecture

## Executive Overview
**AI Krushi Mitra** (`KrushiVerse-AI`) is an Autonomous Operating System for Agriculture. Unlike simple chatbots or standard RAG implementations, this platform unifies **RAG, Multi-Agent Orchestration, Knowledge Graphs (GraphRAG), Computer Vision, Live Weather, Mandi Market Intelligence, IoT Sensors, Predictive AI Models, Automated Workflows, and Farm Memory**.

Primary language delivery is optimized for **Marathi (मराठी)** to directly empower farmers in regional languages, alongside full English support.

---

## Technical Stack & Architecture Map

### Architecture Layers & Flow

```
                     `Farmer`  
                        │  
                        ▼  
               `AI Krushi Assistant`  
                        │  
                        ▼  
               `Planner Agent`  
                        │  
 ┌──────────────────────┼──────────────────────┐  
 │                      │                      │  
 ▼                      ▼                      ▼  
Weather Agent      Crop Agent          Disease Agent  
 │                      │                      │  
 ▼                      ▼                      ▼  
Market Agent       Soil Agent        Fertilizer Agent  
 │                      │                      │  
 ▼                      ▼                      ▼  
Government Agent   Vision Agent       Finance Agent  
                        │  
                        ▼  
               `Knowledge Layer`  
       ┌──────────────┬───────────────┬──────────────┐  
       ▼              ▼               ▼  
  GraphRAG       Hybrid Search     Vector Search  
       ▼              ▼               ▼  
         `Unified Knowledge Platform`  
                        │  
                        ▼  
                     `LLM`  
                        │  
                        ▼  
                 `Marathi Answer`
```

### Stack Alignment
- **Agent Framework:** LangGraph / AutoGen style Planner-Agent Architecture with 9 Specialized Sub-Agents.
- **Knowledge Graph:** Neo4j / NetworkX GraphRAG engine storing crops, pests, soil, schemes, weather conditions, and inter-entity relationships.
- **Hybrid Search:** Rank-BM25 Lexical Matching + Cosine Vector Semantic Search with Reciprocal Rank Fusion (RRF) Re-ranking.
- **Computer Vision & OCR:** Vision AI leaf disease classifier (GPT-4o / Qwen-VL architecture) + Soil Health Card OCR parser.
- **Live RAG & Feeds:** IMD Weather, Agmarknet Mandi Prices, IoT Soil Telemetry, Sentinel-2 Satellite Vegetation Index (NDVI/NDWI).
- **Predictive AI Engines:** Yield Forecasting, Pest Outbreak Risk Probability, Smart Irrigation Runtime Calculator, Target NPK Fertilizer Calculator.
- **Farm Memory Store:** Multi-season field profiles, historical soil tests, past actions, and customized farm logs.

---

## 9 Specialized Sub-Agents

1. **Weather Agent:** Evaluates IMD forecasts, frost alerts, monsoon trends, and microclimate advisories.
2. **Crop Agent:** Provides university growth-stage guidelines, crop calendar windows, and intercultural operations.
3. **Disease Agent:** Identifies plant pathogens and prescribes integrated organic & chemical disease management.
4. **Market Agent:** Fetches Agmarknet mandi modal prices, market arrivals, price trends, and optimal sale timing.
5. **Soil Agent:** Evaluates Soil Health Card readings (pH, NPK, Organic Carbon, EC) and determines crop-soil compatibility.
6. **Fertilizer Agent:** Calculates exact dosages of commercial fertilizers (Urea, DAP, MOP) and fertigation runtime schedules.
7. **Government Agent:** Maps eligible state/central schemes (PM-KISAN, PMFBY 1-Rupee Crop Insurance, PoCRA, Magel Tyala Shettale).
8. **Vision Agent:** Processes leaf photo uploads and OCR document text via computer vision diagnostics.
9. **Finance Agent:** Computes budget estimates, return on investment (ROI), crop loan eligibility, and insurance claim workflows.

---

## Priority Realization
1. **Hybrid RAG** (Vector + BM25 + Re-ranking) implemented in `app/knowledge/hybrid_search.py`.
2. **GraphRAG** for multi-hop agricultural knowledge graph reasoning in `app/knowledge/graph_rag.py`.
3. **Agentic RAG** with Planner Agent orchestrating specialized agents in `app/agents/`.
4. **Memory RAG** for personalized farm history recommendations in `app/memory/farm_memory.py`.
5. **Live RAG** integrating weather, mandi prices, IoT, and satellite feeds in `app/live_feeds/`.
6. **Computer Vision** for plant disease diagnostic analysis in `app/vision/disease_classifier.py`.
7. **Predictive AI** for yield, pest risk, irrigation, and fertilizer target planning in `app/predictive/`.
