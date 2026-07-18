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

---

## Ultimate Agentic GraphRAG Blueprint (2026)

This section is the target production architecture for AI Krushi Mitra. The metrics below are **acceptance targets**, not claims about the current local/demo implementation. Every target must be validated with a versioned evaluation set and production telemetry before it is published.

### Product targets

| Target | Goal |
|---|---|
| Languages | Marathi, Hindi, English (with transliteration and voice support) |
| Retrieval quality | MRR ≥ 0.89; faithfulness (RAGAS) ≥ 0.91 |
| Intent accuracy | ≥ 94% on a balanced Marathi/Hindi/English test set |
| Latency | p95 ≤ 1.2 s for text retrieval/answering; voice latency measured separately |
| Availability | 99.9% for the API, with graceful degradation when live feeds are unavailable |
| Scale | Designed for 140M farmers, but capacity-tested in staged traffic increments |

### Nine-layer reference stack

**Layer 0 — Foundation and sources.** A governed lakehouse (S3-compatible object storage plus Parquet) holds versioned structured and unstructured data: government schemes, ICAR research, IMD weather, Agmarknet/APMC mandi prices, Soil Health Cards, crop images, and farmer voice/text queries. APIs, PDFs, images, and audio are retained with provenance, licensing, language, timestamp, and freshness metadata.

**Layer 1 — Multimodal ingestion.** An idempotent pipeline uses Docling or Unstructured for layout-aware PDF parsing, OCR for scanned documents, Whisper for speech-to-text, and a vision encoder for images. It preserves page/table coordinates, extracts document structure, detects language, removes duplicates, and sends low-confidence OCR/transcripts to review. Chunking is semantic and structure-aware; 512 tokens with 128-token overlap is only a starting configuration.

**Layer 2 — Enrichment and ontology.** BGE-M3 (1024-dimensional) embeddings, optional ColBERT late interaction, entity extraction, normalization, and entity linking populate an AgriKG ontology: crop, variety, pest, disease, symptom, stage, soil, nutrient, weather, location, treatment, scheme, market, and source. Every relationship has evidence, confidence, effective dates, and a source citation.

**Layer 3 — Persistence.** Qdrant (HNSW and payload filters) stores vectors; OpenSearch/Elasticsearch or PostgreSQL FTS provides BM25; Neo4j stores graph entities and relationships; PostgreSQL stores transactional profiles, consent, feedback, and audit records; object storage stores originals and derived artifacts. Backups, retention, encryption, migrations, and disaster recovery are mandatory. `pgvector` is a valid smaller-deployment alternative to Qdrant, not a second required vector store.

**Layer 4 — Query understanding.** Language and script detection, transliteration, Marathi/Hindi normalization, intent and crop/entity extraction, query rewriting, HyDE, and decomposition prepare subqueries while preserving farmer terms, quantities, units, location, and uncertainty. Translation is an internal aid; the original query and answer language remain available for citation and safety review.

**Layer 5 — Agentic orchestration.** A stateful LangGraph workflow routes each subquery to vector, lexical, graph, SQL, or approved live-feed tools. It enforces typed state, tool schemas, authorization, timeouts, budgets, maximum six hops, retry limits, and human escalation. ReAct/self-reflection is bounded and observable; agents cannot silently invent facts or execute irreversible actions.

**Layer 6 — Hybrid retrieval.** BM25 + dense + sparse/late-interaction retrieval are fused with Reciprocal Rank Fusion (default `k=60`), followed by metadata filtering, graph expansion, deduplication, and a cross-encoder reranker such as Cohere Rerank 3.5 (or an on-premise equivalent). Contextual compression reduces prompt size only after evidence selection, never by dropping citation spans.

**Layer 7 — Grounded generation.** The answer generator must cite source document, page/section, publisher, and retrieval timestamp. It separates retrieved facts from estimates and recommendations, states when evidence is missing or stale, and uses Self-RAG/CRAG-style relevance and contradiction checks. Agricultural chemical advice includes label compliance, dose/unit checks, PPE, pre-harvest interval, and a qualified-expert escalation path.

**Layer 8 — Delivery and protection.** Output validation applies PII redaction, prompt-injection defense, toxicity and abuse filters, schema checks, citation validation, and farm-safety guardrails. Redis semantic caching is keyed by normalized query, locale, geography, freshness class, and knowledge version; live weather and prices must not be served from an unsafe stale cache. FastAPI streams text over SSE and exposes Marathi TTS; Next.js 15 provides the multilingual, low-bandwidth voice UI.

### Canonical voice-to-answer trace

1. Receive voice/text/image and explicit consent; create a correlation ID.
2. Detect language/script and transcribe with confidence; preserve the original media.
3. Normalize/transliterate Marathi or Hindi while preserving agricultural entities.
4. Classify intent, crop, stage, location, urgency, and safety risk.
5. Rewrite/decompose and optionally generate a HyDE representation.
6. Route subqueries through the bounded LangGraph policy.
7. Retrieve from lexical, dense, vector, graph, SQL, and approved live sources.
8. Fuse, deduplicate, rerank, and compress evidence with citation spans.
9. Generate a structured grounded answer and recommended next action.
10. Run citation, contradiction, freshness, safety, and PII checks.
11. If checks fail, retry with corrective retrieval or return a transparent limitation/escalation.
12. Render text, citations, and Marathi/Hindi/English audio; record feedback and trace metrics.

### Production concerns that are part of the architecture

- **Trust and governance:** source allowlists, license tracking, freshness SLAs, dataset/model versioning, lineage, consent, data minimization, deletion/export, role-based access, tenant isolation, encryption, secrets management, and audit logs.
- **Reliability:** circuit breakers for IMD/mandi APIs, queue-based ingestion, idempotency, rate limits, offline/read-only fallback, backups, restore drills, and regional failover.
- **Evaluation:** golden multilingual queries, temporal holdouts, retrieval recall@k/MRR/nDCG, citation precision, faithfulness, answer correctness, translation quality, vision/OCR accuracy, safety red-team tests, and farmer usability. Track metrics by language, crop, state, and connectivity tier.
- **Observability:** OpenTelemetry traces across STT, retrieval, graph hops, reranking, generation, and TTS; LangSmith/equivalent traces; prompt/model/data version tags; cost, token, cache, latency, error, and feedback dashboards. Never log raw voice or PII by default.
- **Deployment:** containerized FastAPI workers, separate ingestion/indexing and online-serving planes, autoscaling GPU/CPU pools, blue-green model releases, feature flags, canary evaluation, and a model/data rollback path.

### Recommended implementation sequence (10 weeks)

1. **Weeks 1–2 — Foundation:** lakehouse, provenance schema, Qdrant/lexical index, Neo4j, PostgreSQL, BGE-M3 service, secrets and CI.
2. **Weeks 3–4 — Core RAG:** hybrid retrieval, RRF, reranking, citation spans, multilingual golden set, and evaluation harness.
3. **Weeks 5–6 — Graph and agents:** AgriKG ontology, entity linking, bounded LangGraph router, graph traversal, and escalation policy.
4. **Weeks 7–8 — Multimodal/live:** Whisper, OCR/layout, crop-image analysis, Marathi/Hindi TTS, and IMD/mandi adapters with freshness metadata.
5. **Weeks 9–10 — Production:** guardrails, PII/consent, Redis cache, OpenTelemetry, load and failure tests, canary release, and farmer pilot.

**Cost and performance note:** a figure such as `$420/month at 10k QPD` is environment-, model-, token-, storage-, and availability-dependent. Publish it only with a dated workload definition and cloud bill; benchmark text, image, and voice paths separately.
