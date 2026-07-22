from pathlib import Path

from fastapi import Body, FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from app.config import settings
from app.agents.planner import planner_agent
from app.live_feeds.weather_feed import weather_feed
from app.live_feeds.market_feed import market_feed
from app.live_feeds.iot_feed import iot_feed
from app.live_feeds.satellite_feed import satellite_feed
from app.vision.disease_classifier import vision_classifier
from app.vision.ocr_processor import ocr_processor
from app.predictive.yield_model import yield_model
from app.predictive.pest_outbreak_model import pest_outbreak_model
from app.predictive.irrigation_model import irrigation_model
from app.predictive.fertilizer_planner import fertilizer_planner
from app.workflows.automation import workflow_engine
from app.memory.farm_memory import farm_memory_store
from app.knowledge.graph_rag import graph_rag
from app.knowledge.hybrid_search import hybrid_retriever
from app.knowledge.advanced_rag import advanced_rag
from app.knowledge.dataset_loader import kb_loader
from app.knowledge.tools.registry import tool_registry
from app.knowledge.embeddings import embedding_provider
from app.live_feeds.opendata_client import opendata_client
from app import ui_dashboard

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Generation 10.2 — Multi-Source RAG with dense embeddings/Qdrant, data.gov.in Agmarknet, web tools, GraphRAG"
)


@app.on_event("startup")
def preload_local_models():
    """Warm-up local KrushiVerse-AI LLM model on startup for 10x faster response times."""
    try:
        import torch
        from mini.eval.harness import load_checkpoint, resolve_model_dir
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = resolve_model_dir("v0.4-agri-qa")
        load_checkpoint(model_dir, device=dev)
    except Exception:
        pass

# Dev + production browser UI (React SPA in ui/web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_DIST = Path(__file__).resolve().parent.parent / "ui" / "web" / "dist"
UI_PUBLIC = Path(__file__).resolve().parent.parent / "ui" / "web" / "public"

# Request Models
class QueryRequest(BaseModel):
    query: str
    farm_id: Optional[str] = "FARM_101"
    language: Optional[str] = "mr"  # "mr" for Marathi, "en" for English
    enable_web: Optional[bool] = None
    use_local_llm: Optional[bool] = None  # True to force local KrushiVerse-AI model (v2-12M-fixed)

class MiniChatRequest(BaseModel):
    query: str
    language: Optional[str] = "en"
    crop: Optional[str] = None
    location: Optional[str] = "Pune"
    mode: Optional[str] = None  # grounded | open
    enable_web: Optional[bool] = False
    enable_agents: Optional[bool] = True
    max_new_tokens: Optional[int] = None
    seed: Optional[int] = 42

class AdvancedRAGRequest(BaseModel):
    query: str
    crop: Optional[str] = None
    location: Optional[str] = "Pune"
    top_k: Optional[int] = 8
    enable_web: Optional[bool] = True
    enable_tools: Optional[bool] = True
    force_web: Optional[bool] = False

class YieldPredictRequest(BaseModel):
    crop: str
    acreage: float = 2.0
    N_status: Optional[str] = "Medium"
    P_status: Optional[str] = "Medium"
    K_status: Optional[str] = "Medium"
    irrigation_quality: Optional[str] = "Excellent Drip"

class IrrigationPredictRequest(BaseModel):
    crop: str
    acreage: float = 2.0
    temperature_c: float = 30.0
    humidity_pct: float = 75.0

class FertilizerPredictRequest(BaseModel):
    crop: str
    acreage: float = 2.0
    N_kg_ha: float = 180.0
    P_kg_ha: float = 22.0
    K_kg_ha: float = 280.0

# Endpoints
@app.get("/api/health")
def api_health():
    return {
        "platform": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "Online",
        "local_model": "v2-12M-fixed (10 layers, 8192 vocab, 512 block)",
        "architecture": "Multi-Agent GraphRAG + KrushiVerse-AI Local LLM + Dense RAG (Qdrant/local) + data.gov.in/Agmarknet + Web Tools + Predictive AI",
        "supported_languages": ["mr (Marathi)", "en (English)", "hi (Hindi)"],
        "embeddings": embedding_provider.info(),
        "opendata": opendata_client.status(),
        "hybrid_backends": hybrid_retriever.backend_info(),
        "ui": "built" if (UI_DIST / "index.html").exists() else "run npm run build in ui/web",
    }

@app.get("/")
def read_root():
    """Prefer React OS UI when built; otherwise return API health JSON."""
    index = UI_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return api_health()

@app.post("/api/query")
def process_farmer_query(request: QueryRequest):
    """Main Agentic Assistant query pipeline handling Multi-Agent planning and answer synthesis."""
    try:
        response = planner_agent.plan_and_execute(
            query=request.query,
            farm_id=request.farm_id,
            language=request.language,
            enable_web=request.enable_web,
            use_local_llm=request.use_local_llm,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mini/status")
def mini_status():
    """Mini product integration flags + latest infer report (Sprint 16 / FP-9)."""
    from mini.paths import INFERENCE_DIR, MODELS_DIR
    import json
    from mini import __feature_phase__, __sprint__, __version__ as mini_ver

    latest = INFERENCE_DIR / "INFER_LATEST.json"
    infer = None
    if latest.exists():
        try:
            infer = json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            infer = {"ok": False, "error": "unreadable INFER_LATEST"}
    return {
        "ok": True,
        "mini_version": mini_ver,
        "sprint": __sprint__,
        "feature_phase": __feature_phase__,
        "use_mini_llm": settings.USE_MINI_LLM,
        "mini_default_mode": settings.MINI_DEFAULT_MODE,
        "mini_model_version": settings.MINI_MODEL_VERSION,
        "serve_dir_exists": (MODELS_DIR / "serve").exists(),
        "latest_infer": {
            "ok": (infer or {}).get("ok"),
            "engine": (infer or {}).get("engine"),
            "n_sources": (infer or {}).get("n_sources"),
            "created_at": (infer or {}).get("created_at"),
        }
        if infer
        else None,
    }

@app.post("/api/mini/chat")
def mini_chat(request: MiniChatRequest):
    """Direct Mini+RAG chat endpoint (Sprint 16). Always uses Mini chain (not gated by USE_MINI_LLM)."""
    try:
        from app.llm.mini_bridge import run_mini_chat

        return run_mini_chat(
            request.query,
            language=request.language or "en",
            crop=request.crop,
            location=request.location or "Pune",
            mode=request.mode or settings.MINI_DEFAULT_MODE,
            enable_web=bool(request.enable_web) if request.enable_web is not None else False,
            enable_agents=bool(request.enable_agents) if request.enable_agents is not None else True,
            max_new_tokens=request.max_new_tokens,
            seed=int(request.seed or 42),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mini/release")
def mini_release_status():
    """Latest Mini v1.0 release gate report (Sprint 17)."""
    from mini.paths import REPO_ROOT
    import json

    latest = REPO_ROOT / "mini" / "release" / "RELEASE_LATEST.json"
    checklist = REPO_ROOT / "mini" / "release" / "CHECKLIST_SIGNED.json"
    out: dict = {}
    if latest.exists():
        out = json.loads(latest.read_text(encoding="utf-8"))
    if checklist.exists():
        out["checklist_signed"] = json.loads(checklist.read_text(encoding="utf-8")).get("summary")
    if out:
        return out
    return {"ok": False, "message": "No release yet. POST /api/mini/release?execute=true"}

@app.post("/api/mini/release")
def mini_release_run(
    execute: bool = False,
    run_eval: bool = True,
    run_smoke: bool = True,
    eval_version: str = "v0.4",
    smoke_rounds: int = 2,
    seed: int = 42,
):
    """Run W-RELEASE v1.0 RC gate (checklist + optional eval + load smoke)."""
    from mini.workers.base import get_worker

    result = get_worker("W-RELEASE").run(
        dry_run=not execute,
        run_eval=run_eval,
        run_smoke=run_smoke,
        eval_version=eval_version,
        smoke_rounds=smoke_rounds,
        seed=seed,
    )
    return result.model_dump()

@app.post("/api/rag/advanced")
def advanced_rag_search(request: AdvancedRAGRequest):
    """Advanced multi-source RAG: local hybrid + GraphRAG + external tools + web search fusion."""
    try:
        return advanced_rag.retrieve(
            request.query,
            crop=request.crop,
            location=request.location or "Pune",
            top_k=request.top_k or settings.RAG_TOP_K,
            enable_web=request.enable_web if request.enable_web is not None else settings.ENABLE_WEB_RAG,
            enable_tools=request.enable_tools if request.enable_tools is not None else settings.ENABLE_TOOL_RAG,
            force_web=bool(request.force_web),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge/stats")
def knowledge_stats():
    """Expanded open-source knowledge base statistics."""
    return {
        "version": settings.VERSION,
        "stats": kb_loader.knowledge_stats(),
        "web_rag_enabled": settings.ENABLE_WEB_RAG,
        "tool_rag_enabled": settings.ENABLE_TOOL_RAG,
        "dense_rag_enabled": settings.ENABLE_DENSE_RAG,
        "embeddings": embedding_provider.info(),
        "hybrid_backends": hybrid_retriever.backend_info(),
        "opendata": opendata_client.status(),
        "tools": tool_registry.list_tools(),
    }

@app.get("/api/opendata/agmarknet")
def agmarknet_prices(
    commodity: Optional[str] = None,
    district: Optional[str] = None,
    state: str = "Maharashtra",
    limit: int = 40,
):
    """Live Agmarknet commodity prices via data.gov.in (falls back to local open KB)."""
    return opendata_client.fetch_commodity_prices(
        state=state,
        district=district,
        commodity=commodity,
        limit=limit,
    )

@app.get("/api/rag/backends")
def rag_backends():
    return {
        "version": settings.VERSION,
        "embeddings": embedding_provider.info(),
        "hybrid": hybrid_retriever.backend_info(),
        "opendata": opendata_client.status(),
        "qdrant_url": settings.QDRANT_URL,
        "collection": settings.QDRANT_COLLECTION,
    }

@app.get("/api/taxonomy")
def taxonomy_summary_api():
    """Frozen agriculture domain taxonomy (Sprint 1)."""
    from mini.taxonomy.service import taxonomy_service

    return {
        "summary": taxonomy_service.summary(),
        "categories": taxonomy_service.category_details(),
        "crops": taxonomy_service.crops(),
        "stages": taxonomy_service.stages(),
        "languages": ["en", "mr", "hi"],
    }

@app.get("/api/taxonomy/validate")
def taxonomy_validate_api():
    from mini.taxonomy.service import taxonomy_service

    report = taxonomy_service.validate()
    if not report.get("ok"):
        # still 200 with ok=false so clients can display errors; use 422 if preferred
        return report
    return report

@app.get("/api/taxonomy/resolve")
def taxonomy_resolve(crop: Optional[str] = None, district: Optional[str] = None, text: Optional[str] = None):
    from mini.taxonomy.service import taxonomy_service

    return {
        "crop": taxonomy_service.resolve_crop(crop or text or "") if (crop or text) else None,
        "crops_in_text": taxonomy_service.extract_crops(text or crop or ""),
        "region": taxonomy_service.resolve_region(district=district) if district else None,
        "categories": taxonomy_service.detect_category(text or crop or "") if (text or crop) else [],
    }

@app.get("/api/lake/status")
def lake_status_api():
    """Data lake raw inventory (Sprint 2)."""
    from mini.lake.ingest import lake_tree_summary
    from mini.lake.registry import load_source_registry

    return {
        "registry": load_source_registry().summary(),
        "lake": lake_tree_summary(),
    }

@app.post("/api/lake/ingest")
def lake_ingest_api(execute: bool = False, skip_http: bool = False):
    """Trigger W-INGEST (default dry-run unless execute=true)."""
    from mini.workers.base import get_worker

    result = get_worker("W-INGEST").run(dry_run=not execute, include_http=not skip_http)
    return result.model_dump()

@app.get("/api/lake/quality")
def lake_quality_status():
    """Latest quality pipeline report if present."""
    from pathlib import Path
    from mini.paths import LAKE_ROOT
    import json

    latest = LAKE_ROOT / "QUALITY_LATEST.json"
    if not latest.exists():
        return {"ok": False, "message": "No quality report yet. Run POST /api/lake/quality?execute=true"}
    return json.loads(latest.read_text(encoding="utf-8"))

@app.post("/api/lake/quality")
def lake_quality_run(execute: bool = False, near_threshold: float = 0.92):
    """Run validate → clean → dedup quality pipeline."""
    from mini.workers.base import get_worker

    result = get_worker("W-QUALITY").run(
        dry_run=not execute,
        near_threshold=near_threshold,
    )
    return result.model_dump()

@app.get("/api/lake/standard")
def lake_standard_status():
    """Latest standardized dataset export report."""
    from pathlib import Path
    from mini.paths import LAKE_ROOT, DATASETS_DIR
    import json

    latest = LAKE_ROOT / "STANDARD_LATEST.json"
    ver = DATASETS_DIR / "LATEST_VERSION.json"
    out = {"ok": latest.exists() or ver.exists()}
    if latest.exists():
        out["standard"] = json.loads(latest.read_text(encoding="utf-8"))
    if ver.exists():
        out["latest_version"] = json.loads(ver.read_text(encoding="utf-8"))
    if not out["ok"]:
        out["message"] = "No standard export yet. POST /api/lake/standard?execute=true"
    return out

@app.post("/api/lake/standard")
def lake_standard_run(execute: bool = False):
    """Extract Schema v1 StandardRecords and export train/val/test (+ parquet)."""
    from mini.workers.base import get_worker

    result = get_worker("W-STANDARDIZE").run(dry_run=not execute)
    return result.model_dump()

@app.get("/api/lake/analyze")
def lake_analyze_status():
    """Latest coverage analysis report."""
    from mini.paths import LAKE_ROOT
    import json

    latest = LAKE_ROOT / "ANALYZE_LATEST.json"
    if not latest.exists():
        return {
            "ok": False,
            "message": "No analysis yet. POST /api/lake/analyze?execute=true",
        }
    return json.loads(latest.read_text(encoding="utf-8"))

@app.post("/api/lake/analyze")
def lake_analyze_run(execute: bool = False):
    """Run W-ANALYZE coverage/quality intelligence."""
    from mini.workers.base import get_worker

    result = get_worker("W-ANALYZE").run(dry_run=not execute)
    return result.model_dump()

@app.get("/api/lake/qasynth")
def lake_qasynth_status():
    """Latest QA synthesis export report."""
    from mini.paths import LAKE_ROOT
    import json

    latest = LAKE_ROOT / "QASYNTH_LATEST.json"
    if not latest.exists():
        return {"ok": False, "message": "No synth run yet. POST /api/lake/qasynth?execute=true"}
    return json.loads(latest.read_text(encoding="utf-8"))

@app.post("/api/lake/qasynth")
def lake_qasynth_run(execute: bool = False, target: int = 62500):
    """Run W-QASYNTH multilingual expert QA synthesis (S7 default ≥50k train)."""
    from mini.workers.base import get_worker

    result = get_worker("W-QASYNTH").run(dry_run=not execute, target_min_total=target)
    return result.model_dump()

@app.get("/api/lake/kg")
def lake_kg_status():
    """Latest knowledge graph build report (W-KGBUILD)."""
    from mini.paths import DATASETS_DIR, LAKE_ROOT
    import json

    for path in (DATASETS_DIR / "KG_LATEST.json", LAKE_ROOT / "KG_LATEST.json"):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            # full graph file has meta; lake marker may be meta-only
            if "meta" in data:
                return {"ok": True, **(data.get("meta") or {}), "path": str(path)}
            return {"ok": True, **data, "path": str(path)}
    return {"ok": False, "message": "No KG build yet. POST /api/lake/kg?execute=true"}

@app.post("/api/lake/kg")
def lake_kg_run(execute: bool = False, write_seed: bool = False):
    """Run W-KGBUILD automated knowledge graph construction (S8).

    write_seed defaults false on API so platform seed is not overwritten unless requested.
    """
    from mini.workers.base import get_worker

    result = get_worker("W-KGBUILD").run(
        dry_run=not execute,
        write_platform_seed=write_seed,
    )
    return result.model_dump()

@app.get("/api/lake/tokenizer")
def lake_tokenizer_status():
    """Latest domain tokenizer report (W-TOKEN)."""
    from mini.paths import TOKENIZER_DIR
    import json

    latest = TOKENIZER_DIR / "TOKENIZER_LATEST.json"
    if not latest.exists():
        return {"ok": False, "message": "No tokenizer yet. POST /api/lake/tokenizer?execute=true"}
    return json.loads(latest.read_text(encoding="utf-8"))

@app.post("/api/lake/tokenizer")
def lake_tokenizer_run(execute: bool = False, vocab_size: int = 32000, version: str = "v0.1"):
    """Run W-TOKEN agriculture SentencePiece training (S9 default 32k vocab)."""
    from mini.workers.base import get_worker

    result = get_worker("W-TOKEN").run(
        dry_run=not execute,
        vocab_size=vocab_size,
        version=version,
    )
    return result.model_dump()

@app.get("/api/lake/pretrain")
def lake_pretrain_status():
    """Latest Mini pretrain/smoke report (W-PRETRAIN)."""
    from mini.paths import MODELS_DIR
    import json

    latest = MODELS_DIR / "PRETRAIN_LATEST.json"
    param = MODELS_DIR / "PARAM_COUNT.json"
    if latest.exists():
        return json.loads(latest.read_text(encoding="utf-8"))
    if param.exists():
        return {"ok": True, "param_count": json.loads(param.read_text(encoding="utf-8"))}
    return {"ok": False, "message": "No pretrain run yet. POST /api/lake/pretrain?execute=true"}

@app.post("/api/lake/pretrain")
def lake_pretrain_run(
    execute: bool = False,
    steps: int = 200,
    mode: str = "domain",
    seed: int = 42,
    batch_size: int = 8,
):
    """Run W-PRETRAIN domain pretrain (S11 v0.2-base) or smoke."""
    from mini.workers.base import get_worker

    result = get_worker("W-PRETRAIN").run(
        dry_run=not execute,
        mode=mode,
        steps=steps,
        seed=seed,
        batch_size=batch_size,
    )
    return result.model_dump()

@app.get("/api/lake/sft")
def lake_sft_status():
    """Latest Mini SFT report (W-SFT v0.3/v0.4)."""
    from mini.paths import MODELS_DIR
    import json

    latest = MODELS_DIR / "SFT_LATEST.json"
    if latest.exists():
        return json.loads(latest.read_text(encoding="utf-8"))
    return {"ok": False, "message": "No SFT run yet. POST /api/lake/sft?execute=true"}

@app.post("/api/lake/sft")
def lake_sft_run(
    execute: bool = False,
    steps_v03: int = 120,
    steps_v04: int = 120,
    seed: int = 42,
    batch_size: int = 4,
    max_train: int = 4000,
    max_val: int = 400,
):
    """Run W-SFT instruction + agri-QA fine-tune (S12 v0.3-instruct → v0.4-agri-qa)."""
    from mini.workers.base import get_worker

    result = get_worker("W-SFT").run(
        dry_run=not execute,
        steps_v03=steps_v03,
        steps_v04=steps_v04,
        seed=seed,
        batch_size=batch_size,
        max_train=max_train,
        max_val=max_val,
    )
    return result.model_dump()

@app.get("/api/lake/eval")
def lake_eval_status():
    """Latest Mini eval scorecard (W-EVAL)."""
    from mini.paths import EVAL_DIR
    import json

    latest = EVAL_DIR / "EVAL_LATEST.json"
    if latest.exists():
        return json.loads(latest.read_text(encoding="utf-8"))
    return {"ok": False, "message": "No eval run yet. POST /api/lake/eval?execute=true"}

@app.post("/api/lake/eval")
def lake_eval_run(
    execute: bool = False,
    version: str = "v0.4",
    profile: str = "default",
    seed: int = 42,
    max_new_tokens: int = 28,
    max_gold: Optional[int] = None,
):
    """Run W-EVAL gold + probes + gates (S13). ok=false when gates fail."""
    from mini.workers.base import get_worker

    result = get_worker("W-EVAL").run(
        dry_run=not execute,
        version=version,
        gate_profile=profile,
        seed=seed,
        max_new_tokens=max_new_tokens,
        max_gold=max_gold,
    )
    return result.model_dump()

@app.get("/api/lake/quant")
def lake_quant_status():
    """Latest Mini quantization report (W-QUANT)."""
    from mini.paths import MODELS_DIR
    import json

    latest = MODELS_DIR / "QUANT_LATEST.json"
    if latest.exists():
        return json.loads(latest.read_text(encoding="utf-8"))
    return {"ok": False, "message": "No quant run yet. POST /api/lake/quant?execute=true"}

@app.post("/api/lake/quant")
def lake_quant_run(
    execute: bool = False,
    version: str = "v0.4",
    include_int4: bool = True,
    seed: int = 42,
    latency_runs: int = 6,
):
    """Run W-QUANT INT8/INT4 export + size/latency benchmarks (S14)."""
    from mini.workers.base import get_worker

    result = get_worker("W-QUANT").run(
        dry_run=not execute,
        version=version,
        include_int4=include_int4,
        seed=seed,
        latency_runs=latency_runs,
    )
    return result.model_dump()

@app.get("/api/lake/deploy")
def lake_deploy_status():
    """Latest Mini deploy package / version registry (W-DEPLOY)."""
    from mini.paths import MODELS_DIR
    import json

    latest = MODELS_DIR / "DEPLOY_LATEST.json"
    reg = MODELS_DIR / "VERSION_REGISTRY.json"
    out: dict = {}
    if latest.exists():
        out = json.loads(latest.read_text(encoding="utf-8"))
    if reg.exists():
        out["registry"] = json.loads(reg.read_text(encoding="utf-8"))
    if out:
        return out
    return {"ok": False, "message": "No deploy package yet. POST /api/lake/deploy?execute=true"}

@app.post("/api/lake/deploy")
def lake_deploy_run(
    execute: bool = False,
    version: str = "v0.4",
    tag: str = "v0.5-quant",
    force: bool = True,
    include_quant: bool = True,
    reasoning_lite: bool = True,
):
    """Run W-DEPLOY package-only publish (S14)."""
    from mini.workers.base import get_worker

    result = get_worker("W-DEPLOY").run(
        dry_run=not execute,
        source_version=version,
        tag=tag,
        force=force,
        include_quant=include_quant,
        reasoning_lite=reasoning_lite,
    )
    return result.model_dump()

@app.get("/api/lake/infer")
def lake_infer_status():
    """Latest Mini inference report (W-INFER)."""
    from mini.paths import INFERENCE_DIR
    import json

    latest = INFERENCE_DIR / "INFER_LATEST.json"
    if latest.exists():
        return json.loads(latest.read_text(encoding="utf-8"))
    return {"ok": False, "message": "No infer run yet. POST /api/lake/infer?execute=true"}

@app.post("/api/lake/infer")
def lake_infer_run(
    execute: bool = False,
    query: str = "How do I manage pink bollworm in cotton with IPM in Maharashtra?",
    mode: str = "grounded",
    crop: Optional[str] = None,
    location: str = "Pune",
    version: str = "auto",
    enable_web: bool = False,
    enable_agents: bool = True,
    max_new_tokens: int = 40,
    seed: int = 42,
):
    """Run W-INFER grounded chain: intent → RAG → Mini → validate (S15)."""
    from mini.workers.base import get_worker

    result = get_worker("W-INFER").run(
        dry_run=not execute,
        query=query,
        mode=mode,
        crop=crop,
        location=location,
        version=version,
        enable_web=enable_web,
        enable_agents=enable_agents,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    return result.model_dump()

@app.get("/api/tools")
def list_external_tools():
    return {"tools": tool_registry.list_tools()}

@app.post("/api/tools/{tool_name}")
def run_external_tool(tool_name: str, params: Optional[dict] = None):
    result = tool_registry.run(tool_name, params or {})
    if not result.get("ok") and result.get("error", "").startswith("Unknown tool"):
        raise HTTPException(status_code=404, detail=result.get("error"))
    return result

@app.post("/api/vision/diagnose")
async def diagnose_leaf_image(
    file: Optional[UploadFile] = File(None),
    crop_hint: Optional[str] = Form("Pomegranate")
):
    """Vision AI endpoint for plant disease classification from uploaded leaf images."""
    filename = file.filename if file else "leaf_sample.jpg"
    image_bytes = await file.read() if file else None
    
    result = vision_classifier.diagnose_image(image_bytes=image_bytes, filename=filename, crop_hint=crop_hint)
    return result

@app.post("/api/soil/ocr")
def process_soil_health_card(raw_ocr_text: Optional[str] = Form(None)):
    """OCR processing endpoint for Soil Health Cards."""
    sample_text = raw_ocr_text or "pH: 7.4, EC: 0.45, Organic Carbon: 0.52%, Nitrogen: 180 kg/ha, Phosphorus: 22 kg/ha, Potassium: 280 kg/ha"
    result = ocr_processor.process_soil_card(sample_text)
    return result

@app.get("/api/live/weather")
def get_weather(location: str = "Pune"):
    return weather_feed.get_weather(location)

@app.get("/api/live/market")
def get_market(crop: Optional[str] = None, district: Optional[str] = None):
    return market_feed.get_market_prices(crop=crop, district=district)

@app.get("/api/live/iot")
def get_iot_telemetry(farm_id: str = "FARM_101"):
    return iot_feed.get_sensor_telemetry(farm_id)

@app.get("/api/live/satellite")
def get_satellite_telemetry(farm_id: str = "FARM_101", crop: str = "Pomegranate"):
    return satellite_feed.get_satellite_indices(farm_id=farm_id, crop_name=crop)

@app.post("/api/predict/yield")
def predict_crop_yield(req: YieldPredictRequest):
    return yield_model.predict_yield(
        crop=req.crop,
        acreage=req.acreage,
        N_status=req.N_status,
        P_status=req.P_status,
        K_status=req.K_status,
        irrigation_quality=req.irrigation_quality
    )

@app.post("/api/predict/irrigation")
def predict_irrigation_schedule(req: IrrigationPredictRequest):
    return irrigation_model.calculate_water_requirement(
        crop=req.crop,
        acreage=req.acreage,
        temperature_c=req.temperature_c,
        humidity_pct=req.humidity_pct
    )

@app.post("/api/predict/fertilizer")
def predict_fertilizer_schedule(req: FertilizerPredictRequest):
    return fertilizer_planner.calculate_fertilizer_bags(
        crop=req.crop,
        acreage=req.acreage,
        N_kg_ha=req.N_kg_ha,
        P_kg_ha=req.P_kg_ha,
        K_kg_ha=req.K_kg_ha
    )

@app.get("/api/memory/{farm_id}")
def get_farm_memory(farm_id: str):
    farm = farm_memory_store.get_farm(farm_id)
    if not farm:
        raise HTTPException(status_code=404, detail=f"Farm ID {farm_id} not found.")
    return farm

@app.get("/api/knowledge/graph/{crop}")
def get_crop_graph(crop: str):
    return graph_rag.get_crop_ecosystem(crop)

@app.post("/api/workflows/audit")
def audit_farm_workflows(farm_id: str = "FARM_101"):
    return workflow_engine.run_farm_health_checks(farm_id)


# --- Unified React OS dashboard data (live when available, else managed static) ---
@app.get("/api/ui/bootstrap")
def ui_bootstrap(farm_id: str = "FARM_101"):
    return ui_dashboard.bootstrap(farm_id=farm_id)

@app.get("/api/ui/live")
def ui_live_feeds(farm_id: str = "FARM_101", location: str = "Solapur", crop: str = "Pomegranate"):
    return ui_dashboard.live_feeds(farm_id=farm_id, location=location, crop=crop)

@app.get("/api/ui/vision/samples")
def ui_vision_samples():
    return ui_dashboard.vision_samples()

@app.get("/api/ui/graph/{crop}")
def ui_graph(crop: str):
    return ui_dashboard.graph_for_ui(crop)

@app.post("/api/ui/soil")
def ui_soil_plan(payload: dict = Body(default={})):
    body = payload or {}
    return ui_dashboard.soil_plan(
        crop=body.get("crop") or "Pomegranate",
        acreage=float(body.get("acreage") or body.get("acres") or 2.5),
        soil_text=body.get("soil_text"),
        farm_id=body.get("farm_id") or "FARM_101",
    )

@app.post("/api/ui/predict")
def ui_predict(payload: dict = Body(default={})):
    body = payload or {}
    return ui_dashboard.predictive(
        crop=body.get("crop") or "Pomegranate",
        acreage=float(body.get("acreage") or body.get("acres") or 2.5),
        temperature_c=float(body.get("temperature_c") or 30),
        humidity_pct=float(body.get("humidity_pct") or 75),
        farm_id=body.get("farm_id") or "FARM_101",
    )

@app.get("/api/ui/taxonomy")
def ui_taxonomy():
    return ui_dashboard.taxonomy_bundle()

@app.get("/api/ui/factory")
def ui_factory():
    return ui_dashboard.factory_status()

@app.post("/api/ui/rag")
def ui_rag(payload: dict = Body(default={})):
    body = payload or {}
    return ui_dashboard.rag_explorer(
        query=body.get("query") or "",
        crop=body.get("crop"),
        top_k=int(body.get("top_k") or 8),
        enable_web=bool(body.get("enable_web", True)),
        enable_tools=bool(body.get("enable_tools", True)),
    )


# --- React OS UI (built assets from ui/web) ---
if UI_PUBLIC.exists():
    # public images (field aerial, leaf samples) available at /leaves/*, /field-aerial.jpg
    app.mount("/leaves", StaticFiles(directory=str(UI_PUBLIC / "leaves")), name="leaf-images")

    @app.get("/field-aerial.jpg")
    def field_aerial():
        p = UI_PUBLIC / "field-aerial.jpg"
        if p.exists():
            return FileResponse(p)
        raise HTTPException(404, "field image missing")


@app.get("/ui")
@app.get("/ui/")
@app.get("/dashboard")
def serve_ui_root():
    index = UI_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(
        status_code=503,
        detail="UI not built. From repo: cd ui/web && npm install && npm run build",
    )


@app.get("/ui/{asset_path:path}")
def serve_ui_assets(asset_path: str):
    """SPA fallback for nested client routes (if any)."""
    candidate = UI_DIST / asset_path
    if candidate.is_file():
        return FileResponse(candidate)
    index = UI_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(404, "UI asset not found")
