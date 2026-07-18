from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Generation 10.2 — Multi-Source RAG with dense embeddings/Qdrant, data.gov.in Agmarknet, web tools, GraphRAG"
)

# Request Models
class QueryRequest(BaseModel):
    query: str
    farm_id: Optional[str] = "FARM_101"
    language: Optional[str] = "mr"  # "mr" for Marathi, "en" for English
    enable_web: Optional[bool] = None

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
@app.get("/")
def read_root():
    return {
        "platform": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "Online",
        "architecture": "Multi-Agent GraphRAG + Dense RAG (Qdrant/local) + data.gov.in/Agmarknet + Web Tools + Predictive AI",
        "supported_languages": ["mr (Marathi)", "en (English)", "hi (Hindi)"],
        "embeddings": embedding_provider.info(),
        "opendata": opendata_client.status(),
        "hybrid_backends": hybrid_retriever.backend_info(),
    }

@app.post("/api/query")
def process_farmer_query(request: QueryRequest):
    """Main Agentic Assistant query pipeline handling Multi-Agent planning and Marathi answer synthesis."""
    try:
        response = planner_agent.plan_and_execute(
            query=request.query,
            farm_id=request.farm_id,
            language=request.language,
            enable_web=request.enable_web,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
