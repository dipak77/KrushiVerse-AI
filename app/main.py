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

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Generation 10 — Autonomous AI Agriculture Platform (AI Krushi Mitra)"
)

# Request Models
class QueryRequest(BaseModel):
    query: str
    farm_id: Optional[str] = "FARM_101"
    language: Optional[str] = "mr"  # "mr" for Marathi, "en" for English

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
        "architecture": "Multi-Agent GraphRAG + Live RAG + Computer Vision + Predictive AI",
        "supported_languages": ["mr (Marathi)", "en (English)", "hi (Hindi)"]
    }

@app.post("/api/query")
def process_farmer_query(request: QueryRequest):
    """Main Agentic Assistant query pipeline handling Multi-Agent planning and Marathi answer synthesis."""
    try:
        response = planner_agent.plan_and_execute(
            query=request.query,
            farm_id=request.farm_id,
            language=request.language
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
