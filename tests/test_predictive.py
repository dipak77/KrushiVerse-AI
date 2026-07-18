from app.predictive.yield_model import yield_model
from app.predictive.pest_outbreak_model import pest_outbreak_model
from app.predictive.irrigation_model import irrigation_model
from app.predictive.fertilizer_planner import fertilizer_planner

def test_yield_model():
    yp = yield_model.predict_yield(crop="Cotton", acreage=2.0, N_status="Deficient")
    assert yp["acreage"] == 2.0
    assert yp["total_predicted_yield"] > 0

def test_pest_outbreak_model():
    p_risk = pest_outbreak_model.calculate_outbreak_risk("Pomegranate", temperature_c=30.0, humidity_pct=85.0, rainfall_mm=12.0)
    assert len(p_risk["assessed_risks"]) > 0
    assert p_risk["assessed_risks"][0]["risk_percentage"] > 50

def test_irrigation_model():
    req = irrigation_model.calculate_water_requirement(crop="Soybean", acreage=2.0, temperature_c=32.0, humidity_pct=60.0)
    assert req["total_farm_water_required_liters_day"] > 0
    assert req["drip_irrigation_schedule"]["drip_runtime_hours_per_day"] > 0

def test_fertilizer_planner():
    f_plan = fertilizer_planner.calculate_fertilizer_bags(crop="Cotton", acreage=2.0, N_kg_ha=180, P_kg_ha=22, K_kg_ha=280)
    bags = f_plan["recommended_fertilizer_bags"]
    assert bags["Urea_45kg_bags"] > 0
    assert bags["DAP_50kg_bags"] > 0
