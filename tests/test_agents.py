from app.agents.planner import planner_agent
from app.agents.weather_agent import WeatherAgent
from app.agents.disease_agent import DiseaseAgent
from app.agents.market_agent import MarketAgent
from app.agents.soil_agent import SoilAgent
from app.agents.fertilizer_agent import FertilizerAgent

def test_planner_agent_execution():
    res = planner_agent.plan_and_execute(
        query="डाळिंबावरील तेल्या रोगासाठी कोणते औषध फवारावे?",
        farm_id="FARM_101",
        language="mr"
    )
    assert res["farm_id"] == "FARM_101"
    assert len(res["active_agent_names"]) > 0
    assert "synthesized_answer" in res
    assert "तेल्या" in res["synthesized_answer"] or "डाळिंब" in res["synthesized_answer"] or "रोग" in res["synthesized_answer"]

def test_specialized_agents():
    context = {"location": "Pune", "crop": "Pomegranate", "acreage": 2.5}
    
    w_res = WeatherAgent().execute("weather info", context)
    assert w_res["status"] == "success"

    d_res = DiseaseAgent().execute("disease diagnosis", context)
    assert "diagnosis" in d_res

    m_res = MarketAgent().execute("market price", context)
    assert "market_summary" in m_res

    s_res = SoilAgent().execute("soil card", context)
    assert "soil_analysis" in s_res

    f_res = FertilizerAgent().execute("fertilizer plan", context)
    assert "fertilizer_plan" in f_res
