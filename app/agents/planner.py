from app.agents.weather_agent import WeatherAgent
from app.agents.crop_agent import CropAgent
from app.agents.disease_agent import DiseaseAgent
from app.agents.market_agent import MarketAgent
from app.agents.soil_agent import SoilAgent
from app.agents.fertilizer_agent import FertilizerAgent
from app.agents.government_agent import GovernmentAgent
from app.agents.vision_agent import VisionAgent
from app.agents.finance_agent import FinanceAgent

from app.knowledge.hybrid_search import hybrid_retriever
from app.knowledge.graph_rag import graph_rag
from app.memory.farm_memory import farm_memory_store
from app.llm.generator import response_synthesizer

class PlannerAgent:
    """Central Planner Agent orchestrating specialized sub-agents and knowledge retrieval."""

    def __init__(self):
        self.weather_agent = WeatherAgent()
        self.crop_agent = CropAgent()
        self.disease_agent = DiseaseAgent()
        self.market_agent = MarketAgent()
        self.soil_agent = SoilAgent()
        self.fertilizer_agent = FertilizerAgent()
        self.government_agent = GovernmentAgent()
        self.vision_agent = VisionAgent()
        self.finance_agent = FinanceAgent()

    def plan_and_execute(self, query: str, farm_id: str = "FARM_101", image_filename: str | None = None, language: str = "mr") -> dict:
        # Retrieve farm memory context
        farm = farm_memory_store.get_farm(farm_id) or farm_memory_store.get_farm("FARM_101")
        
        crop_name = farm["current_crop"].get("crop_name", "Pomegranate")
        district = farm["location"].get("district", "Pune")
        soil_profile = farm.get("soil_profile", {})

        agent_context = {
            "query": query,
            "crop": crop_name,
            "location": district,
            "district": district,
            "acreage": farm.get("land_area_acres", 2.0),
            "soil_card_text": f"pH: {soil_profile.get('pH', 7.2)}, Nitrogen: {soil_profile.get('nitrogen_kg_ha', 180)} kg/ha, Phosphorus: {soil_profile.get('phosphorus_kg_ha', 22)} kg/ha, Potassium: {soil_profile.get('potassium_kg_ha', 280)} kg/ha",
            "image_filename": image_filename or "leaf_sample.jpg"
        }

        # Step 1: Knowledge Layer Retrieval (Hybrid Search + GraphRAG)
        hybrid_docs = hybrid_retriever.hybrid_search(f"{query} {crop_name}", top_k=3)
        graph_data = graph_rag.get_crop_ecosystem(crop_name)

        # Step 2: Query intent analysis and agent selection
        query_lower = query.lower()
        active_agents = []

        if "weather" in query_lower or "rain" in query_lower or "पाऊस" in query_lower or "हवामान" in query_lower:
            active_agents.append(self.weather_agent)

        if "disease" in query_lower or "pest" in query_lower or "spot" in query_lower or "रोग" in query_lower or "कीड" in query_lower or image_filename:
            active_agents.extend([self.disease_agent, self.vision_agent, self.weather_agent])

        if "market" in query_lower or "price" in query_lower or "mandi" in query_lower or "भाव" in query_lower or "बाजार" in query_lower:
            active_agents.append(self.market_agent)

        if "fertilizer" in query_lower or "soil" in query_lower or "npk" in query_lower or "खत" in query_lower or "माती" in query_lower:
            active_agents.extend([self.soil_agent, self.fertilizer_agent])

        if "scheme" in query_lower or "subsidy" in query_lower or "yojana" in query_lower or "योजना" in query_lower or "अनुदान" in query_lower:
            active_agents.append(self.government_agent)

        # Default comprehensive execution if query is general
        if not active_agents:
            active_agents = [
                self.weather_agent,
                self.crop_agent,
                self.disease_agent,
                self.market_agent,
                self.soil_agent,
                self.fertilizer_agent,
                self.government_agent,
                self.finance_agent
            ]

        # Deduplicate agents
        active_agents = list({ag.name: ag for ag in active_agents}.values())

        # Step 3: Execute active agents
        agent_results = {}
        for ag in active_agents:
            res = ag.execute(query, agent_context)
            key = ag.name.lower().replace(" agent", "")
            agent_results[key] = res.get("data") or res.get("diagnosis") or res.get("market_summary") or res.get("soil_analysis") or res.get("fertilizer_plan") or res.get("applicable_schemes") or res.get("financial_summary") or res

        # Step 4: Synthesize Marathi / English Answer
        plan_summary = f"Processed farmer query '{query}' for {crop_name} in {district}. Coordinated {len(active_agents)} specialized sub-agents with Knowledge Layer (Hybrid RAG & GraphRAG)."
        final_answer = response_synthesizer.synthesize(plan_summary, agent_results, language=language)

        # Log action in Farm Memory
        farm_memory_store.log_action(
            farm_id=farm["farm_id"],
            action_type="Farmer Query Handled",
            details=f"Query: '{query}'. Active agents: {[ag.name for ag in active_agents]}"
        )

        return {
            "query": query,
            "farm_id": farm["farm_id"],
            "farmer_name": farm["farmer_name"],
            "crop": crop_name,
            "language": language,
            "active_agent_names": [ag.name for ag in active_agents],
            "knowledge_layer": {
                "hybrid_rag_hits": len(hybrid_docs),
                "graph_rag_ecosystem": graph_data
            },
            "agent_outputs": agent_results,
            "synthesized_answer": final_answer
        }

planner_agent = PlannerAgent()
