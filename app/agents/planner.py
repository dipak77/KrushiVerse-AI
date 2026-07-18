from app.agents.weather_agent import WeatherAgent
from app.agents.crop_agent import CropAgent
from app.agents.disease_agent import DiseaseAgent
from app.agents.market_agent import MarketAgent
from app.agents.soil_agent import SoilAgent
from app.agents.fertilizer_agent import FertilizerAgent
from app.agents.government_agent import GovernmentAgent
from app.agents.vision_agent import VisionAgent
from app.agents.finance_agent import FinanceAgent

from app.config import settings
from app.knowledge.advanced_rag import advanced_rag
from app.knowledge.query_understanding import query_understanding
from app.memory.farm_memory import farm_memory_store
from app.llm.generator import response_synthesizer


class PlannerAgent:
    """Central Planner Agent orchestrating specialized sub-agents and advanced multi-source RAG."""

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

    def plan_and_execute(
        self,
        query: str,
        farm_id: str = "FARM_101",
        image_filename: str | None = None,
        language: str = "mr",
        enable_web: bool | None = None,
    ) -> dict:
        farm = farm_memory_store.get_farm(farm_id) or farm_memory_store.get_farm("FARM_101")

        farm_crop = farm["current_crop"].get("crop_name", "Pomegranate")
        district = farm["location"].get("district", "Pune")
        soil_profile = farm.get("soil_profile", {})

        qplan = query_understanding.understand(query, default_crop=farm_crop)
        # Prefer crop mentioned in the farmer query over farm memory default
        crop_name = qplan.crops[0] if qplan.crops else farm_crop

        agent_context = {
            "query": query,
            "crop": crop_name,
            "location": district,
            "district": district,
            "acreage": farm.get("land_area_acres", 2.0),
            "soil_card_text": (
                f"pH: {soil_profile.get('pH', 7.2)}, "
                f"Nitrogen: {soil_profile.get('nitrogen_kg_ha', 180)} kg/ha, "
                f"Phosphorus: {soil_profile.get('phosphorus_kg_ha', 22)} kg/ha, "
                f"Potassium: {soil_profile.get('potassium_kg_ha', 280)} kg/ha"
            ),
            "image_filename": image_filename or "leaf_sample.jpg",
        }

        use_web = settings.ENABLE_WEB_RAG if enable_web is None else enable_web

        # Step 1: Advanced multi-source RAG (local hybrid + GraphRAG + tools + web)
        rag = advanced_rag.retrieve(
            query,
            farm_id=farm["farm_id"],
            crop=crop_name,
            location=district,
            top_k=settings.RAG_TOP_K,
            enable_web=use_web,
            enable_tools=settings.ENABLE_TOOL_RAG,
        )
        agent_context["rag_context"] = rag.get("context_text", "")
        agent_context["rag_citations"] = rag.get("citations", [])

        # Step 2: Intent-based agent selection (query understanding + keywords)
        active_agents = []
        intents = set(qplan.intents)
        query_lower = query.lower()

        if "weather" in intents or any(k in query_lower for k in ("weather", "rain", "पाऊस", "हवामान")):
            active_agents.append(self.weather_agent)

        if (
            "disease" in intents
            or image_filename
            or any(k in query_lower for k in ("disease", "pest", "spot", "रोग", "कीड"))
        ):
            active_agents.extend([self.disease_agent, self.vision_agent, self.weather_agent])

        if "market" in intents or any(k in query_lower for k in ("market", "price", "mandi", "भाव", "बाजार")):
            active_agents.append(self.market_agent)

        if "fertilizer" in intents or any(k in query_lower for k in ("fertilizer", "soil", "npk", "खत", "माती")):
            active_agents.extend([self.soil_agent, self.fertilizer_agent])

        if "scheme" in intents or any(k in query_lower for k in ("scheme", "subsidy", "yojana", "योजना", "अनुदान")):
            active_agents.append(self.government_agent)

        if "irrigation" in intents:
            active_agents.extend([self.crop_agent, self.weather_agent])

        if "seed" in intents:
            active_agents.append(self.crop_agent)

        if "finance" in intents:
            active_agents.append(self.finance_agent)

        if not active_agents:
            active_agents = [
                self.weather_agent,
                self.crop_agent,
                self.disease_agent,
                self.market_agent,
                self.soil_agent,
                self.fertilizer_agent,
                self.government_agent,
                self.finance_agent,
            ]

        active_agents = list({ag.name: ag for ag in active_agents}.values())

        # Step 3: Execute active agents
        agent_results = {}
        for ag in active_agents:
            res = ag.execute(query, agent_context)
            key = ag.name.lower().replace(" agent", "")
            agent_results[key] = (
                res.get("data")
                or res.get("diagnosis")
                or res.get("market_summary")
                or res.get("soil_analysis")
                or res.get("fertilizer_plan")
                or res.get("applicable_schemes")
                or res.get("financial_summary")
                or res
            )

        # Inject compact RAG evidence for the synthesizer
        agent_results["advanced_rag"] = {
            "mode": rag.get("retrieval_mode"),
            "query_plan": rag.get("query_plan"),
            "tools_used": rag.get("tools_used"),
            "top_documents": [
                {
                    "title": d.get("title"),
                    "origin": d.get("origin"),
                    "source": d.get("source"),
                    "score": d.get("fusion_score"),
                    "category": d.get("category"),
                }
                for d in (rag.get("fused_documents") or [])[:6]
            ],
            "citations": rag.get("citations", [])[:8],
            "web_result_count": len(rag.get("web_results") or []),
            "local_hit_count": rag.get("local_hit_count"),
            "knowledge_stats": rag.get("knowledge_stats"),
        }

        plan_summary = (
            f"Processed farmer query '{query}' for {crop_name} in {district}. "
            f"Coordinated {len(active_agents)} specialized sub-agents with Advanced Multi-Source RAG "
            f"(local hybrid + GraphRAG + {len(rag.get('tools_used') or [])} tools"
            f"{' + web' if use_web else ''})."
        )
        # Sprint 16: Mini synthesizer when USE_MINI_LLM=1; agents stay tool specialists.
        final_answer = response_synthesizer.synthesize(
            plan_summary,
            agent_results,
            language=language,
            rag_context=rag.get("context_text"),
            citations=rag.get("citations"),
            query=query,
            crop=crop_name,
            location=district,
            enable_web=use_web,
            use_mini=settings.USE_MINI_LLM,
        )
        synth_meta = getattr(response_synthesizer, "last_meta", None) or {
            "synthesizer": "template",
            "use_mini_llm": False,
        }

        farm_memory_store.log_action(
            farm_id=farm["farm_id"],
            action_type="Farmer Query Handled",
            details=(
                f"Query: '{query}'. Crop resolved: {crop_name}. "
                f"Active agents: {[ag.name for ag in active_agents]}. "
                f"Tools: {rag.get('tools_used')}. "
                f"Synthesizer: {synth_meta.get('synthesizer')}"
            ),
        )

        return {
            "query": query,
            "farm_id": farm["farm_id"],
            "farmer_name": farm["farmer_name"],
            "crop": crop_name,
            "language": language,
            "active_agent_names": [ag.name for ag in active_agents],
            "knowledge_layer": {
                "retrieval_mode": rag.get("retrieval_mode"),
                "query_plan": rag.get("query_plan"),
                "hybrid_rag_hits": rag.get("local_hit_count"),
                "fused_document_count": len(rag.get("fused_documents") or []),
                "tools_used": rag.get("tools_used"),
                "web_result_count": len(rag.get("web_results") or []),
                "graph_rag_ecosystem": (rag.get("graph") or {}).get("crop_ecosystems", []),
                "citations": rag.get("citations", [])[:10],
                "knowledge_stats": rag.get("knowledge_stats"),
            },
            "agent_outputs": agent_results,
            "synthesized_answer": final_answer,
            "synthesizer": synth_meta.get("synthesizer"),
            "use_mini_llm": bool(settings.USE_MINI_LLM),
            "mini_meta": synth_meta if settings.USE_MINI_LLM else None,
        }


planner_agent = PlannerAgent()
