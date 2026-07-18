from app.knowledge.dataset_loader import kb_loader
from app.knowledge.query_understanding import query_understanding
from app.knowledge.advanced_rag import advanced_rag
from app.knowledge.tools.registry import tool_registry
from app.agents.planner import planner_agent
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_expanded_knowledge_volume():
    stats = kb_loader.knowledge_stats()
    assert stats["total_documents"] >= 100
    assert stats["crops"] >= 15
    assert stats["diseases_pests"] >= 15
    assert stats["schemes"] >= 10
    assert stats["advisories"] >= 20
    assert "Advisory" in stats["by_category"]
    assert "Market" in stats["by_category"]


def test_query_understanding_extracts_crop():
    plan = query_understanding.understand("What fertilizers for Cotton in black soil?")
    assert "Cotton" in plan.crops
    assert "fertilizer" in plan.intents
    assert len(plan.expanded_queries) >= 2


def test_query_understanding_marathi():
    plan = query_understanding.understand("कापूस रोग नियंत्रण कसे करावे?")
    assert "Cotton" in plan.crops
    assert plan.language_hint == "mr"


def test_advanced_rag_local_only():
    result = advanced_rag.retrieve(
        "Cotton pink bollworm organic control",
        crop="Cotton",
        location="Pune",
        top_k=5,
        enable_web=False,
        enable_tools=True,
    )
    assert result["retrieval_mode"].startswith("advanced_multi_source_rag_v10")
    assert result["local_hit_count"] > 0
    assert len(result["fused_documents"]) > 0
    assert result["context_text"]
    assert result["citations"]
    assert "crop_knowledge" in result["tools_used"] or "open_source_catalog" in result["tools_used"]


def test_advanced_rag_with_web_flag():
    result = advanced_rag.retrieve(
        "latest soybean mandi price Maharashtra",
        location="Latur",
        top_k=5,
        enable_web=True,
        enable_tools=True,
        force_web=True,
    )
    assert "query_plan" in result
    assert len(result["fused_documents"]) > 0
    # web_results may be live hits or offline catalog stubs
    assert isinstance(result["web_results"], list)


def test_tool_registry_lists_and_runs():
    tools = tool_registry.list_tools()
    assert len(tools) >= 5
    res = tool_registry.run("crop_knowledge", {"crop": "Onion"})
    assert res["ok"] is True
    assert len(res["documents"]) > 0


def test_planner_uses_query_crop_not_only_farm_memory():
    res = planner_agent.plan_and_execute(
        query="What fertilizers for Cotton?",
        farm_id="FARM_101",
        language="en",
        enable_web=False,
    )
    assert res["crop"] == "Cotton"
    assert "knowledge_layer" in res
    assert res["knowledge_layer"].get("retrieval_mode", "").startswith("advanced_multi_source_rag_v10")
    assert "synthesized_answer" in res


def test_api_knowledge_stats():
    r = client.get("/api/knowledge/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["stats"]["total_documents"] >= 100
    assert len(data["tools"]) >= 5


def test_api_advanced_rag():
    r = client.post(
        "/api/rag/advanced",
        json={
            "query": "Onion purple blotch treatment",
            "enable_web": False,
            "enable_tools": True,
            "top_k": 5,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["fused_documents"]) > 0
    assert data["local_hit_count"] > 0


def test_api_query_includes_advanced_knowledge_layer():
    r = client.post(
        "/api/query",
        json={"query": "What fertilizers for Cotton?", "language": "en", "enable_web": False},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["crop"] == "Cotton"
    assert data["knowledge_layer"]["retrieval_mode"].startswith("advanced_multi_source_rag_v10")
