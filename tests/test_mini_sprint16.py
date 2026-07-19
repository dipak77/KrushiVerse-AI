"""Sprint 16 — platform Mini integration (FP-9)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import settings
from app.llm.generator import response_synthesizer
from app.llm.mini_bridge import agent_outputs_to_notes, run_mini_chat
from app.agents.planner import planner_agent
from mini import __feature_phase__, __sprint__
from mini.orchestrator.dag import PIPELINES
from app.main import app

client = TestClient(app)


def test_sprint16_markers():
    assert __sprint__ in {"S16", "S17", "S18"}
    assert __feature_phase__ in {"FP-9", "FP-10", "v2-15M"}


def test_use_mini_llm_default_off():
    # Acceptance: flag off = classic behavior
    assert settings.USE_MINI_LLM is False


def test_template_synthesizer_when_flag_off():
    text = response_synthesizer.synthesize(
        "Plan for cotton",
        {"disease": {"summary_en": "Scout bollworm with traps"}},
        language="en",
        query="cotton bollworm",
        use_mini=False,
    )
    assert "Greetings" in text or "Advisory" in text or "bollworm" in text.lower()
    assert response_synthesizer.last_meta.get("synthesizer") == "template"
    assert response_synthesizer.last_meta.get("use_mini_llm") is False


def test_agent_notes_helper():
    notes = agent_outputs_to_notes(
        {
            "disease": {"summary_en": "Pink bollworm risk high"},
            "advanced_rag": {"citations": []},
        }
    )
    assert any("bollworm" in n.lower() for n in notes)


def test_run_mini_chat_api_shape():
    chat = run_mini_chat(
        "How do I manage pink bollworm in cotton with IPM?",
        language="en",
        crop="Cotton",
        mode="grounded",
        enable_web=False,
        enable_agents=False,
        use_platform_rag=False,
        max_new_tokens=16,
        seed=42,
    )
    assert "answer" in chat
    assert "synthesized_answer" in chat
    assert chat.get("use_mini_llm") is True
    assert chat.get("n_sources", 0) >= 1
    assert chat.get("citations")


def test_api_mini_status_and_chat():
    s = client.get("/api/mini/status")
    assert s.status_code == 200
    body = s.json()
    assert body["ok"] is True
    assert body["sprint"] in {"S16", "S17", "S18"}
    assert "use_mini_llm" in body

    r = client.post(
        "/api/mini/chat",
        json={
            "query": "How do I manage pink bollworm in cotton with IPM?",
            "language": "en",
            "crop": "Cotton",
            "mode": "grounded",
            "enable_web": False,
            "enable_agents": False,
            "max_new_tokens": 16,
            "seed": 42,
        },
    )
    assert r.status_code == 200
    chat = r.json()
    assert chat.get("answer")
    assert chat.get("n_sources", 0) >= 1


def test_planner_flag_off_classic():
    """Existing planner path still works with USE_MINI_LLM false."""
    assert settings.USE_MINI_LLM is False
    res = planner_agent.plan_and_execute(
        query="What fertilizers for cotton?",
        farm_id="FARM_101",
        language="en",
        enable_web=False,
    )
    assert res["farm_id"] == "FARM_101"
    assert res.get("synthesized_answer")
    assert res.get("use_mini_llm") is False
    assert res.get("synthesizer") in (None, "template") or res.get("synthesizer") == "template"


def test_sprint16_pipeline():
    assert "sprint16" in PIPELINES
    assert "W-INFER" in PIPELINES["sprint16"]
