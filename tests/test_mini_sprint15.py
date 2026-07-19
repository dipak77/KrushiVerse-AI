"""Sprint 15 — Mini inference chain (W-INFER / W-RAG / W-AGENT, FP-8)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mini import __feature_phase__, __sprint__
from mini.inference.context import build_context_pack, normalize_sources
from mini.inference.pipeline import run_infer
from mini.inference.rag_wrap import retrieve_for_mini
from mini.inference.validate import banned_hits, validate_answer
from mini.orchestrator.dag import PIPELINES
from mini.paths import INFERENCE_DIR
from mini.workers.base import get_worker, list_workers
from app.main import app

client = TestClient(app)


def test_sprint15_markers():
    assert __sprint__ in {"S15", "S16", "S17"}
    assert __feature_phase__ in {"FP-8", "FP-9", "FP-10"}


def test_infer_rag_agent_ready():
    st = {w["worker_id"]: w["status"] for w in list_workers()}
    assert st["W-INFER"] == "ready"
    assert st["W-RAG"] == "ready"
    assert st["W-AGENT"] == "ready"


def test_context_and_validator():
    srcs = normalize_sources(
        [
            {
                "id": "a1",
                "title": "Cotton IPM",
                "text": "Scout pink bollworm with pheromone traps and follow ETL.",
                "origin": "local",
            }
        ]
    )
    pack = build_context_pack(query="cotton bollworm?", sources=srcs)
    assert pack["has_sources"]
    assert "[1]" in pack["context_text"]
    assert banned_hits("Yes, double the pesticide dose now")
    v = validate_answer(
        answer="Use traps [1] and ETL for bollworm.",
        context=pack["context_text"],
        citations=pack["citations"],
        mode="grounded",
        min_grounding=0.01,
    )
    assert v["ok"] is True


def test_grounded_requires_sources():
    v = validate_answer(answer="Anything", context="", citations=[], mode="grounded")
    assert v["ok"] is False
    assert "no_sources" in v["reasons"]


def test_retrieve_cotton_local():
    rag = retrieve_for_mini(
        "How do I manage pink bollworm in cotton with IPM?",
        crop="Cotton",
        enable_web=False,
        enable_tools=False,
        use_platform_rag=False,
    )
    assert rag["has_sources"] is True
    assert rag["n_sources"] >= 1


def test_run_infer_cotton_demo():
    report = run_infer(
        query="How do I manage pink bollworm in cotton with IPM in Maharashtra?",
        mode="grounded",
        crop="Cotton",
        enable_web=False,
        enable_tools=False,
        enable_agents=False,
        use_platform_rag=False,
        max_new_tokens=24,
        seed=42,
    )
    assert report.get("sprint") == "S15"
    assert report.get("n_sources", 0) >= 1
    assert report.get("citations")
    assert report.get("answer")
    # grounded: must include sources section or markers
    ans = report["answer"]
    assert "Sources" in ans or "[1]" in ans
    assert report.get("ok") is True
    assert INFERENCE_DIR.joinpath("INFER_LATEST.json").exists()


def test_worker_infer_and_rag():
    rag = get_worker("W-RAG").run(
        dry_run=False,
        query="pink bollworm cotton IPM Maharashtra",
        crop="Cotton",
        enable_web=False,
        enable_tools=False,
        use_platform_rag=False,
    )
    assert rag.ok is True
    assert (rag.metrics or {}).get("n_sources", 0) >= 1

    inf = get_worker("W-INFER").run(
        dry_run=False,
        query="How do I manage pink bollworm in cotton?",
        mode="grounded",
        crop="Cotton",
        enable_web=False,
        enable_tools=False,
        enable_agents=False,
        use_platform_rag=False,
        max_new_tokens=20,
    )
    assert inf.ok is True
    assert inf.metrics.get("sprint") == "S15"
    assert inf.metrics.get("n_sources", 0) >= 1


def test_sprint15_pipeline():
    assert "sprint15" in PIPELINES
    assert PIPELINES["sprint15"] == ["W-RAG", "W-AGENT", "W-INFER"]
    assert PIPELINES["infer"] == ["W-INFER"]


def test_api_infer_s15():
    r = client.post(
        "/api/lake/infer",
        params={
            "execute": True,
            "query": "How do I manage pink bollworm in cotton with IPM?",
            "mode": "grounded",
            "crop": "Cotton",
            "enable_web": False,
            "enable_agents": False,
            "max_new_tokens": 16,
            "seed": 42,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["metrics"]["sprint"] == "S15"
    assert body["metrics"]["n_sources"] >= 1
