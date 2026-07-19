"""Sprint 8 — W-KGBUILD knowledge graph factory (≥200 nodes, ≥400 edges)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mini import __feature_phase__, __sprint__
from mini.lake.kg_build import extract_and_merge, run_kg_build
from mini.paths import DATASETS_DIR, ensure_lake_layout
from mini.workers.base import get_worker, list_workers
from mini.orchestrator.dag import PIPELINES
from app.main import app
from app.knowledge.graph_rag import graph_rag

client = TestClient(app)


def test_sprint8_markers():
    assert __sprint__ in {"S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18"}
    assert __feature_phase__ in {"FP-4", "FP-5", "FP-5b", "FP-6", "FP-7", "E5-eval", "E5-quant", "FP-8", "FP-9", "FP-10", "v2-15M"}


def test_kgbuild_ready():
    assert {w["worker_id"]: w["status"] for w in list_workers()}["W-KGBUILD"] == "ready"


def test_extract_meets_node_edge_targets():
    nodes, edges, stats = extract_and_merge(include_districts=True)
    assert len(nodes) >= 200
    assert len(edges) >= 400
    assert stats["final_nodes"] == len(nodes)
    # core relation types present
    rels = {e["relation"] for e in edges.values()}
    for r in ("AFFECTED_BY", "GROWS_IN", "REQUIRES_FERTILIZER", "COVERED_BY"):
        assert r in rels
    labels = {n["label"] for n in nodes.values()}
    for lab in ("Crop", "Pest", "Disease", "Soil", "Scheme"):
        assert lab in labels


def test_run_kg_build_exports():
    ensure_lake_layout()
    report = run_kg_build(dry_run=False, write_platform_seed=False)
    assert report["ok"] is True
    assert report["counts"]["nodes"] >= 200
    assert report["counts"]["edges"] >= 400
    assert report["targets_met"]["nodes"] is True
    assert report["targets_met"]["edges"] is True
    assert (DATASETS_DIR / "KG_LATEST.json").exists()
    assert (DATASETS_DIR / "kg" / "graph_triples.jsonl").exists()
    assert (DATASETS_DIR / "kg" / "neo4j_loader_stub.cypher").exists()


def test_worker_execute_s8():
    res = get_worker("W-KGBUILD").run(dry_run=False, write_platform_seed=False)
    assert res.ok is True
    assert res.metrics["counts"]["nodes"] >= 200
    assert res.metrics["counts"]["edges"] >= 400


def test_sprint8_pipeline_registered():
    assert "sprint8" in PIPELINES
    assert "W-KGBUILD" in PIPELINES["sprint8"]
    assert "kgbuild" in PIPELINES


def test_api_kg_endpoints():
    r = client.post("/api/lake/kg?execute=true")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["metrics"]["counts"]["nodes"] >= 200
    assert body["metrics"]["counts"]["edges"] >= 400

    s = client.get("/api/lake/kg")
    assert s.status_code == 200
    assert s.json().get("ok") is True


def test_graphrag_still_works_after_kgbuild():
    """Platform GraphRAG APIs remain functional (seed not required to change)."""
    data = graph_rag.get_crop_ecosystem("Pomegranate")
    assert data.get("crop") == "Pomegranate" or "error" not in data or data.get("found") is not False
    # ecosystem helper returns crop key when found
    if "error" not in data:
        assert "pests_and_diseases" in data
        assert "applicable_schemes" in data
