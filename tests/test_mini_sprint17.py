"""Sprint 17 — Mini v1.0 release gate (FP-10)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mini import __feature_phase__, __sprint__, __version__
from mini.orchestrator.dag import PIPELINES
from mini.release.checklist import build_checklist
from mini.release.load_smoke import run_load_smoke
from mini.release.rc_gate import run_release
from mini.release.scale_roadmap import build_scale_report
from mini.release.version_matrix import build_version_matrix_report
from mini.workers.base import get_worker, list_workers
from app.main import app

client = TestClient(app)


def test_sprint17_markers():
    assert __sprint__ in {"S17", "S18", "S20"}
    assert __feature_phase__ in {"FP-10", "v2-15M"}
    assert __version__ in {"1.0.0", "2.0.0-dev"}


def test_release_worker_ready():
    assert {w["worker_id"]: w["status"] for w in list_workers()}["W-RELEASE"] == "ready"


def test_checklist_must_haves_no_hard_fail():
    cl = build_checklist()
    assert cl["summary"]["must_total"] >= 10
    assert cl["ok"] is True
    assert not cl["summary"]["blocking_failures"]
    assert len(cl["deferred"]) >= 4


def test_version_matrix_and_scale():
    mx = build_version_matrix_report()
    tags = {v["tag"] for v in mx["versions"]}
    assert "v1.0-mini" in tags
    assert "v0.4-agri-qa" in tags
    sc = build_scale_report()
    assert sc["decision"] == "proceed_with_family_scale"
    assert any(x["target_params"] == "10M" for x in sc["ladder"])


def test_load_smoke_short():
    smoke = run_load_smoke(rounds=1, max_new_tokens=8, enable_agents=False)
    assert smoke["n_calls"] >= 3
    assert smoke["latency_ms"]["p95"] is not None
    assert smoke["ok"] is True


def test_run_release_gate():
    report = run_release(
        dry_run=False,
        run_eval=True,
        run_smoke=True,
        eval_version="v0.4",
        smoke_rounds=1,
        seed=42,
    )
    assert report.get("sprint") == "S17"
    assert report.get("release") == "v1.0"
    assert report.get("checklist", {}).get("ok") is True
    assert report.get("gates", {}).get("checklist_ok") is True
    assert report.get("runbook")
    assert report.get("artifacts")
    # full gate should pass on this repo
    assert report.get("ok") is True


def test_worker_release():
    res = get_worker("W-RELEASE").run(
        dry_run=False,
        run_eval=True,
        run_smoke=True,
        smoke_rounds=1,
        eval_version="v0.4",
    )
    assert res.ok is True
    assert res.metrics.get("release") == "v1.0"


def test_sprint17_pipeline():
    assert "sprint17" in PIPELINES
    assert PIPELINES["sprint17"] == ["W-EVAL", "W-RELEASE"]
    assert PIPELINES["release"] == ["W-RELEASE"]


def test_api_mini_release():
    r = client.post(
        "/api/mini/release?execute=true&run_eval=true&run_smoke=true&smoke_rounds=1&eval_version=v0.4"
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["metrics"]["sprint"] == "S17"
