"""Sprint 14 — quantize + deploy packaging (W-QUANT, W-DEPLOY)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mini import __feature_phase__, __sprint__
from mini.models.deploy import SEMVER_POLICY, run_deploy
from mini.models.quantize import run_quantize
from mini.orchestrator.dag import PIPELINES
from mini.paths import MODELS_DIR
from mini.workers.base import get_worker, list_workers
from app.main import app

client = TestClient(app)


def test_sprint14_markers():
    assert __sprint__ in {"S14", "S15", "S16", "S17", "S18"}
    assert __feature_phase__ in {"E5-quant", "FP-8", "FP-9", "FP-10", "v2-15M"}


def test_quant_deploy_workers_ready():
    st = {w["worker_id"]: w["status"] for w in list_workers()}
    assert st["W-QUANT"] == "ready"
    assert st["W-DEPLOY"] == "ready"


def test_semver_policy():
    assert "format" in SEMVER_POLICY
    assert any("v0.5" in e for e in SEMVER_POLICY.get("examples") or [])


def test_run_quantize_int8_smaller():
    report = run_quantize(
        dry_run=False,
        version="v0.4",
        include_int4=True,
        seed=42,
        latency_runs=3,
    )
    assert report.get("ok") is True
    assert report.get("sprint") == "S14"
    cmp_ = report["comparison"]
    assert cmp_["fp32"]["weight_bytes"] > 0
    assert cmp_["int8"]["weight_bytes"] > 0
    assert cmp_["int8"]["weight_bytes"] < cmp_["fp32"]["weight_bytes"]
    assert cmp_["int8"]["within_budget"] is True
    assert cmp_["int8"]["latency"]["p95_ms"] is not None
    assert cmp_["fp32"]["latency"]["p95_ms"] is not None
    assert "int4" in cmp_
    assert MODELS_DIR.joinpath("QUANT_LATEST.json").exists()
    assert MODELS_DIR.joinpath("v0.5-quant", "int8").exists()


def test_worker_quantize():
    res = get_worker("W-QUANT").run(
        dry_run=False,
        version="v0.4",
        include_int4=True,
        seed=42,
        latency_runs=2,
    )
    assert res.ok is True
    assert res.metrics.get("sprint") == "S14"


def test_run_deploy_package():
    # ensure quant artifacts exist
    run_quantize(dry_run=False, version="v0.4", include_int4=True, seed=42, latency_runs=2)
    report = run_deploy(
        dry_run=False,
        source_version="v0.4",
        tag="v0.5-quant",
        force=True,
        include_quant=True,
        reasoning_lite=True,
    )
    assert report.get("ok") is True
    assert report.get("tag") == "v0.5-quant"
    assert "v0.5-reasoning-lite" in (report.get("tags_written") or [])
    serve = MODELS_DIR / "serve" / "v0.5-quant"
    assert (serve / "MODEL_CARD.json").exists()
    assert (serve / "LICENSE.txt").exists()
    assert (serve / "manifest.json").exists()
    assert MODELS_DIR.joinpath("VERSION_REGISTRY.json").exists()
    assert MODELS_DIR.joinpath("DEPLOY_LATEST.json").exists()


def test_worker_deploy():
    res = get_worker("W-DEPLOY").run(
        dry_run=False,
        source_version="v0.4",
        tag="v0.5-quant",
        force=True,
        include_quant=True,
        reasoning_lite=True,
    )
    assert res.ok is True
    assert res.metrics.get("sprint") == "S14"


def test_sprint14_pipeline():
    assert "sprint14" in PIPELINES
    assert PIPELINES["sprint14"] == ["W-QUANT", "W-DEPLOY"]
    assert PIPELINES["quant"] == ["W-QUANT"]
    assert PIPELINES["deploy"] == ["W-DEPLOY"]


def test_api_quant_and_deploy():
    r = client.post("/api/lake/quant?execute=true&version=v0.4&latency_runs=2")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["metrics"]["sprint"] == "S14"

    r2 = client.post("/api/lake/deploy?execute=true&version=v0.4&tag=v0.5-quant&force=true")
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["ok"] is True
    assert body2["metrics"]["tag"] == "v0.5-quant"
