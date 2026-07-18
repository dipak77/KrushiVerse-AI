"""Sprint 2 — Source registry + W-INGEST lake seed acceptance tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from mini import __feature_phase__, __sprint__
from mini.lake.ingest import IngestEngine, lake_tree_summary, sha256_file
from mini.lake.registry import load_source_registry
from mini.paths import LAKE_RAW, LAKE_ROOT, REPO_ROOT, ensure_lake_layout
from mini.workers.base import get_worker, list_workers
from mini.orchestrator.dag import PIPELINES, run_pipeline
from app.main import app

client = TestClient(app)


def test_sprint2_markers():
    # S2 deliverables remain; markers advance with later sprints
    assert __sprint__ in {"S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"}
    assert __feature_phase__ in {"FP-1", "FP-2", "FP-3", "FP-4", "FP-5", "FP-5b", "FP-6", "FP-7", "E5-eval", "E5-quant", "FP-8", "FP-9"}


def test_source_registry_loads():
    reg = load_source_registry()
    assert reg.version == "1.0.0"
    assert len(reg.sources) >= 10
    local = reg.enabled_sources(types={"local_file"})
    assert len(local) >= 8
    # all local paths exist
    for s in local:
        p = s.resolve_path()
        assert p is not None and p.exists(), f"missing source file {s.id}: {s.path}"


def test_ingest_worker_registered_ready():
    workers = {w["worker_id"]: w for w in list_workers()}
    assert workers["W-INGEST"]["status"] == "ready"


def test_ingest_dry_run():
    worker = get_worker("W-INGEST")
    res = worker.run(dry_run=True, include_http=False)
    assert res.ok is True
    # May copy or skip depending on prior lake state
    assert res.metrics["files_copied"] + res.metrics["files_skipped"] >= 8
    assert res.metrics["sources_considered"] >= 8


def test_ingest_execute_and_idempotent():
    ensure_lake_layout()
    worker = get_worker("W-INGEST")

    first = worker.run(dry_run=False, include_http=False)
    assert first.ok is True
    assert first.metrics["files_copied"] + first.metrics["files_skipped"] >= 8
    assert (LAKE_ROOT / "INGEST_LATEST.json").exists()
    manifests = list((LAKE_ROOT / "manifests").glob("ingest_*.json"))
    assert len(manifests) >= 1

    # sample file exists under raw
    crop_files = list((LAKE_RAW / "crop").rglob("*.json"))
    crop_files = [p for p in crop_files if not p.name.endswith(".meta.json")]
    assert len(crop_files) >= 1
    digest1 = sha256_file(crop_files[0])

    second = worker.run(dry_run=False, include_http=False)
    assert second.ok is True
    # second run must be fully idempotent (no re-copy)
    assert second.metrics["files_skipped"] >= 8
    assert second.metrics["files_copied"] == 0
    digest2 = sha256_file(crop_files[0])
    assert digest1 == digest2


def test_manifest_has_hashes():
    ensure_lake_layout()
    get_worker("W-INGEST").run(dry_run=False, include_http=False)
    latest = json.loads((LAKE_ROOT / "INGEST_LATEST.json").read_text(encoding="utf-8"))
    assert "run_id" in latest
    assert latest["files_copied"] + latest["files_skipped"] >= 8
    # find a copied or skipped result with sha
    with_hash = [r for r in latest["results"] if r.get("sha256")]
    assert len(with_hash) >= 1

    man_dir = LAKE_ROOT / "manifests"
    mans = sorted(man_dir.glob("ingest_*.json"))
    assert mans
    man = json.loads(mans[-1].read_text(encoding="utf-8"))
    assert man["worker_id"] == "W-INGEST"
    assert "entries" in man


def test_multi_domain_copy():
    """crops_and_diseases should land in crop, disease, pest domains."""
    get_worker("W-INGEST").run(dry_run=False, include_http=False)
    name = "crops_and_diseases.json"
    assert any(p.name == name for p in (LAKE_RAW / "crop").rglob(name))
    assert any(p.name == name for p in (LAKE_RAW / "disease").rglob(name))
    assert any(p.name == name for p in (LAKE_RAW / "pest").rglob(name))


def test_lake_tree_summary():
    get_worker("W-INGEST").run(dry_run=False, include_http=False)
    tree = lake_tree_summary()
    assert tree["file_count"] >= 8
    assert "crop" in tree["domains_with_data"] or "market" in tree["domains_with_data"]


def test_sprint2_pipeline_dry():
    assert "sprint2" in PIPELINES
    assert "ingest" in PIPELINES
    result = run_pipeline("sprint2", dry_run=True)
    assert result.ok is True


def test_cli_sources_and_ingest():
    proc = subprocess.run(
        [sys.executable, "-m", "mini.orchestrator", "sources"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["summary"]["total_sources"] >= 10

    proc2 = subprocess.run(
        [sys.executable, "-m", "mini.orchestrator", "ingest", "--skip-http"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc2.returncode == 0, proc2.stderr + proc2.stdout


def test_api_lake_endpoints():
    r = client.get("/api/lake/status")
    assert r.status_code == 200
    body = r.json()
    assert "registry" in body
    assert body["registry"]["total_sources"] >= 10

    r2 = client.post("/api/lake/ingest?execute=false&skip_http=true")
    assert r2.status_code == 200
    assert r2.json()["ok"] is True
