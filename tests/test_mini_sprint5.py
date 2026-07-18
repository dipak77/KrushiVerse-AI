"""Sprint 5 — W-ANALYZE coverage intelligence tests."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from mini import __sprint__
from mini.lake.analyze import analyze_records, run_analysis, taxonomy_coverage_gaps
from mini.paths import LAKE_ROOT, ensure_lake_layout
from mini.workers.base import get_worker, list_workers
from mini.orchestrator.dag import PIPELINES, run_pipeline
from app.main import app

client = TestClient(app)


def _prepare_dataset():
    ensure_lake_layout()
    get_worker("W-INGEST").run(dry_run=False, include_http=False)
    get_worker("W-QUALITY").run(dry_run=False)
    get_worker("W-STANDARDIZE").run(dry_run=False)


def test_sprint5_markers():
    assert __sprint__ in {"S5", "S6", "S7", "S8", "S9", "S10"}


def test_analyze_worker_ready():
    st = {w["worker_id"]: w["status"] for w in list_workers()}
    assert st["W-ANALYZE"] == "ready"


def test_analyze_records_metrics():
    sample = [
        {
            "id": "1",
            "category": "crop",
            "crop": "Cotton",
            "language": "en",
            "question": "What about cotton?",
            "answer": "Cotton is a fiber crop grown in black soil.",
            "source": "test",
            "split": "train",
            "schema_version": "1.0",
            "region": {"state": "Maharashtra"},
            "confidence": 0.8,
            "verified": False,
            "subcategory": "crop_guide",
        },
        {
            "id": "2",
            "category": "crop",
            "crop": "Cotton",
            "language": "en",
            "question": "What about cotton?",
            "answer": "Cotton is a fiber crop grown in black soil.",
            "source": "test",
            "split": "train",
            "schema_version": "1.0",
            "region": {"state": "Maharashtra"},
            "confidence": 0.8,
            "verified": False,
        },
        {
            "id": "3",
            "category": "disease",
            "crop": None,
            "language": "mr",
            "question": "कापूस रोग?",
            "answer": "तेल्या रोगासाठी कॉपर फवारणी करा.",
            "source": "test",
            "split": "val",
            "schema_version": "1.0",
            "region": {},
            "confidence": 0.7,
            "verified": False,
        },
    ]
    m = analyze_records(sample)
    assert m["total_records"] == 3
    assert m["duplicates"]["exact_qa_duplicates"] >= 1
    assert "en" in m["balance"]["language"]
    assert m["missingness"]["crop"] >= 1
    assert m["length"]["question"]["histogram"]


def test_taxonomy_gaps_nonempty_on_sparse_data():
    # Only cotton present → many missing crops
    sparse = [
        {
            "category": "crop",
            "crop": "Cotton",
            "language": "en",
            "question": "q",
            "answer": "a long enough answer",
            "subcategory": "crop_guide",
            "region": {},
        }
    ]
    gaps = taxonomy_coverage_gaps(sparse)
    assert gaps["gap_count"] > 0
    assert len(gaps["missing_crops"]) > 0


def test_run_analysis_writes_report():
    _prepare_dataset()
    report = run_analysis(dry_run=False)
    assert report["ok"] is True
    assert report["summary"]["total_records"] >= 50
    assert report["summary"]["gap_count"] >= 1  # stages / machinery etc.
    assert (LAKE_ROOT / "ANALYZE_LATEST.json").exists()
    assert (LAKE_ROOT / "ANALYZE_LATEST.html").exists()
    data = json.loads((LAKE_ROOT / "ANALYZE_LATEST.json").read_text(encoding="utf-8"))
    assert "records" in data and "taxonomy_gaps" in data


def test_analyze_worker():
    _prepare_dataset()
    res = get_worker("W-ANALYZE").run(dry_run=False)
    assert res.ok is True
    assert res.metrics["summary"]["total_records"] >= 50
    assert len(res.artifacts) >= 1


def test_sprint5_pipeline():
    assert "sprint5" in PIPELINES
    assert "analyze" in PIPELINES
    # dry-run analyze only
    result = run_pipeline("analyze", dry_run=True)
    assert result.ok is True


def test_api_analyze():
    r = client.post("/api/lake/analyze?execute=true")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    r2 = client.get("/api/lake/analyze")
    assert r2.status_code == 200
    assert r2.json().get("summary", {}).get("total_records", 0) >= 1 or r2.json().get("ok")
