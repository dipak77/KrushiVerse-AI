"""Sprint 4 — Normalize, language detect, Schema v1 standard export."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from mini import __feature_phase__, __sprint__
from mini.contracts import DataSplit, LanguageCode, StandardRecord
from mini.lake.langdetect import detect_language, detect_language_pair
from mini.lake.standardize import (
    coverage_stats,
    export_standard_dataset,
    extract_standard_records_from_processed,
    run_standardize_pipeline,
)
from mini.paths import (
    DATASETS_DIR,
    LAKE_TEST,
    LAKE_TRAINING,
    LAKE_VALIDATION,
    ensure_lake_layout,
)
from mini.workers.base import get_worker, list_workers
from mini.orchestrator.dag import PIPELINES, run_pipeline
from app.main import app

client = TestClient(app)


def _ensure_processed():
    """Ingest + quality so processed lake has content."""
    ensure_lake_layout()
    get_worker("W-INGEST").run(dry_run=False, include_http=False)
    get_worker("W-QUALITY").run(dry_run=False)


def test_sprint4_markers():
    assert __sprint__ in {"S4", "S5", "S6", "S7"}
    assert __feature_phase__.startswith("FP-")


def test_workers_ready():
    st = {w["worker_id"]: w["status"] for w in list_workers()}
    for wid in ("W-NORMALIZE", "W-LANGDETECT", "W-STANDARD", "W-STANDARDIZE"):
        assert st[wid] == "ready", wid


def test_language_detect_en_mr():
    assert detect_language("What fertilizer for cotton?") == LanguageCode.EN
    assert detect_language("कापूस पीक माहिती काय आहे") in {
        LanguageCode.MR,
        LanguageCode.HI,
        LanguageCode.MIXED,
    }
    assert detect_language_pair("Hello cotton", "कापूस खत") == LanguageCode.MIXED


def test_extract_standard_records():
    _ensure_processed()
    records = extract_standard_records_from_processed()
    assert len(records) >= 50
    stats = coverage_stats(records)
    assert stats["language_pct"] >= 90
    assert stats["category_pct"] >= 90
    # splits present
    splits = {r.split for r in records}
    assert DataSplit.TRAIN in splits
    assert any(r.crop for r in records)


def test_export_train_val_test_jsonl_parquet():
    _ensure_processed()
    records = extract_standard_records_from_processed()
    report = export_standard_dataset(records, dry_run=False)
    assert report["ok"] is True
    assert report["counts"]["total"] >= 50
    assert report["counts"]["train"] >= 1
    assert (LAKE_TRAINING / "standard_records.jsonl").exists()
    assert (LAKE_VALIDATION / "standard_records.jsonl").exists()
    assert (LAKE_TEST / "standard_records.jsonl").exists()
    assert (DATASETS_DIR / "LATEST_VERSION.json").exists()

    # JSONL lines are valid StandardRecord dicts
    line = (LAKE_TRAINING / "standard_records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    obj = json.loads(line)
    assert obj["schema_version"] == "1.0"
    assert obj["question"]
    assert obj["answer"]
    assert obj["language"]
    assert obj["category"]
    assert obj["split"] == "train"

    # parquet preferred
    if report.get("parquet"):
        assert (LAKE_TRAINING / "standard_records.parquet").exists()


def test_standardize_pipeline_worker():
    _ensure_processed()
    res = get_worker("W-STANDARDIZE").run(dry_run=False)
    assert res.ok is True
    assert res.metrics["export"]["counts"]["total"] >= 50


def test_langdetect_and_normalize_workers():
    _ensure_processed()
    n = get_worker("W-NORMALIZE").run(dry_run=True)
    assert n.ok and n.metrics["records"] >= 50
    l = get_worker("W-LANGDETECT").run(dry_run=True)
    assert l.ok and l.metrics["total"] >= 50
    assert l.metrics["known_pct"] >= 90


def test_sprint4_pipeline():
    assert "sprint4" in PIPELINES
    assert "standardize" in PIPELINES
    result = run_pipeline("standardize", dry_run=True)
    assert result.ok is True


def test_api_standard_endpoints():
    r = client.post("/api/lake/standard?execute=true")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    r2 = client.get("/api/lake/standard")
    assert r2.status_code == 200
    assert r2.json().get("ok") is True
