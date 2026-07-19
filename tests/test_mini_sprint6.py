"""Sprint 6 — W-QASYNTH expert QA synthesis acceptance tests."""

from __future__ import annotations

import json
from collections import defaultdict

from fastapi.testclient import TestClient

from mini import __feature_phase__, __sprint__
from mini.contracts import DataSplit
from mini.lake.qa_synth import export_synth_dataset, synthesize_qa_records
from mini.paths import LAKE_ROOT, ensure_lake_layout
from mini.workers.base import get_worker, list_workers
from mini.orchestrator.dag import PIPELINES, run_pipeline
from app.main import app

client = TestClient(app)


def _prep_processed():
    ensure_lake_layout()
    get_worker("W-INGEST").run(dry_run=False, include_http=False)
    get_worker("W-QUALITY").run(dry_run=False)


def test_sprint6_markers():
    assert __sprint__ in {"S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18"}
    assert __feature_phase__ in {"FP-3", "FP-4", "FP-5", "FP-5b", "FP-6", "FP-7", "E5-eval", "E5-quant", "FP-8", "FP-9", "FP-10", "v2-15M"}


def test_qasynth_worker_ready():
    st = {w["worker_id"]: w["status"] for w in list_workers()}
    assert st["W-QASYNTH"] == "ready"


def test_synthesize_volume_and_splits():
    _prep_processed()
    records = synthesize_qa_records(target_min_total=12000)
    train = [r for r in records if r.split == DataSplit.TRAIN]
    val = [r for r in records if r.split == DataSplit.VAL]
    test = [r for r in records if r.split == DataSplit.TEST]
    assert len(train) >= 10000
    assert len(val) >= 1000
    assert len(test) >= 1
    # categories include core packs
    cats = {r.category.value for r in records}
    for c in ("crop", "disease", "fertilizer", "scheme"):
        assert c in cats
    langs = {r.language.value for r in records}
    assert "en" in langs
    assert "mr" in langs or "hi" in langs


def test_no_train_test_leakage_by_fact_key():
    _prep_processed()
    records = synthesize_qa_records(target_min_total=5000)
    # All records sharing fact_key must share split
    by_fact: dict[str, set[str]] = defaultdict(set)
    for r in records:
        fk = (r.metadata or {}).get("fact_key")
        if fk:
            by_fact[fk].add(r.split.value)
    multi = {k: v for k, v in by_fact.items() if len(v) > 1}
    assert not multi, f"Leakage: fact_keys with multiple splits: {list(multi)[:5]}"


def test_export_synth_and_review_queue():
    _prep_processed()
    records = synthesize_qa_records(target_min_total=12000)
    report = export_synth_dataset(records, dry_run=False)
    assert report["ok"] is True
    assert report["targets_met"]["train"] is True
    assert report["targets_met"]["val"] is True
    assert (LAKE_ROOT / "QASYNTH_LATEST.json").exists()
    # review queue path among artifacts
    assert any("human_review_queue.csv" in a for a in report.get("artifacts") or [])


def test_qasynth_worker_execute():
    _prep_processed()
    res = get_worker("W-QASYNTH").run(dry_run=False, target_min_total=12000)
    assert res.ok is True
    assert res.metrics["counts"]["train"] >= 10000
    assert res.metrics["counts"]["val"] >= 1000


def test_sprint6_pipeline():
    assert "sprint6" in PIPELINES
    assert "qasynth" in PIPELINES
    result = run_pipeline("qasynth", dry_run=True)
    assert result.ok is True


def test_api_qasynth():
    r = client.post("/api/lake/qasynth?execute=true&target=12000")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    r2 = client.get("/api/lake/qasynth")
    assert r2.status_code == 200
    data = r2.json()
    assert data.get("counts", {}).get("train", 0) >= 10000
