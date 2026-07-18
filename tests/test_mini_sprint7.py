"""Sprint 7 — expanded domain QA packs (≥50k train)."""

from __future__ import annotations

from collections import Counter

from fastapi.testclient import TestClient

from mini import __feature_phase__, __sprint__
from mini.contracts import DataSplit
from mini.lake.qa_synth import export_synth_dataset, synthesize_qa_records
from mini.paths import LAKE_ROOT, ensure_lake_layout
from mini.workers.base import get_worker, list_workers
from mini.orchestrator.dag import PIPELINES
from app.main import app

client = TestClient(app)


def _prep():
    ensure_lake_layout()
    get_worker("W-INGEST").run(dry_run=False, include_http=False)
    get_worker("W-QUALITY").run(dry_run=False)


def test_sprint7_markers():
    assert __sprint__ in {"S7", "S8", "S9"}
    assert __feature_phase__ in {"FP-3", "FP-4", "FP-5"}


def test_qasynth_still_ready():
    assert {w["worker_id"]: w["status"] for w in list_workers()}["W-QASYNTH"] == "ready"


def test_synth_50k_train_and_coverage():
    _prep()
    records = synthesize_qa_records(target_min_total=62500, sprint7_expand=True)
    train = [r for r in records if r.split == DataSplit.TRAIN]
    val = [r for r in records if r.split == DataSplit.VAL]
    assert len(train) >= 50000
    assert len(val) >= 1000

    cats = {r.category.value for r in records}
    assert len(cats) >= 8
    # core expanded domains present
    for c in ("soil", "weather", "pest", "irrigation", "market", "finance", "crop", "disease"):
        assert c in cats

    lang = Counter(r.language.value for r in records)
    non_en = sum(v for k, v in lang.items() if k != "en")
    assert 100.0 * non_en / len(records) >= 20.0

    # hard negatives / safety present
    packs = {(r.metadata or {}).get("pack") for r in records}
    assert "hard_negative" in packs or "safety" in packs
    assert "pest" in packs or "nutrient" in packs


def test_export_meets_s7_targets():
    _prep()
    records = synthesize_qa_records(target_min_total=62500, sprint7_expand=True)
    report = export_synth_dataset(records, dry_run=False)
    assert report["ok"] is True
    assert report["targets_met"]["train"] is True
    assert report["targets_met"]["val"] is True
    assert report["targets_met"]["categories"] is True
    assert report["targets_met"]["non_english_pct"] is True
    assert (LAKE_ROOT / "QASYNTH_LATEST.json").exists()


def test_worker_execute_s7():
    _prep()
    res = get_worker("W-QASYNTH").run(dry_run=False, target_min_total=62500)
    assert res.ok is True
    assert res.metrics["counts"]["train"] >= 50000


def test_sprint7_pipeline_registered():
    assert "sprint7" in PIPELINES
    assert PIPELINES["sprint7"][-1] == "W-ANALYZE" or "W-QASYNTH" in PIPELINES["sprint7"]


def test_api_qasynth_s7_default():
    r = client.post("/api/lake/qasynth?execute=true")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["metrics"]["counts"]["train"] >= 50000
