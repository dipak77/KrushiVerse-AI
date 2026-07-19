"""Sprint 3 — Validate / Clean / Dedup quality pipeline tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from mini import __feature_phase__, __sprint__
from mini.lake.quality import clean_text, jaccard, char_ngrams
from mini.lake.dedup import dedupe_list, dedupe_payload
from mini.lake.validate import validate_json_payload, validate_file
from mini.lake.process import run_clean_stage, run_dedup_stage, run_quality_pipeline
from mini.paths import LAKE_PROCESSED, LAKE_QUARANTINE, LAKE_RAW, LAKE_ROOT, ensure_lake_layout
from mini.workers.base import get_worker, list_workers
from mini.orchestrator.dag import PIPELINES, run_pipeline
from app.main import app

client = TestClient(app)


def test_sprint3_markers():
    assert __sprint__ in {"S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17"}
    assert __feature_phase__.startswith("FP-")


def test_workers_ready():
    statuses = {w["worker_id"]: w["status"] for w in list_workers()}
    for wid in ("W-VALIDATE", "W-CLEAN", "W-DEDUP", "W-QUALITY"):
        assert statuses[wid] == "ready"


def test_clean_text_strips_html_and_boilerplate():
    dirty = "<p>Hello&nbsp;<b>Cotton</b></p><script>alert(1)</script>  \n\n\n  world"
    cleaned = clean_text(dirty)
    assert "<" not in cleaned
    assert "script" not in cleaned.lower() or "alert" not in cleaned
    assert "Hello" in cleaned and "Cotton" in cleaned and "world" in cleaned
    assert "\n\n\n" not in cleaned


def test_validate_rejects_empty_and_accepts_crops():
    assert validate_json_payload({}, "crop")
    assert validate_json_payload({"crops": [{"name_en": "Cotton"}]}, "crop") == []
    assert validate_json_payload({"markets": []}, "market") == []  # empty list ok at payload level
    # empty list root
    errs = validate_json_payload([], "general")
    assert errs


def test_dedupe_exact_and_near():
    items = [
        {"id": 1, "text": "Cotton pink bollworm control with neem oil"},
        {"id": 1, "text": "Cotton pink bollworm control with neem oil"},  # exact
        {"id": 2, "text": "Cotton pink bollworm control with neem oil spray"},  # near
        {"id": 3, "text": "Onion purple blotch fungicide schedule completely different topic"},
    ]
    kept, stats = dedupe_list(items, near_threshold=0.85)
    assert stats["exact_removed"] >= 1
    assert stats["kept"] < len(items)
    assert any("Onion" in str(x) for x in kept)


def test_dedupe_payload_list_fields():
    payload = {
        "crops": [
            {"name_en": "Cotton", "note": "A"},
            {"name_en": "Cotton", "note": "A"},
            {"name_en": "Soybean", "note": "B"},
        ]
    }
    out, stats = dedupe_payload(payload)
    assert stats["crops"]["exact_removed"] == 1
    assert len(out["crops"]) == 2


def _seed_dirty_raw():
    """Write synthetic dirty + duplicate content into lake/raw for pipeline test."""
    ensure_lake_layout()
    # ensure ingest baseline exists
    get_worker("W-INGEST").run(dry_run=False, include_http=False)

    dirty_dir = LAKE_RAW / "general" / "synthetic_dirty"
    dirty_dir.mkdir(parents=True, exist_ok=True)

    # invalid JSON → quarantine
    bad = dirty_dir / "broken.json"
    bad.write_text("{not valid json!!", encoding="utf-8")

    # valid but dirty HTML + duplicates
    good = dirty_dir / "dirty_records.json"
    good.write_text(
        json.dumps(
            {
                "advisories": [
                    {
                        "id": "a1",
                        "title_en": "Cotton advice",
                        "content_en": "<b>Use neem</b> for pink bollworm!!!",
                    },
                    {
                        "id": "a1b",
                        "title_en": "Cotton advice",
                        "content_en": "<b>Use neem</b> for pink bollworm!!!",
                    },
                    {
                        "id": "a2",
                        "title_en": "Cotton advice near",
                        "content_en": "Use neem for pink bollworm!!",
                    },
                    {
                        "id": "a3",
                        "title_en": "Wheat rust",
                        "content_en": "Spray propiconazole for wheat rust disease management.",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return bad, good


def test_quality_pipeline_cleans_and_quarantines():
    bad, good = _seed_dirty_raw()
    report = run_quality_pipeline(dry_run=False, quarantine=True, near_threshold=0.85)

    assert report["validation"]["files_scanned"] >= 1
    assert report["validation"]["invalid"] >= 1
    assert len(report["validation"]["quarantined"]) >= 1

    # cleaned file should exist under processed
    dest = LAKE_PROCESSED / "general" / "synthetic_dirty" / "dirty_records.json"
    assert dest.exists()
    data = json.loads(dest.read_text(encoding="utf-8"))
    # HTML stripped
    content = data["advisories"][0]["content_en"]
    assert "<b>" not in content
    # dups reduced
    assert len(data["advisories"]) < 4

    assert (LAKE_ROOT / "QUALITY_LATEST.json").exists()


def test_workers_validate_clean_dedup():
    _seed_dirty_raw()
    v = get_worker("W-VALIDATE").run(dry_run=False)
    assert v.ok is True
    assert v.metrics["files_scanned"] >= 1

    c = get_worker("W-CLEAN").run(dry_run=False)
    assert c.ok is True
    assert c.metrics["cleaned"] >= 1

    d = get_worker("W-DEDUP").run(dry_run=False, near_threshold=0.85)
    assert d.ok is True


def test_sprint3_pipeline_registered():
    assert "sprint3" in PIPELINES
    assert "quality" in PIPELINES
    result = run_pipeline("quality", dry_run=True)
    assert result.ok is True


def test_api_quality_endpoints():
    # dry-run
    r = client.post("/api/lake/quality?execute=false")
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r2 = client.post("/api/lake/quality?execute=true")
    assert r2.status_code == 200
    r3 = client.get("/api/lake/quality")
    assert r3.status_code == 200
    body = r3.json()
    assert "validation" in body or body.get("ok") is not None
