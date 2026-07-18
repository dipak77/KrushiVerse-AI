"""Sprint 1 — Domain taxonomy freeze acceptance tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from mini import __sprint__, __version__
from mini.taxonomy.aliases import CROP_ALIASES, resolve_crop_name, resolve_crops_in_text
from mini.taxonomy.domains import TAXONOMY_STATUS, TAXONOMY_VERSION, list_categories, list_crops
from mini.taxonomy.regions import resolve_district
from mini.taxonomy.service import taxonomy_service
from mini.taxonomy.units import convert_area, convert_mass, normalize_unit_token
from mini.taxonomy.validator import full_validation_report, validate_taxonomy_integrity
from mini.workers.base import get_worker, list_workers
from mini.orchestrator.dag import PIPELINES, run_pipeline
from app.knowledge.query_understanding import query_understanding
from app.main import app

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_sprint1_version_markers():
    assert __sprint__ == "S1"
    assert "0.2" in __version__
    assert TAXONOMY_VERSION == "1.0.0"
    assert TAXONOMY_STATUS == "frozen"


def test_taxonomy_integrity_ok():
    report = validate_taxonomy_integrity()
    assert report["ok"] is True, report["errors"]
    assert report["counts"]["crops"] >= 20
    assert report["counts"]["categories"] >= 10


def test_platform_kb_coverage_ok():
    report = full_validation_report()
    assert report["ok"] is True, {
        "integrity_errors": report["integrity"]["errors"],
        "coverage_errors": report["platform_coverage"]["errors"],
    }


def test_all_taxonomy_crops_have_aliases():
    for crop in list_crops():
        assert crop["name_en"] in CROP_ALIASES
        assert crop.get("name_hi"), f"missing Hindi for {crop['id']}"
        assert crop.get("name_mr"), f"missing Marathi for {crop['id']}"


def test_resolve_crop_en_mr_hi():
    assert resolve_crop_name("cotton") == "Cotton"
    assert resolve_crop_name("कापूस") == "Cotton"
    assert resolve_crop_name("कपास") == "Cotton"
    assert resolve_crop_name("डाळिंब") == "Pomegranate"
    assert resolve_crop_name("गेहूं") == "Wheat"
    assert resolve_crop_name("soybean") == "Soybean"


def test_resolve_multiple_crops_in_text():
    crops = resolve_crops_in_text("कापूस आणि सोयाबीन market price")
    assert "Cotton" in crops
    assert "Soybean" in crops


def test_district_resolve_pune():
    r = resolve_district("Pune")
    assert r is not None
    assert r["state_id"] == "MH"
    assert r["district"] == "Pune"


def test_units_mass_and_area():
    assert abs(convert_mass(1, "quintal", "kg") - 100.0) < 1e-6
    acre_ha = convert_area(1, "acre", "ha")
    assert acre_ha is not None and 0.4 < acre_ha < 0.41
    assert normalize_unit_token("mm") == "mm" or normalize_unit_token("mm") is not None
    assert normalize_unit_token("₹") == "INR" or normalize_unit_token("rs") == "INR"


def test_taxonomy_service_summary():
    s = taxonomy_service.summary()
    assert s["version"] == "1.0.0"
    assert s["status"] == "frozen"
    assert s["crops"] == 22
    assert s["categories"] >= 14


def test_query_understanding_uses_taxonomy_aliases():
    plan = query_understanding.understand("कपास में यूरिया कब डालें?")
    assert "Cotton" in plan.crops
    assert "fertilizer" in plan.intents or "soil" in plan.categories or "fertilizer" in plan.categories


def test_query_understanding_marathi_cotton():
    plan = query_understanding.understand("कापूस रोग नियंत्रण")
    assert "Cotton" in plan.crops
    assert plan.language_hint == "mr"


def test_taxonomy_worker_ready():
    workers = {w["worker_id"]: w for w in list_workers()}
    assert "W-TAXONOMY" in workers
    assert workers["W-TAXONOMY"]["status"] == "ready"
    res = get_worker("W-TAXONOMY").run(dry_run=True)
    assert res.ok is True


def test_normalize_worker_uses_taxonomy():
    res = get_worker("W-NORMALIZE").run(dry_run=True, text="कापूस खत")
    assert res.ok is True
    assert res.metrics.get("resolved_crop") == "Cotton"


def test_sprint1_pipeline():
    assert "sprint1" in PIPELINES
    assert "taxonomy" in PIPELINES
    result = run_pipeline("sprint1", dry_run=True)
    assert result.ok is True


def test_cli_taxonomy_validate():
    proc = subprocess.run(
        [sys.executable, "-m", "mini.orchestrator", "taxonomy-validate"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True


def test_api_taxonomy_endpoints():
    r = client.get("/api/taxonomy")
    assert r.status_code == 200
    data = r.json()
    assert data["summary"]["version"] == "1.0.0"
    assert len(data["crops"]) >= 20

    r2 = client.get("/api/taxonomy/validate")
    assert r2.status_code == 200
    assert r2.json()["ok"] is True

    r3 = client.get("/api/taxonomy/resolve", params={"text": "कापूस Pune"})
    assert r3.status_code == 200
    body = r3.json()
    assert "Cotton" in body["crops_in_text"]


def test_categories_include_phase1_domains():
    cats = set(list_categories())
    for required in (
        "soil",
        "weather",
        "crop",
        "disease",
        "pest",
        "fertilizer",
        "irrigation",
        "scheme",
        "market",
        "finance",
        "machinery",
    ):
        assert required in cats
