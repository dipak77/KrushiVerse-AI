"""Sprint 0 — Mini factory bootstrap acceptance tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from mini import __feature_phase__, __sprint__, __version__
from mini.contracts import (
    Category,
    DataSplit,
    LanguageCode,
    StandardRecord,
    WorkerResult,
)
from mini.orchestrator.dag import PIPELINES, describe_factory, run_pipeline
from mini.paths import (
    LAKE_RAW,
    LAKE_ROOT,
    SCHEMA_VERSION,
    ensure_lake_layout,
)
from mini.taxonomy import list_categories, list_crops
from mini.workers.base import WORKER_REGISTRY, get_worker, list_workers


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_mini_version_markers():
    # Sprint markers advance with each sprint; earlier artifacts remain valid.
    assert __sprint__ in {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"}
    assert __feature_phase__.startswith("FP-")
    assert __version__


def test_schema_version_locked():
    assert SCHEMA_VERSION == "1.0"


def test_standard_record_valid():
    rec = StandardRecord(
        category=Category.DISEASE,
        crop="Cotton",
        language=LanguageCode.EN,
        question="What is pink bollworm?",
        answer="A major cotton pest that bores into bolls.",
        source="ICAR-CICR open advisory",
        split=DataSplit.TRAIN,
        confidence=0.9,
    )
    d = rec.to_training_dict()
    assert d["schema_version"] == "1.0"
    assert d["category"] == "disease"
    assert d["question"]
    assert d["id"]


def test_standard_record_rejects_empty_answer():
    with pytest.raises(ValidationError):
        StandardRecord(
            category=Category.CROP,
            question="What is cotton?",
            answer="   ",
            source="test",
        )


def test_workers_registered():
    workers = list_workers()
    ids = {w["worker_id"] for w in workers}
    # Core catalog from plan + bootstrap
    required = {
        "W-BOOTSTRAP",
        "W-TAXONOMY",
        "W-INGEST",
        "W-VALIDATE",
        "W-CLEAN",
        "W-DEDUP",
        "W-NORMALIZE",
        "W-LANGDETECT",
        "W-STANDARD",
        "W-ANALYZE",
        "W-QASYNTH",
        "W-KGBUILD",
        "W-TOKEN",
        "W-PRETRAIN",
        "W-SFT",
        "W-EVAL",
        "W-QUANT",
        "W-DEPLOY",
        "W-RAG",
        "W-AGENT",
        "W-INFER",
    }
    missing = required - ids
    assert not missing, f"Missing workers: {missing}"
    assert len(workers) >= 21
    # quality workers from Sprint 3
    assert "W-QUALITY" in ids or "W-VALIDATE" in ids


def test_bootstrap_worker_execute(tmp_path, monkeypatch):
    # Use real lake under repo — bootstrap is idempotent
    worker = get_worker("W-BOOTSTRAP")
    dry = worker.run(dry_run=True)
    assert dry.ok is True
    assert dry.dry_run is True

    result = worker.run(dry_run=False)
    assert result.ok is True
    assert LAKE_ROOT.exists()
    marker = LAKE_ROOT / "BOOTSTRAP.json"
    assert marker.exists()
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["sprint"] == "S0"
    assert data["schema_version"] == "1.0"


def test_ensure_lake_layout_domains():
    ensure_lake_layout()
    assert LAKE_RAW.exists()
    for domain in ("crop", "disease", "soil", "market", "seed"):
        assert (LAKE_RAW / domain).is_dir()


def test_pipelines_defined():
    assert "bootstrap" in PIPELINES
    assert "dry-factory" in PIPELINES
    assert "full" in PIPELINES
    assert PIPELINES["bootstrap"] == ["W-BOOTSTRAP"]


def test_dry_factory_pipeline():
    result = run_pipeline("dry-factory", dry_run=True)
    assert result.ok is True
    assert result.dry_run is True
    assert len(result.steps) == len(PIPELINES["dry-factory"])
    assert all(s.ok for s in result.steps)


def test_bootstrap_pipeline_execute():
    result = run_pipeline("bootstrap", dry_run=False)
    assert result.ok is True
    assert result.steps[0].worker_id == "W-BOOTSTRAP"
    assert (LAKE_ROOT / "BOOTSTRAP.json").exists()


def test_describe_factory():
    info = describe_factory()
    assert info["sprint"] in {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11"}
    assert info["feature_phase"].startswith("FP-")
    assert info["worker_count"] >= 20
    assert "bootstrap" in info["pipelines"]


def test_taxonomy_draft():
    cats = list_categories()
    crops = list_crops()
    assert "soil" in cats
    assert "disease" in cats
    assert len(crops) >= 20
    assert any(c["name_en"] == "Cotton" for c in crops)


def test_cli_list_workers():
    proc = subprocess.run(
        [sys.executable, "-m", "mini.orchestrator", "list-workers"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "W-BOOTSTRAP" in proc.stdout
    assert "W-INGEST" in proc.stdout


def test_cli_status():
    proc = subprocess.run(
        [sys.executable, "-m", "mini.orchestrator", "status"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["feature_phase"].startswith("FP-")


def test_cli_run_dry_factory():
    proc = subprocess.run(
        [sys.executable, "-m", "mini.orchestrator", "run", "dry-factory"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["dry_run"] is True


def test_unknown_worker_raises():
    with pytest.raises(KeyError):
        get_worker("W-DOES-NOT-EXIST")
