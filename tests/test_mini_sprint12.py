"""Sprint 12 — instruction + agri-QA SFT (W-SFT v0.3/v0.4)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mini import __feature_phase__, __sprint__
from mini.models.sft import run_sft
from mini.models.sft_data import (
    build_sft_records,
    exact_match,
    format_sft_example,
    prompt_only,
    token_f1,
)
from mini.orchestrator.dag import PIPELINES
from mini.workers.base import get_worker, list_workers
from app.main import app

client = TestClient(app)


def test_sprint12_markers():
    assert __sprint__ in {"S12", "S13", "S14", "S15", "S16", "S17"}
    assert __feature_phase__ in {"FP-7", "E5-eval", "E5-quant", "FP-8", "FP-9", "FP-10"}


def test_sft_worker_ready():
    assert {w["worker_id"]: w["status"] for w in list_workers()}["W-SFT"] == "ready"


def test_sft_format_and_metrics():
    text = format_sft_example(
        system="You are Krushi Mitra.",
        user="What is IPM?",
        assistant="Integrated Pest Management uses ETL and biocontrol first.",
    )
    assert "### System:" in text and "### User:" in text and "### Assistant:" in text
    prompt = prompt_only(text)
    assert prompt.endswith("### Assistant:\n")
    assert token_f1("IPM uses ETL", "IPM uses ETL and traps") > 0.3
    assert exact_match("hello", "hello") == 1.0
    assert exact_match("hello", "Hello!") == 0.0


def test_build_sft_records():
    data = build_sft_records(max_train=200, max_val=40, seed=42)
    assert data["counts"]["train"] >= 32
    assert data["counts"]["val"] >= 1
    packs = data["counts"].get("by_pack") or {}
    # safety templates upsampled and/or rag examples present when lake has QA
    assert len(data["train"]) >= 32
    sample = data["train"][0]
    assert "text" in sample and "### Assistant:" in sample["text"]
    assert packs  # at least one pack key


def test_run_sft_improves():
    report = run_sft(
        dry_run=False,
        steps_v03=30,
        steps_v04=30,
        batch_size=4,
        seed=42,
        max_train=800,
        max_val=80,
        lr=3e-3,
    )
    assert report.get("ok") is True
    assert report.get("sprint") == "S12"
    base = report.get("base_val") or {}
    v4 = ((report.get("v0.4") or {}).get("stage") or {})
    assert v4.get("last_loss") is not None
    assert v4.get("first_loss") is not None
    # beats_base via F1, val loss, or train loss drop
    assert (report.get("v0.4") or {}).get("beats_base") is True
    assert report.get("artifacts")
    assert (report.get("counts") or {}).get("train", 0) >= 32
    # demos present for side-by-side
    assert isinstance(report.get("demos"), list)


def test_worker_sft_short():
    res = get_worker("W-SFT").run(
        dry_run=False,
        steps_v03=20,
        steps_v04=20,
        batch_size=4,
        seed=42,
        max_train=600,
        max_val=60,
    )
    assert res.ok is True
    assert res.metrics.get("sprint") == "S12"
    assert (res.metrics.get("v0.4") or {}).get("beats_base") is True


def test_sprint12_pipeline():
    assert "sprint12" in PIPELINES
    assert PIPELINES["sprint12"] == ["W-SFT"]
    assert "sft" in PIPELINES


def test_api_sft_s12():
    r = client.post(
        "/api/lake/sft?execute=true&steps_v03=15&steps_v04=15&seed=42&batch_size=4&max_train=400&max_val=40"
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["metrics"]["sprint"] == "S12"
