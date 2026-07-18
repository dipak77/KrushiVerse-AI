"""Sprint 9 — W-TOKEN domain SentencePiece tokenizer (30–50k)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mini import __feature_phase__, __sprint__
from mini.paths import TOKENIZER_DIR, ensure_lake_layout
from mini.tokenizer.train import (
    VOCAB_MAX,
    VOCAB_MIN,
    collect_user_defined_symbols,
    fertility_report,
    run_tokenizer_train,
)
from mini.workers.base import get_worker, list_workers
from mini.orchestrator.dag import PIPELINES
from app.main import app

client = TestClient(app)


def test_sprint9_markers():
    assert __sprint__ in {"S9", "S10", "S11", "S12"}
    assert __feature_phase__ in {"FP-5", "FP-5b", "FP-6", "FP-7"}


def test_token_worker_ready():
    assert {w["worker_id"]: w["status"] for w in list_workers()}["W-TOKEN"] == "ready"


def test_user_defined_symbols_cover_domain():
    syms = collect_user_defined_symbols()
    assert len(syms) >= 50
    joined = " ".join(syms).lower()
    assert "cotton" in joined or "कापूस" in joined
    assert "urea" in joined or "नत्र" in joined


def test_tokenizer_train_meets_targets():
    ensure_lake_layout()
    # Reuse existing v0.1 if valid; else train (can take a few minutes)
    latest = TOKENIZER_DIR / "TOKENIZER_LATEST.json"
    model = TOKENIZER_DIR / "v0.1" / "sp_agri.model"
    if latest.exists() and model.exists():
        import json

        meta = json.loads(latest.read_text(encoding="utf-8"))
        if meta.get("ok") and int(meta.get("actual_vocab_size") or 0) >= VOCAB_MIN:
            report = meta
            # still verify fertility report loads
            fert = fertility_report(model)
            assert fert["mean_fertility"] > 0
            assert VOCAB_MIN <= int(meta["actual_vocab_size"]) <= VOCAB_MAX
            assert meta.get("fertility_improved") is True
            return

    report = run_tokenizer_train(
        dry_run=False,
        vocab_size=32000,
        version="v0.1",
        train_baseline=True,
        max_qa_lines=15000,
    )
    assert report["ok"] is True
    actual = int((report.get("train") or {}).get("actual_vocab_size") or 0)
    assert VOCAB_MIN <= actual <= VOCAB_MAX
    assert (report.get("targets") or {}).get("vocab_ok") is True
    assert (report.get("fertility") or {}).get("improved") is True
    assert model.exists()


def test_worker_execute_s9():
    res = get_worker("W-TOKEN").run(dry_run=False, vocab_size=32000, version="v0.1")
    assert res.ok is True
    actual = int((res.metrics.get("train") or {}).get("actual_vocab_size") or 0)
    assert actual >= VOCAB_MIN


def test_sprint9_pipeline_registered():
    assert "sprint9" in PIPELINES
    assert PIPELINES["sprint9"] == ["W-TOKEN"]
    assert "token" in PIPELINES


def test_api_tokenizer_endpoints():
    r = client.get("/api/lake/tokenizer")
    assert r.status_code == 200
    # ensure model exists from prior test/train
    if not (TOKENIZER_DIR / "TOKENIZER_LATEST.json").exists():
        client.post("/api/lake/tokenizer?execute=true&vocab_size=32000")
    s = client.get("/api/lake/tokenizer")
    body = s.json()
    assert body.get("ok") is True
    assert int(body.get("actual_vocab_size") or 0) >= VOCAB_MIN


def test_demo_tokenize_marathi_english():
    model = TOKENIZER_DIR / "v0.1" / "sp_agri.model"
    assert model.exists()
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model))
    en = sp.encode("Cotton bollworm IPM in Nashik", out_type=str)
    mr = sp.encode("कापूस पिकावरील कीड नियंत्रण", out_type=str)
    assert len(en) >= 2
    assert len(mr) >= 2
    # force-included crop should appear as a piece (possibly with ▁ prefix handling)
    pieces_flat = " ".join(en)
    assert "Cotton" in pieces_flat or "cotton" in pieces_flat.lower()
