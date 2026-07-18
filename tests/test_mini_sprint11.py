"""Sprint 11 — domain pretrain Mini v0.2-base (val PPL + seed)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mini import __feature_phase__, __sprint__
from mini.models.corpus import DomainTokenizer, prepare_pretrain_data
from mini.models.pretrain import run_pretrain_s11, train_domain
from mini.orchestrator.dag import PIPELINES
from mini.workers.base import get_worker, list_workers
from app.main import app

client = TestClient(app)


def test_sprint11_markers():
    assert __sprint__ in {"S11", "S12"}
    assert __feature_phase__ in {"FP-6", "FP-7"}


def test_pretrain_worker_ready():
    assert {w["worker_id"]: w["status"] for w in list_workers()}["W-PRETRAIN"] == "ready"


def test_corpus_and_tokenizer():
    data = prepare_pretrain_data(vocab_size=4096, block_size=64, seed=42, max_qa=3000)
    assert data["lines"] >= 50
    assert data["train_blocks"] >= 1
    tok: DomainTokenizer = data["tokenizer"]
    ids = tok.encode("Cotton bollworm IPM Maharashtra")
    assert len(ids) >= 4
    assert all(0 <= i < 4096 for i in ids)


def test_domain_pretrain_ppl_improves():
    # Short run for CI; still should reduce train loss / val ppl vs start
    report = train_domain(
        steps=40,
        batch_size=4,
        block_size=64,
        vocab_size=4096,
        seed=42,
        max_qa=8000,
        eval_every=20,
    )
    assert report.get("corpus", {}).get("train_blocks", 0) >= 1
    train = report.get("train") or {}
    assert train.get("first_loss") is not None and train.get("last_loss") is not None
    assert train["last_loss"] <= train["first_loss"] * 1.05  # allow tiny noise
    val = report.get("val") or {}
    # Prefer ppl improvement; accept strong train loss drop as fallback
    improved = bool(val.get("ppl_improved")) or (
        train["last_loss"] < train["first_loss"] * 0.9
    )
    assert improved
    assert report.get("checkpoint")


def test_seed_reproducible_short():
    a = train_domain(steps=15, batch_size=4, block_size=64, seed=7, max_qa=4000, eval_every=15)
    b = train_domain(steps=15, batch_size=4, block_size=64, seed=7, max_qa=4000, eval_every=15)
    la = (a.get("train") or {}).get("last_loss")
    lb = (b.get("train") or {}).get("last_loss")
    assert la is not None and lb is not None
    assert abs(la - lb) < 1e-3


def test_worker_domain_mode():
    res = get_worker("W-PRETRAIN").run(
        dry_run=False,
        mode="domain",
        steps=25,
        batch_size=4,
        block_size=64,
        seed=42,
        max_qa=5000,
    )
    assert res.ok is True
    domain = res.metrics.get("domain") or {}
    assert (domain.get("train") or {}).get("last_loss") is not None


def test_sprint11_pipeline():
    assert "sprint11" in PIPELINES
    assert PIPELINES["sprint11"] == ["W-PRETRAIN"]


def test_api_pretrain_s11():
    r = client.post("/api/lake/pretrain?execute=true&steps=20&mode=domain&seed=42&batch_size=4")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["metrics"]["sprint"] == "S11"
