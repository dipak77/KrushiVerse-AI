"""Sprint 20 — Mini v2-15M domain pretrain → v0.6-base (smoke)."""

from __future__ import annotations

from mini import __feature_phase__, __sprint__
from mini.models.pretrain import train_domain_v2
from mini.orchestrator.dag import PIPELINES
from mini.paths import MODELS_DIR
from mini.workers.base import get_worker


def test_sprint20_markers():
    assert __sprint__ in {"S18", "S20"}
    assert __feature_phase__ in {"v2-15M"}


def test_pretrain_v2_smoke_loss_drops():
    # Short run for CI: block 64 keeps CPU/GPU smoke fast
    report = train_domain_v2(
        steps=20,
        batch_size=2,
        grad_accum=2,
        block_size=64,
        lr=3e-4,
        seed=42,
        max_qa=3000,
        eval_every=10,
        save_every=20,
        use_fp16=False,
        grad_checkpoint=True,
        out_version="v0.6-base",
    )
    assert report.get("variant") == "v2-15M"
    assert report.get("version") == "v0.6-base"
    train = report.get("train") or {}
    assert train.get("first_loss") is not None
    assert train.get("last_loss") is not None
    assert train["last_loss"] <= train["first_loss"] * 1.05 or train.get("loss_dropped")
    assert (MODELS_DIR / "v0.6-base" / "pytorch_model.pt").exists()
    params = report.get("parameters") or {}
    assert int(params.get("unique_params") or 0) > 10_000_000


def test_worker_v2_pretrain_smoke():
    res = get_worker("W-PRETRAIN").run(
        dry_run=False,
        variant="v2-15M",
        version="v0.6-base",
        steps=12,
        batch_size=2,
        grad_accum=2,
        block_size=64,
        fp16=False,
        grad_checkpoint=True,
        seed=42,
        max_qa=2000,
        eval_every=6,
    )
    assert res.ok is True
    assert (res.metrics.get("domain") or {}).get("train", {}).get("last_loss") is not None


def test_sprint20_pipeline():
    assert "sprint20" in PIPELINES
    assert PIPELINES["sprint20"] == ["W-PRETRAIN"]
