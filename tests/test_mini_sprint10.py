"""Sprint 10 — Mini ~1M architecture + train harness (W-PRETRAIN skeleton)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from mini import __feature_phase__, __sprint__
from mini.models.config import MiniConfig
from mini.models.model import MiniLM, count_parameters
from mini.models.train import run_overfit_smoke, run_pretrain_skeleton, write_param_count_report
from mini.paths import MODELS_DIR
from mini.workers.base import get_worker, list_workers
from mini.orchestrator.dag import PIPELINES
from app.main import app

client = TestClient(app)


def test_sprint10_markers():
    assert __sprint__ in {"S10", "S11", "S12", "S13"}
    assert __feature_phase__ in {"FP-5b", "FP-6", "FP-7", "E5-eval"}


def test_pretrain_worker_ready():
    assert {w["worker_id"]: w["status"] for w in list_workers()}["W-PRETRAIN"] == "ready"


def test_param_count_in_range():
    cfg = MiniConfig()
    model = MiniLM(cfg)
    counts = count_parameters(model)
    n = counts["unique_params"]
    assert 800_000 <= n <= 1_500_000
    report = write_param_count_report(model)
    assert report["in_range"] is True
    assert (MODELS_DIR / "PARAM_COUNT.json").exists()


def test_model_forward_shapes():
    import torch

    cfg = MiniConfig(vocab_size=4096, block_size=128)
    model = MiniLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(x, x)
    assert logits.shape == (2, 32, cfg.vocab_size)
    assert loss is not None and loss.item() > 0


def test_overfit_smoke_loss_drops():
    smoke = run_overfit_smoke(steps=30, batch_size=4, seq_len=32, use_amp=False)
    assert smoke["loss_dropped"] is True
    assert smoke["last_loss"] < smoke["first_loss"]


def test_run_pretrain_skeleton():
    report = run_pretrain_skeleton(dry_run=False, overfit_steps=30, batch_size=4, seq_len=32)
    assert report["ok"] is True
    assert report["in_range"] is True
    assert report["overfit_smoke"]["loss_dropped"] is True


def test_worker_execute_s10():
    res = get_worker("W-PRETRAIN").run(dry_run=False, overfit_steps=25, batch_size=4, seq_len=32)
    assert res.ok is True
    assert res.metrics["in_range"] is True


def test_sprint10_pipeline_registered():
    assert "sprint10" in PIPELINES
    assert "W-PRETRAIN" in PIPELINES["sprint10"]


def test_api_pretrain():
    r = client.post("/api/lake/pretrain?execute=true&steps=20")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    s = client.get("/api/lake/pretrain")
    assert s.status_code == 200
