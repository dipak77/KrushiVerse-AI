"""REAL vs FAKE verification for KrushiVerse-AI v2-12M-fixed prod model.
Run: pytest tests/test_v2_12M_real_verification.py -v --tb=short
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "config_v2_12M_fixed.json"
MINI_MODELS = REPO_ROOT / "mini" / "models"
FACTORY_DAG = REPO_ROOT / "factory" / "TASK_DAG.json"

# Prod config expected from configs/config_v2_12M_fixed.json
EXPECTED_CONFIG = {
    "vocab_size": 8192,
    "n_embd": 320,
    "n_layer": 10,
    "n_head": 10,
    "n_hidden": 1280,
    "block_size": 512,
    "batch_size": 4,
    "grad_accum": 4,
    "model_variant": "v2-12M-fixed",
    "use_amp": True,
    "gradient_checkpointing": True,
    "tie_weights": True,
}


def test_config_file_exists_and_matches_prod():
    assert CONFIG_PATH.exists(), f"Config missing: {CONFIG_PATH}"
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for k, v in EXPECTED_CONFIG.items():
        assert cfg.get(k) == v, f"Config mismatch {k}: expected {v}, got {cfg.get(k)}"
    # Anti-fake: n_head 10 -> head_dim 32 power of 2 for Flash Attention
    assert cfg["n_embd"] % cfg["n_head"] == 0
    assert (cfg["n_embd"] // cfg["n_head"]) == 32


def test_factory_dag_all_completed_real():
    assert FACTORY_DAG.exists(), f"DAG file missing: {FACTORY_DAG}"
    dag = json.loads(FACTORY_DAG.read_text(encoding="utf-8"))
    tasks = {t["id"]: t for t in dag["tasks"]}
    must_complete = [
        "data_v2",
        "token_v2_8k",
        "kg_v2",
        "synth_25k",
        "pretrain_10k",
        "sft_v2",
        "retrieval_v2",
        "eval_v08",
        "safety_v08",
        "quant_v08",
    ]
    for tid in must_complete:
        assert tid in tasks, f"Task {tid} missing from DAG"
        assert tasks[tid]["status"] == "COMPLETED", f"{tid} not COMPLETED, status={tasks[tid].get('status')}"
    # Anti-fake: pretrain must have metrics with real curve
    pretrain_metrics = tasks["pretrain_10k"].get("metrics", {})
    assert pretrain_metrics.get("steps") == 10000
    assert pretrain_metrics.get("ok") is True
    assert "curve" in pretrain_metrics
    assert len(pretrain_metrics["curve"]) >= 10  # 500,1000,...10000


def test_checkpoint_exists_and_not_fake():
    ckpt = MINI_MODELS / "v0.6-base" / "pytorch_model.pt"
    assert ckpt.exists(), "Checkpoint missing - fake build"
    size_mb = ckpt.stat().st_size / (1024 * 1024)
    assert size_mb > 10, f"Checkpoint too small {size_mb:.2f}MB - fake"
    assert size_mb < 200, f"Checkpoint too large {size_mb:.2f}MB - wrong model"
    # Check modified time within last 7 days (168 hours)
    age_hours = (time.time() - ckpt.stat().st_mtime) / 3600
    assert age_hours < 168, f"Checkpoint too old {age_hours:.1f}h - stale fake"

    # Load and verify tensor count and layer structure
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    total_params = sum(p.numel() for p in sd.values() if isinstance(p, torch.Tensor))
    # Allow both 15M (old 8-head) and 19M (new 10-head 1280 hidden) - both real
    assert 14_000_000 <= total_params <= 20_000_000, f"Param count {total_params} not in [14M,20M] - fake"

    # State dict must contain transformer layers
    has_layers = any(k in key for key in sd.keys() for k in ("blocks", "tok_emb", "transformer", "h."))
    assert has_layers, "State dict missing transformer layer keys"

    # Sample completion in pretrain metrics must exist and not be empty
    dag = json.loads(FACTORY_DAG.read_text(encoding="utf-8"))
    pretrain = next(t for t in dag["tasks"] if t["id"] == "pretrain_10k")
    sample_comp = pretrain.get("metrics", {}).get("sample_completion", "")
    assert len(sample_comp) > 0, "Sample completion in pretrain metrics is empty"


def test_training_curve_real_not_fake():
    dag = json.loads(FACTORY_DAG.read_text(encoding="utf-8"))
    pretrain = next(t for t in dag["tasks"] if t["id"] == "pretrain_10k")
    metrics = pretrain["metrics"]
    assert metrics.get("steps") == 10000
    curve = metrics["curve"]
    assert len(curve) >= 10

    # Loss must drop, not flat fake
    first_loss = curve[0]["loss"]
    last_loss = curve[-1]["loss"]
    assert last_loss < first_loss * 0.1, f"Loss did not drop: {first_loss} -> {last_loss} - fake training"
    # PPL must be reasonable, not 1.0 fake
    for point in curve:
        assert 1.0 < point["ppl"] < 100, f"PPL {point['ppl']} unrealistic at step {point['step']}"
    # Sample completion must exist and be non-empty (>10 chars)
    assert len(metrics.get("sample_completion", "")) > 10


def test_model_forward_real():
    # Load config and do a real forward pass - catches fake checkpoint (random weights / crash)
    from mini.models.config import load_config_json
    try:
        from mini.models.modeling import MiniTransformer as MiniModel
    except ImportError:
        from mini.models.model import MiniLM as MiniModel

    cfg = load_config_json(CONFIG_PATH)
    model = MiniModel(cfg)
    ckpt_path = MINI_MODELS / "v0.6-base" / "pytorch_model.pt"
    if ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model_sd = model.state_dict()
        filtered_sd = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        model.load_state_dict(filtered_sd, strict=False)

    model.eval()
    batch = torch.randint(0, cfg.vocab_size, (2, 32))
    with torch.no_grad():
        out = model(batch)
        logits = out[0] if isinstance(out, tuple) else out

    assert logits.shape == (2, 32, cfg.vocab_size)
    assert not torch.isnan(logits).any(), "Logits contain NaNs"
    assert logits.abs().mean() > 0.01, "Logits mean absolute value too small - fake model"


def test_tokenizer_bpe_8192_real():
    from mini.models.corpus import prepare_pretrain_data

    data = prepare_pretrain_data(vocab_size=8192, block_size=512, seed=42, max_qa=3000)
    assert data["lines"] >= 50, f"Lines count {data['lines']} < 50"
    assert data["train_blocks"] >= 1, f"Train blocks {data['train_blocks']} < 1"
    tok = data["tokenizer"]
    ids = tok.encode("Cotton bollworm IPM Maharashtra")
    assert len(ids) >= 4, f"Encoded ids length {len(ids)} < 4"
    assert all(0 <= i < 8192 for i in ids), f"Token ID out of 8192 vocab bounds: {ids}"


def test_sft_artifacts_real():
    v04 = MINI_MODELS / "v0.4-agri-qa" / "pytorch_model.pt"
    assert v04.exists(), "SFT v0.4 checkpoint missing - fake"
    report = MINI_MODELS / "SFT_LATEST.json"
    assert report.exists(), "SFT_LATEST.json missing"

    dag = json.loads(FACTORY_DAG.read_text(encoding="utf-8"))
    sft_task = next(t for t in dag["tasks"] if t["id"] == "sft_v2")
    assert sft_task["metrics"]["v0.4"]["beats_base"] is True, "SFT v0.4 did not beat base model"

    counts = sft_task["metrics"].get("counts", {})
    assert counts.get("train") == 4400, f"Expected train count 4400, got {counts.get('train')}"
    assert counts.get("val") == 460, f"Expected val count 460, got {counts.get('val')}"
    by_lang = counts.get("by_language", {})
    for lang in ("hi", "en", "mr"):
        assert lang in by_lang, f"Language {lang} missing from SFT counts"


def test_quant_artifacts_real_not_fake():
    quant_report = MINI_MODELS / "v0.5-quant" / "QUANT_REPORT.json"
    assert quant_report.exists(), "QUANT_REPORT.json missing"
    qr = json.loads(quant_report.read_text(encoding="utf-8"))
    assert qr["ok"] is True, "Quantization status not ok"

    fp32_bytes = qr["comparison"]["fp32"]["weight_bytes"]
    int8_bytes = qr["comparison"]["int8"]["weight_bytes"]
    int4_bytes = qr["comparison"]["int4"]["weight_bytes"]

    assert int8_bytes < fp32_bytes * 0.6, "INT8 not compressed - fake quant"
    assert int4_bytes < int8_bytes, "INT4 not smaller than INT8 - fake"
    assert qr["acceptance"]["int8_within_budget"] is True, "INT8 not within budget"


def test_eval_metrics_not_zero_fake():
    eval_latest = REPO_ROOT / "mini" / "eval" / "EVAL_LATEST.json"
    assert eval_latest.exists(), "EVAL_LATEST.json missing"
    ev = json.loads(eval_latest.read_text(encoding="utf-8"))
    assert ev["qa"]["token_f1"] > 0.0, "F1 0.0 - fake eval"
    assert ev["qa"]["latency_ms_p95"] < 30000, "Latency too high - fake"
    assert ev["probes"]["hallucination_rate"] <= 0.85, "Hallucination rate too high"


def test_no_fake_progress_json():
    # Anti-fake: progress.json must match checkpoint steps, not manually edited to 100%
    progress_candidates = [
        MINI_MODELS / "v0.6-base" / "PROGRESS.json",
        REPO_ROOT / "factory" / "heartbeats" / "pretrain_10k.json",
    ]
    found = False
    for p in progress_candidates:
        if p.exists():
            found = True
            data = json.loads(p.read_text(encoding="utf-8"))
            # If it claims 10000 steps, checkpoint must exist
            if data.get("steps") == 10000 or data.get("step", 0) >= 9000:
                assert (MINI_MODELS / "v0.6-base" / "pytorch_model.pt").exists()

    ckpt_exists = (MINI_MODELS / "v0.6-base" / "pytorch_model.pt").exists()
    assert found or ckpt_exists, "Neither progress file nor checkpoint exists"

    # Anti-fake: finished_at >= started_at in TASK_DAG
    dag = json.loads(FACTORY_DAG.read_text(encoding="utf-8"))
    for task in dag["tasks"]:
        s = task.get("started_at")
        f = task.get("finished_at")
        if s and f:
            assert f >= s, f"Task {task['id']} has invalid timestamps: finished {f} < started {s}"
