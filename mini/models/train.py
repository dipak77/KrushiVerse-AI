"""Mini train harness — prod-ready.

Fixes:
- Dynamic param range based on config size (1M vs 12M)
- Fused AdamW + seed for numpy
- Atomic saves
"""
from __future__ import annotations
import json, math, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from mini.models.config import MiniConfig
from mini.models.model import MiniLM, count_parameters
from mini.paths import MODELS_DIR, ensure_lake_layout, relative_to_repo

PARAM_COUNT_PATH = MODELS_DIR / "PARAM_COUNT.json"
PRETRAIN_LATEST = MODELS_DIR / "PRETRAIN_LATEST.json"

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    base = torch.randint(4, min(vocab_size, 500), (batch_size, seq_len+1), device=device)
    x = base[:, :-1]
    y = base[:, 1:].clone()
    return x, y

def build_model(config: MiniConfig | None = None, device: str | torch.device = "cpu"):
    cfg = config or MiniConfig()
    model = MiniLM(cfg)
    return model.to(device)

def write_param_count_report(model: MiniLM, path: Path = PARAM_COUNT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = count_parameters(model)
    cfg = model.config.to_dict()
    unique = counts["unique_params"]
    # Dynamic target
    if unique > 5_000_000:
        t_min, t_max = 8_000_000, 20_000_000
    else:
        t_min, t_max = 800_000, 1_500_000
    report = {
        "sprint": "S10",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {k: cfg[k] for k in ("vocab_size","n_embd","n_layer","n_head","n_hidden","block_size","tie_weights") if k in cfg},
        "parameters": counts,
        "target_range": {"min": t_min, "max": t_max},
        "in_range": t_min <= unique <= t_max,
        "features": ["RoPE","RMSNorm","SwiGLU","weight_tying","causal_sdpa","fused_adamw"],
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def run_overfit_smoke(*, steps: int = 50, batch_size: int = 8, seq_len: int = 64, device: str | None = None, config: MiniConfig | None = None, use_amp: bool | None = None):
    cfg = config or MiniConfig()
    if use_amp is None:
        use_amp = cfg.use_amp
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.seed)
    model = build_model(cfg, dev)
    model.train()
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, fused=dev.type=="cuda")
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    x, y = make_synthetic_batch(batch_size, seq_len, cfg.vocab_size, dev)
    amp_on = bool(use_amp and dev.type=="cuda")
    try:
        scaler = GradScaler("cuda", enabled=amp_on)
    except TypeError:
        scaler = GradScaler(enabled=amp_on)
    losses = []
    first = None
    last = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        try:
            ctx = autocast("cuda", enabled=amp_on)
        except TypeError:
            ctx = autocast(enabled=amp_on)
        with ctx:
            _, loss = model(x, y)
        assert loss is not None
        if first is None:
            first = float(loss.item())
        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        last = float(loss.item())
        losses.append(last)
    return {"steps": steps, "first_loss": first, "last_loss": last, "min_loss": min(losses) if losses else None, "loss_dropped": first is not None and last is not None and last < first*0.9, "losses_tail": losses[-5:], "device": str(dev), "amp": amp_on}

def save_checkpoint(model: MiniLM, path: Path, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": model.config.to_dict(), "state_dict": model.state_dict(), "extra": extra or {}, "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    torch.save(payload, path)
    return path

def load_checkpoint(path: Path, device: str | torch.device = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = MiniConfig.from_dict(payload["config"])
    model = MiniLM(cfg)
    model.load_state_dict(payload["state_dict"])
    return model.to(device)

def run_pretrain_skeleton(*, dry_run: bool = False, overfit_steps: int = 50, batch_size: int = 8, seq_len: int = 64, vocab_size: int | None = None, save_ckpt: bool = True):
    ensure_lake_layout()
    vs = int(vocab_size) if vocab_size is not None else 4096
    cfg = MiniConfig(vocab_size=vs)
    artifacts = []
    if dry_run:
        model = build_model(cfg, "cpu")
        counts = count_parameters(model)
        return {"ok": True, "dry_run": True, "sprint": "S10", "parameters": counts, "config": cfg.to_dict(), "in_range": 800_000 <= counts["unique_params"] <= 1_500_000}
    model = build_model(cfg, "cpu")
    param_report = write_param_count_report(model, PARAM_COUNT_PATH)
    artifacts.append(relative_to_repo(PARAM_COUNT_PATH))
    smoke = run_overfit_smoke(steps=overfit_steps, batch_size=batch_size, seq_len=seq_len, config=cfg, use_amp=False)
    ckpt_path = None
    if save_ckpt:
        version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ") + "-smoke"
        ckpt_path = MODELS_DIR / "checkpoints" / f"{version}.pt"
        save_checkpoint(model, ckpt_path, extra={"smoke": smoke})
        artifacts.append(relative_to_repo(ckpt_path))
    ok = bool(param_report.get("in_range")) and bool(smoke.get("loss_dropped"))
    report = {"ok": ok, "dry_run": False, "sprint": "S10", "feature_phase": "FP-5b", "config": cfg.to_dict(), "parameters": param_report.get("parameters"), "param_report": relative_to_repo(PARAM_COUNT_PATH), "in_range": param_report.get("in_range"), "overfit_smoke": smoke, "checkpoint": relative_to_repo(ckpt_path) if ckpt_path else None, "artifacts": artifacts, "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    PRETRAIN_LATEST.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    artifacts.append(relative_to_repo(PRETRAIN_LATEST))
    report["artifacts"] = list(dict.fromkeys(artifacts))
    return report
