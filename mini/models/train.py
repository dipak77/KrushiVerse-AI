"""Mini train harness — smoke overfit + checkpointing (Sprint 10)."""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast

from mini.models.config import MiniConfig
from mini.models.model import MiniLM, count_parameters
from mini.paths import MODELS_DIR, TOKENIZER_DIR, ensure_lake_layout, relative_to_repo

PARAM_COUNT_PATH = MODELS_DIR / "PARAM_COUNT.json"
PRETRAIN_LATEST = MODELS_DIR / "PRETRAIN_LATEST.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random token sequences for overfit smoke (labels = next-token)."""
    # fixed small set of sequences so overfit is easy
    base = torch.randint(4, min(vocab_size, 500), (batch_size, seq_len + 1), device=device)
    x = base[:, :-1]
    y = base[:, 1:].clone()
    return x, y


def build_model(config: MiniConfig | None = None, device: str | torch.device = "cpu") -> MiniLM:
    cfg = config or MiniConfig()
    model = MiniLM(cfg)
    return model.to(device)


def write_param_count_report(model: MiniLM, path: Path = PARAM_COUNT_PATH) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = count_parameters(model)
    cfg = model.config.to_dict()
    report = {
        "sprint": "S10",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "vocab_size": cfg["vocab_size"],
            "n_embd": cfg["n_embd"],
            "n_layer": cfg["n_layer"],
            "n_head": cfg["n_head"],
            "n_hidden": cfg["n_hidden"],
            "block_size": cfg["block_size"],
            "tie_weights": cfg["tie_weights"],
        },
        "parameters": counts,
        "target_range": {"min": 800_000, "max": 1_500_000},
        "in_range": 800_000 <= counts["unique_params"] <= 1_500_000,
        "features": ["RoPE", "RMSNorm", "SwiGLU", "weight_tying", "causal_sdpa"],
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_overfit_smoke(
    *,
    steps: int = 50,
    batch_size: int = 8,
    seq_len: int = 64,
    device: str | None = None,
    config: MiniConfig | None = None,
    use_amp: bool | None = None,
) -> dict[str, Any]:
    """Overfit a fixed random batch; loss should drop."""
    cfg = config or MiniConfig()
    if use_amp is None:
        use_amp = cfg.use_amp
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.seed)
    model = build_model(cfg, dev)
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    # same batch every step for pure overfit
    x, y = make_synthetic_batch(batch_size, seq_len, cfg.vocab_size, dev)
    amp_on = bool(use_amp and dev.type == "cuda")
    try:
        scaler = GradScaler("cuda", enabled=amp_on)
    except TypeError:
        scaler = GradScaler(enabled=amp_on)
    losses: list[float] = []
    first_loss = None
    last_loss = None

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        try:
            ctx = autocast("cuda", enabled=amp_on)
        except TypeError:
            ctx = autocast(enabled=amp_on)
        with ctx:
            _, loss = model(x, y)
        assert loss is not None
        if first_loss is None:
            first_loss = float(loss.item())
        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        last_loss = float(loss.item())
        losses.append(last_loss)

    return {
        "steps": steps,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "min_loss": min(losses) if losses else None,
        "loss_dropped": first_loss is not None
        and last_loss is not None
        and last_loss < first_loss * 0.9,
        "losses_tail": losses[-5:],
        "device": str(dev),
        "amp": amp_on,
    }


def save_checkpoint(model: MiniLM, path: Path, extra: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "extra": extra or {},
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> MiniLM:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = MiniConfig.from_dict(payload["config"])
    model = MiniLM(cfg)
    model.load_state_dict(payload["state_dict"])
    return model.to(device)


def run_pretrain_skeleton(
    *,
    dry_run: bool = False,
    overfit_steps: int = 50,
    batch_size: int = 8,
    seq_len: int = 64,
    vocab_size: int | None = None,
    save_ckpt: bool = True,
) -> dict[str, Any]:
    """W-PRETRAIN skeleton: build model, param report, overfit smoke, optional ckpt."""
    ensure_lake_layout()
    # Architecture default vocab keeps ~1M params; override via kwargs if needed.
    # (Full 32k SP tokenizer is larger; embedding resize is a S11 concern.)
    vs = int(vocab_size) if vocab_size is not None else 4096
    cfg = MiniConfig(vocab_size=vs)
    artifacts: list[str] = []

    if dry_run:
        model = build_model(cfg, "cpu")
        counts = count_parameters(model)
        return {
            "ok": True,
            "dry_run": True,
            "sprint": "S10",
            "parameters": counts,
            "config": cfg.to_dict(),
            "in_range": 800_000 <= counts["unique_params"] <= 1_500_000,
        }

    model = build_model(cfg, "cpu")
    param_report = write_param_count_report(model, PARAM_COUNT_PATH)
    artifacts.append(relative_to_repo(PARAM_COUNT_PATH))

    smoke = run_overfit_smoke(
        steps=overfit_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        config=cfg,
        use_amp=False,  # CPU smoke
    )

    ckpt_path = None
    if save_ckpt:
        version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ") + "-smoke"
        ckpt_path = MODELS_DIR / "checkpoints" / f"{version}.pt"
        save_checkpoint(model, ckpt_path, extra={"smoke": smoke})
        artifacts.append(relative_to_repo(ckpt_path))

    ok = bool(param_report.get("in_range")) and bool(smoke.get("loss_dropped"))
    report = {
        "ok": ok,
        "dry_run": False,
        "sprint": "S10",
        "feature_phase": "FP-5b",
        "config": cfg.to_dict(),
        "parameters": param_report.get("parameters"),
        "param_report": relative_to_repo(PARAM_COUNT_PATH),
        "in_range": param_report.get("in_range"),
        "overfit_smoke": smoke,
        "checkpoint": relative_to_repo(ckpt_path) if ckpt_path else None,
        "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    PRETRAIN_LATEST.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # PRETRAIN_LATEST is under models/ which is gitignored except PARAM_COUNT
    artifacts.append(relative_to_repo(PRETRAIN_LATEST))
    report["artifacts"] = list(dict.fromkeys(artifacts))
    return report
