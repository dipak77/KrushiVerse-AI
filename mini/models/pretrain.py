"""Domain pretraining loop for MiniLM (Sprint 11 / FP-6).

Streams packed agri token blocks, trains with reproducible seed, reports
train loss + val perplexity, and writes checkpoint v0.2-base (local only).
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from mini.models.config import MiniConfig
from mini.models.corpus import DomainTokenizer, prepare_pretrain_data
from mini.models.model import MiniLM, count_parameters
from mini.models.train import (
    PARAM_COUNT_PATH,
    PRETRAIN_LATEST,
    build_model,
    run_overfit_smoke,
    save_checkpoint,
    set_seed,
    write_param_count_report,
)
from mini.paths import MODELS_DIR, ensure_lake_layout, relative_to_repo

V02_DIR = MODELS_DIR / "v0.2-base"
V02_CKPT = V02_DIR / "pytorch_model.pt"
V02_CFG = V02_DIR / "config.json"
V02_TOK = V02_DIR / "tokenizer.json"
V02_REPORT = V02_DIR / "train_report.json"


class BlockDataset(Dataset):
    def __init__(self, blocks: list[list[int]]):
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int):
        ids = torch.tensor(self.blocks[idx], dtype=torch.long)
        # next-token LM: input[:-1], labels[1:] — use full block with shift in collate
        return ids


def _batchify(batch: list[torch.Tensor], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    # all same length from packing
    x = torch.stack(batch, dim=0)
    # inputs = all but last, labels = all but first
    inp = x[:, :-1]
    lab = x[:, 1:].clone()
    return inp, lab


@torch.no_grad()
def eval_perplexity(
    model: MiniLM,
    blocks: list[list[int]],
    *,
    device: torch.device,
    batch_size: int = 8,
    max_batches: int = 50,
) -> dict[str, float]:
    if not blocks:
        return {"loss": float("inf"), "ppl": float("inf"), "batches": 0}
    model.eval()
    ds = BlockDataset(blocks)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: _batchify(b))
    total_loss = 0.0
    n = 0
    for i, (inp, lab) in enumerate(loader):
        if i >= max_batches:
            break
        inp = inp.to(device)
        lab = lab.to(device)
        _, loss = model(inp, lab)
        if loss is None:
            continue
        total_loss += float(loss.item())
        n += 1
    model.train()
    if n == 0:
        return {"loss": float("inf"), "ppl": float("inf"), "batches": 0}
    avg = total_loss / n
    ppl = math.exp(min(20.0, avg))  # clamp overflow
    return {"loss": round(avg, 6), "ppl": round(ppl, 4), "batches": n}


def train_domain(
    *,
    steps: int = 200,
    batch_size: int = 8,
    block_size: int = 128,
    vocab_size: int = 4096,
    lr: float | None = None,
    seed: int = 42,
    device: str | None = None,
    max_qa: int = 25_000,
    eval_every: int = 50,
) -> dict[str, Any]:
    """Run domain LM pretrain; return metrics + paths."""
    ensure_lake_layout()
    set_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data = prepare_pretrain_data(
        vocab_size=vocab_size,
        block_size=block_size,
        seed=seed,
        max_qa=max_qa,
    )
    tok: DomainTokenizer = data["tokenizer"]
    train_blocks: list[list[int]] = data["train"]
    val_blocks: list[list[int]] = data["val"]

    cfg = MiniConfig(
        vocab_size=vocab_size,
        block_size=max(block_size, 128),
        learning_rate=lr if lr is not None else 3e-3,
        batch_size=batch_size,
        seed=seed,
        max_steps=steps,
    )
    model = build_model(cfg, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # initial val PPL (random-ish init)
    ppl_start = eval_perplexity(model, val_blocks or train_blocks[: max(1, len(train_blocks) // 10)], device=dev, batch_size=batch_size)

    if not train_blocks:
        return {
            "ok": False,
            "error": "no train blocks",
            "corpus": {k: data[k] for k in ("lines", "docs", "blocks", "train_blocks", "val_blocks")},
        }

    loader = DataLoader(
        BlockDataset(train_blocks),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: _batchify(b),
    )
    model.train()
    losses: list[float] = []
    curve: list[dict[str, Any]] = []
    it = iter(loader)
    step = 0
    while step < steps:
        try:
            inp, lab = next(it)
        except StopIteration:
            it = iter(loader)
            inp, lab = next(it)
        inp = inp.to(dev)
        lab = lab.to(dev)
        opt.zero_grad(set_to_none=True)
        _, loss = model(inp, lab)
        assert loss is not None
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        lv = float(loss.item())
        losses.append(lv)
        step += 1
        if step % eval_every == 0 or step == steps:
            ppl_mid = eval_perplexity(model, val_blocks or train_blocks[:8], device=dev, batch_size=batch_size)
            curve.append(
                {
                    "step": step,
                    "train_loss": round(lv, 6),
                    "val_loss": ppl_mid["loss"],
                    "val_ppl": ppl_mid["ppl"],
                }
            )

    ppl_end = eval_perplexity(model, val_blocks or train_blocks[: max(1, len(train_blocks) // 10)], device=dev, batch_size=batch_size)
    first = losses[0] if losses else None
    last = losses[-1] if losses else None
    improved = (
        ppl_start.get("ppl", float("inf")) < float("inf")
        and ppl_end.get("ppl", float("inf")) < ppl_start["ppl"]
    ) or (
        first is not None and last is not None and last < first * 0.95
    )

    # sample completion (raw tokens)
    sample_text = None
    sample_ids = None
    try:
        model.eval()
        prompt = tok.encode("Cotton IPM advisory for Maharashtra", add_special=True)[:32]
        idx = torch.tensor([prompt], dtype=torch.long, device=dev)
        out = model.generate(idx, max_new_tokens=24, temperature=0.9)
        sample_ids = out[0].tolist()
        sample_text = tok.decode(sample_ids)
        model.train()
    except Exception as e:
        sample_text = f"<generate failed: {e}>"

    # save artifacts (local)
    V02_DIR.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model,
        V02_CKPT,
        extra={
            "sprint": "S11",
            "seed": seed,
            "steps": steps,
            "ppl_start": ppl_start,
            "ppl_end": ppl_end,
        },
    )
    V02_CFG.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    tok.save(V02_TOK)

    report = {
        "ok": bool(improved),
        "sprint": "S11",
        "feature_phase": "FP-6",
        "version": "v0.2-base",
        "seed": seed,
        "device": str(dev),
        "steps": steps,
        "batch_size": batch_size,
        "block_size": block_size,
        "vocab_size": vocab_size,
        "corpus": {
            "lines": data["lines"],
            "docs": data["docs"],
            "blocks": data["blocks"],
            "train_blocks": data["train_blocks"],
            "val_blocks": data["val_blocks"],
        },
        "parameters": count_parameters(model),
        "train": {
            "first_loss": first,
            "last_loss": last,
            "min_loss": min(losses) if losses else None,
            "loss_dropped": first is not None and last is not None and last < first,
        },
        "val": {
            "ppl_start": ppl_start,
            "ppl_end": ppl_end,
            "ppl_improved": ppl_end.get("ppl", float("inf")) < ppl_start.get("ppl", float("inf")),
        },
        "curve": curve,
        "sample_completion": sample_text,
        "sample_ids_head": (sample_ids or [])[:40],
        "checkpoint": relative_to_repo(V02_CKPT),
        "tokenizer": relative_to_repo(V02_TOK),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    V02_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report"] = relative_to_repo(V02_REPORT)
    return report


def run_pretrain_s11(
    *,
    dry_run: bool = False,
    mode: str = "domain",  # domain | smoke | both
    steps: int = 200,
    batch_size: int = 8,
    block_size: int = 128,
    vocab_size: int = 4096,
    seed: int = 42,
    overfit_steps: int = 30,
    max_qa: int = 25_000,
) -> dict[str, Any]:
    """W-PRETRAIN entry for Sprint 11."""
    ensure_lake_layout()
    set_seed(seed)
    artifacts: list[str] = []

    cfg = MiniConfig(vocab_size=vocab_size, seed=seed, batch_size=batch_size, max_steps=steps)
    if dry_run:
        model = build_model(cfg, "cpu")
        counts = count_parameters(model)
        return {
            "ok": True,
            "dry_run": True,
            "sprint": "S11",
            "mode": mode,
            "parameters": counts,
            "config": cfg.to_dict(),
            "planned_steps": steps,
        }

    model = build_model(cfg, "cpu")
    param_report = write_param_count_report(model)
    artifacts.append(relative_to_repo(PARAM_COUNT_PATH))

    out: dict[str, Any] = {
        "ok": False,
        "sprint": "S11",
        "feature_phase": "FP-6",
        "mode": mode,
        "seed": seed,
        "config": cfg.to_dict(),
        "parameters": param_report.get("parameters"),
        "in_range": param_report.get("in_range"),
        "artifacts": artifacts,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if mode in ("smoke", "both"):
        smoke = run_overfit_smoke(
            steps=overfit_steps,
            batch_size=min(batch_size, 8),
            seq_len=min(block_size, 64),
            config=cfg,
            use_amp=False,
        )
        out["overfit_smoke"] = smoke

    if mode in ("domain", "both"):
        domain = train_domain(
            steps=steps,
            batch_size=batch_size,
            block_size=block_size,
            vocab_size=vocab_size,
            seed=seed,
            max_qa=max_qa,
            eval_every=max(10, steps // 4),
        )
        out["domain"] = domain
        for k in ("checkpoint", "tokenizer", "report"):
            if domain.get(k):
                artifacts.append(domain[k])
        out["ok"] = bool(domain.get("ok")) and bool(param_report.get("in_range"))
        if mode == "both":
            out["ok"] = out["ok"] and bool((out.get("overfit_smoke") or {}).get("loss_dropped"))
    else:
        out["ok"] = bool(param_report.get("in_range")) and bool((out.get("overfit_smoke") or {}).get("loss_dropped"))

    out["artifacts"] = list(dict.fromkeys(artifacts))
    PRETRAIN_LATEST.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    out["artifacts"] = list(dict.fromkeys(artifacts + [relative_to_repo(PRETRAIN_LATEST)]))
    return out


def verify_seed_repro(steps: int = 20, seed: int = 42) -> dict[str, Any]:
    """Two short runs with same seed should match last train loss closely."""
    a = train_domain(steps=steps, batch_size=4, block_size=64, seed=seed, max_qa=5000, eval_every=steps)
    b = train_domain(steps=steps, batch_size=4, block_size=64, seed=seed, max_qa=5000, eval_every=steps)
    la = (a.get("train") or {}).get("last_loss")
    lb = (b.get("train") or {}).get("last_loss")
    match = la is not None and lb is not None and abs(la - lb) < 1e-4
    return {"ok": match, "loss_a": la, "loss_b": lb, "seed": seed, "steps": steps}
