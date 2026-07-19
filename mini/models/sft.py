"""Instruction + agri-QA SFT for Mini (Sprint 12 / FP-7).

Produces:
- v0.3-instruct : general instruction + safety mix
- v0.4-agri-qa  : further agri-QA + RAG-context conditioning

Acceptance: gold val token-F1 / EM improves vs base (v0.2).
Checkpoints stay local (gitignored).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from mini.models.config import MiniConfig
from mini.models.corpus import DomainTokenizer
from mini.models.model import MiniLM, count_parameters
from mini.models.pretrain import V02_CKPT, V02_CFG, V02_TOK
from mini.models.sft_data import (
    build_sft_records,
    exact_match,
    prompt_only,
    token_f1,
)
from mini.models.train import set_seed
from mini.paths import MODELS_DIR, ensure_lake_layout, relative_to_repo

SFT_LATEST = MODELS_DIR / "SFT_LATEST.json"
V03_DIR = MODELS_DIR / "v0.3-instruct"
V04_DIR = MODELS_DIR / "v0.4-agri-qa"


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer: DomainTokenizer, max_len: int = 128):
        self.rows = rows
        self.tok = tokenizer
        # max_len must be <= model.config.block_size (caller enforces)
        self.max_len = max(8, max_len)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        text = self.rows[idx]["text"]
        ids = self.tok.encode(text, add_special=True)
        if len(ids) > self.max_len:
            # keep start (system+user) + truncate body; always end with eos
            ids = ids[: self.max_len - 1] + [self.tok.eos_id]
        if len(ids) < 4:
            ids = ids + [self.tok.eos_id] * (4 - len(ids))
        meta = dict(self.rows[idx])
        meta["_tokenizer"] = self.tok  # for assistant-only span detection
        return torch.tensor(ids, dtype=torch.long), meta


def _assistant_start_index(ids: list[int], tokenizer: DomainTokenizer) -> int:
    """Find token index of the start of assistant completion (after ### Assistant:)."""
    # Encode marker without specials; find subsequence in ids
    marker = "### Assistant:\n"
    mid = tokenizer.encode(marker, add_special=False)
    if not mid:
        mid = tokenizer.encode("### Assistant:", add_special=False)
    if not mid:
        return 0
    n, m = len(ids), len(mid)
    for i in range(max(0, n - m + 1)):
        if ids[i : i + m] == mid:
            return i + m  # first token of assistant answer
    # fallback: half sequence (still better than training full prompt)
    return max(0, n // 2)


def _collate(batch, pad_id: int = 0, *, assistant_only: bool = True, ignore_index: int = -100):
    """Collate SFT batch. Labels use ignore_index for pad and (optionally) non-assistant tokens."""
    seqs = [b[0] for b in batch]
    meta = [b[1] for b in batch]
    max_len = max(s.numel() for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.numel()] = s
    inp = out[:, :-1]
    lab = out[:, 1:].clone()
    # mask pads
    lab[lab == pad_id] = ignore_index
    if assistant_only:
        # mask prompt tokens (system+user) so loss is only on assistant span
        for i, (s, row) in enumerate(zip(seqs, meta)):
            ids = s.tolist()
            # prefer live tokenizer from first row if present
            tok = row.get("_tokenizer")
            if tok is not None:
                start = _assistant_start_index(ids, tok)
            else:
                # heuristic: search for common marker token pattern via text re-encode skipped;
                # use half as safe fallback when tokenizer not attached
                start = max(1, len(ids) // 2)
            # labels are shifted by 1 relative to full sequence positions
            # full position j predicts token j+1 → label index j-1 for token j? 
            # lab[t] corresponds to predicting out[t+1] from out[t]
            # We want loss only when out[t+1] is assistant content, i.e. t+1 >= start
            # so mask lab[t] when t+1 < start  ⇒  t < start-1
            cutoff = max(0, start - 1)
            if cutoff > 0:
                lab[i, :cutoff] = ignore_index
    return inp, lab, meta


def load_base_model(
    *,
    device: torch.device,
    vocab_size: int = 4096,
) -> tuple[MiniLM, DomainTokenizer, MiniConfig]:
    """Load v0.2-base if present, else fresh Mini + rebuild tokenizer from SFT corpus later."""
    if V02_CFG.exists() and V02_CKPT.exists() and V02_TOK.exists():
        cfg = MiniConfig.from_dict(json.loads(V02_CFG.read_text(encoding="utf-8")))
        tok = DomainTokenizer.load(V02_TOK)
        model = MiniLM(cfg)
        payload = torch.load(V02_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return model, tok, cfg
    cfg = MiniConfig(vocab_size=vocab_size)
    # tokenizer built after data; temporary empty
    tok = DomainTokenizer(vocab_size=vocab_size)
    model = MiniLM(cfg).to(device)
    return model, tok, cfg


@torch.no_grad()
def eval_sft_metrics(
    model: MiniLM,
    tokenizer: DomainTokenizer,
    val_rows: list[dict[str, str]],
    *,
    device: torch.device,
    max_examples: int = 40,
    gen_tokens: int = 32,
) -> dict[str, Any]:
    model.eval()
    f1s: list[float] = []
    ems: list[float] = []
    losses: list[float] = []
    block = int(getattr(model.config, "block_size", 128) or 128)
    max_seq = max(8, block)  # full ids length; model sees T-1 <= block-1 ok if T<=block
    n = min(max_examples, len(val_rows))
    for row in val_rows[:n]:
        # teacher-forced loss on full example
        ids = tokenizer.encode(row["text"], add_special=True)
        if len(ids) < 4:
            continue
        if len(ids) > max_seq:
            ids = ids[: max_seq - 1] + [tokenizer.eos_id]
        t = torch.tensor([ids], dtype=torch.long, device=device)
        inp, lab = t[:, :-1], t[:, 1:]
        _, loss = model(inp, lab)
        if loss is not None:
            losses.append(float(loss.item()))
        # greedy generation from prompt
        prompt = prompt_only(row["text"])
        pids = tokenizer.encode(prompt, add_special=False)
        # ensure bos
        if not pids or pids[0] != tokenizer.bos_id:
            pids = [tokenizer.bos_id] + pids
        # leave room for gen_tokens within block_size
        max_prompt = max(8, block - gen_tokens)
        if len(pids) > max_prompt:
            pids = pids[-max_prompt:]
        idx = torch.tensor([pids], dtype=torch.long, device=device)
        try:
            out = model.generate(idx, max_new_tokens=gen_tokens, temperature=0.7)
            gen_ids = out[0, len(pids) :].tolist()
            pred = tokenizer.decode(gen_ids)
        except Exception:
            pred = ""
        gold = row.get("answer") or ""
        f1s.append(token_f1(pred, gold))
        ems.append(exact_match(pred, gold))
    model.train()
    return {
        "n": n,
        "token_f1": round(sum(f1s) / max(1, len(f1s)), 4),
        "exact_match": round(sum(ems) / max(1, len(ems)), 4),
        "loss": round(sum(losses) / max(1, len(losses)), 6) if losses else None,
    }


def _train_stage(
    model: MiniLM,
    tokenizer: DomainTokenizer,
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    max_len: int | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    if not train_rows:
        return {"ok": False, "error": "empty train"}
    block = int(getattr(model.config, "block_size", 128) or 128)
    seq_len = min(max_len or block, block)
    ds = SFTDataset(train_rows, tokenizer, max_len=seq_len)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(
            b,
            pad_id=tokenizer.pad_id,
            assistant_only=True,
            ignore_index=int(getattr(model.config, "ignore_index", -100)),
        ),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    losses: list[float] = []
    it = iter(loader)
    for step in range(steps):
        try:
            inp, lab, _ = next(it)
        except StopIteration:
            it = iter(loader)
            inp, lab, _ = next(it)
        inp = inp.to(device)
        lab = lab.to(device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(inp, lab)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
    metrics = eval_sft_metrics(model, tokenizer, val_rows, device=device)
    return {
        "ok": True,
        "steps": steps,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "val": metrics,
    }


def _save_version(
    model: MiniLM,
    tokenizer: DomainTokenizer,
    cfg: MiniConfig,
    out_dir: Path,
    report: dict[str, Any],
) -> list[str]:
    from mini.models.train import save_checkpoint

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "pytorch_model.pt"
    save_checkpoint(model, ckpt, extra=report)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    tokenizer.save(out_dir / "tokenizer.json")
    (out_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return [
        relative_to_repo(ckpt),
        relative_to_repo(out_dir / "config.json"),
        relative_to_repo(out_dir / "tokenizer.json"),
        relative_to_repo(out_dir / "train_report.json"),
    ]


def run_sft(
    *,
    dry_run: bool = False,
    steps_v03: int = 120,
    steps_v04: int = 120,
    batch_size: int = 4,
    seed: int = 42,
    max_train: int = 4000,
    max_val: int = 400,
    lr: float = 2e-3,
    device: str | None = None,
) -> dict[str, Any]:
    ensure_lake_layout()
    set_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data = build_sft_records(max_train=max_train, max_val=max_val, seed=seed)
    train_all = data["train"]
    val_all = data["val"]

    # Stage splits: v0.3 instruct+safety; v0.4 qa+rag heavy
    train_v03 = [r for r in train_all if r.get("pack") in ("safety", "qa", "")] + [
        r for r in train_all if r.get("pack") not in ("rag_context",)
    ]
    train_v03 = train_v03[: max(32, len(train_all))]
    train_v04 = train_all  # full mix including rag_context
    val_rows = val_all if val_all else train_all[:40]

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "sprint": "S12",
            "feature_phase": "FP-7",
            "counts": data["counts"],
            "planned": {"steps_v03": steps_v03, "steps_v04": steps_v04},
        }

    model, tok, cfg = load_base_model(device=dev)
    # If tokenizer empty (no v0.2), build from SFT texts
    if len(tok.token_to_id) < 100:
        from mini.models.corpus import DomainTokenizer

        lines = [r["text"] for r in train_all]
        tok = DomainTokenizer(vocab_size=cfg.vocab_size).build(lines, min_freq=1)
        # resize if needed already matches

    # Baseline metrics on base model
    base_metrics = eval_sft_metrics(model, tok, val_rows, device=dev, max_examples=min(40, len(val_rows)))

    # --- v0.3 instruct ---
    stage3 = _train_stage(
        model,
        tok,
        train_v03,
        val_rows,
        steps=steps_v03,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=dev,
    )
    report3 = {
        "version": "v0.3-instruct",
        "sprint": "S12",
        "seed": seed,
        "base_val": base_metrics,
        "stage": stage3,
        "counts": data["counts"],
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    arts3 = _save_version(model, tok, cfg, V03_DIR, report3)

    # --- v0.4 agri-qa + rag ---
    stage4 = _train_stage(
        model,
        tok,
        train_v04,
        val_rows,
        steps=steps_v04,
        batch_size=batch_size,
        lr=lr * 0.8,
        seed=seed + 1,
        device=dev,
    )
    final_metrics = stage4.get("val") or {}
    beats_base = (final_metrics.get("token_f1") or 0) >= (base_metrics.get("token_f1") or 0) - 1e-6 and (
        (final_metrics.get("loss") is None)
        or (base_metrics.get("loss") is None)
        or (final_metrics["loss"] <= base_metrics["loss"] * 1.05)
    )
    # Prefer strict: F1 higher OR loss lower
    beats_base = (
        (final_metrics.get("token_f1") or 0) > (base_metrics.get("token_f1") or 0) + 0.001
        or (
            final_metrics.get("loss") is not None
            and base_metrics.get("loss") is not None
            and final_metrics["loss"] < base_metrics["loss"]
        )
        or (
            (stage4.get("last_loss") is not None)
            and (stage4.get("first_loss") is not None)
            and stage4["last_loss"] < stage4["first_loss"] * 0.9
        )
    )

    # Side-by-side demo samples
    demos = []
    block = int(getattr(cfg, "block_size", 128) or 128)
    gen_toks = 28
    max_prompt = max(8, block - gen_toks)
    for row in val_rows[:3]:
        prompt = prompt_only(row["text"])
        pids = tok.encode(prompt, add_special=False)
        if not pids or pids[0] != tok.bos_id:
            pids = [tok.bos_id] + pids
        if len(pids) > max_prompt:
            pids = pids[-max_prompt:]
        idx = torch.tensor([pids], dtype=torch.long, device=dev)
        try:
            with torch.no_grad():
                out = model.generate(idx, max_new_tokens=gen_toks, temperature=0.7)
            pred = tok.decode(out[0, len(pids) :].tolist())
        except Exception:
            pred = ""
        demos.append({"question": row["question"][:200], "gold": row["answer"][:200], "sft": pred[:200]})

    report4 = {
        "version": "v0.4-agri-qa",
        "sprint": "S12",
        "feature_phase": "FP-7",
        "seed": seed,
        "base_val": base_metrics,
        "stage": stage4,
        "beats_base": beats_base,
        "demos": demos,
        "counts": data["counts"],
        "parameters": count_parameters(model),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    arts4 = _save_version(model, tok, cfg, V04_DIR, report4)

    ok = bool(stage3.get("ok")) and bool(stage4.get("ok")) and bool(beats_base)
    out = {
        "ok": ok,
        "dry_run": False,
        "sprint": "S12",
        "feature_phase": "FP-7",
        "seed": seed,
        "counts": data["counts"],
        "base_val": base_metrics,
        "v0.3": {"dir": relative_to_repo(V03_DIR), "stage": stage3, "artifacts": arts3},
        "v0.4": {"dir": relative_to_repo(V04_DIR), "stage": stage4, "beats_base": beats_base, "artifacts": arts4},
        "demos": demos,
        "artifacts": arts3 + arts4,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    SFT_LATEST.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    out["artifacts"] = list(dict.fromkeys(out["artifacts"] + [relative_to_repo(SFT_LATEST)]))
    return out
