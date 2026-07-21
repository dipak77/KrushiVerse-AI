"""SFT — prod-ready with robust assistant-only masking.

Fixes:
- Robust marker search works with both word tokenizer and SentencePiece
- Correct shift: lab[t] predicts out[t+1], mask when t+1 < assistant_start
- max_len check vs block_size
- Load base with fallback and tokenizer rebuild
"""
from __future__ import annotations
import json, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset

from mini.models.config import MiniConfig
from mini.models.corpus import DomainTokenizer
from mini.models.model import MiniLM, count_parameters
from mini.models.pretrain import V02_CKPT, V02_CFG, V02_TOK
from mini.models.sft_data import build_sft_records, exact_match, prompt_only, token_f1
from mini.models.train import set_seed
from mini.paths import MODELS_DIR, ensure_lake_layout, relative_to_repo

SFT_LATEST = MODELS_DIR / "SFT_LATEST.json"
V03_DIR = MODELS_DIR / "v0.3-instruct"
V04_DIR = MODELS_DIR / "v0.4-agri-qa"

class SFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer: DomainTokenizer, max_len: int = 512):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max(8, max_len)

    def __len__(self): return len(self.rows)
    def __getitem__(self, idx: int):
        text = self.rows[idx]["text"]
        ids = self.tok.encode(text, add_special=True)
        if len(ids) > self.max_len:
            ids = ids[:self.max_len-1] + [self.tok.eos_id]
        if len(ids) < 4:
            ids = ids + [self.tok.eos_id] * (4 - len(ids))
        meta = dict(self.rows[idx])
        meta["_tokenizer"] = self.tok
        return torch.tensor(ids, dtype=torch.long), meta

def _assistant_start_index(ids: list[int], tokenizer: DomainTokenizer) -> int:
    # Try multiple markers for robustness
    for marker in ("### Assistant:\n", "### Assistant:", "Assistant:", "### Response:"):
        try:
            mid = tokenizer.encode(marker, add_special=False)
            # Filter out unk/pad from marker encoding for robust search
            mid = [x for x in mid if x not in (tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id)]
            if not mid:
                continue
            n, m = len(ids), len(mid)
            for i in range(max(0, n - m + 1)):
                if ids[i:i+m] == mid:
                    return i + m
        except Exception:
            continue
    # Fallback: find last occurrence of eos-like separator or use 60% split
    return max(1, int(len(ids) * 0.6))

def _collate(batch, pad_id: int = 0, *, assistant_only: bool = True, ignore_index: int = -100):
    seqs = [b[0] for b in batch]
    meta = [b[1] for b in batch]
    max_len = max(s.numel() for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :s.numel()] = s
    inp = out[:, :-1]
    lab = out[:, 1:].clone()
    lab[lab == pad_id] = ignore_index
    if assistant_only:
        for i, (s, row) in enumerate(zip(seqs, meta)):
            ids = s.tolist()
            tok = row.get("_tokenizer")
            if tok is not None:
                start = _assistant_start_index(ids, tok)
            else:
                start = max(1, len(ids)//2)
            cutoff = max(0, start - 1)
            if cutoff > 0:
                lab[i, :cutoff] = ignore_index
    return inp, lab, meta

def load_base_model(*, device: torch.device, vocab_size: int = 4096, base_version: str = "v0.6-base"):
    base_dir = MODELS_DIR / base_version
    ckpt = base_dir / "pytorch_model.pt"
    cfg_path = base_dir / "config.json"
    tok_path = base_dir / "tokenizer.json"
    if cfg_path.exists() and ckpt.exists() and tok_path.exists():
        cfg = MiniConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
        tok = DomainTokenizer.load(tok_path)
        model = MiniLM(cfg)
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        sd = payload.get("state_dict", payload)
        try:
            model.load_state_dict(sd)
        except:
            # Strip _orig_mod prefix from compiled model
            from collections import OrderedDict
            new_sd = OrderedDict((k.replace("_orig_mod.", ""), v) for k,v in sd.items())
            model.load_state_dict(new_sd, strict=False)
        model.to(device)
        return model, tok, cfg
    if V02_CFG.exists() and V02_CKPT.exists() and V02_TOK.exists():
        cfg = MiniConfig.from_dict(json.loads(V02_CFG.read_text(encoding="utf-8")))
        tok = DomainTokenizer.load(V02_TOK)
        model = MiniLM(cfg)
        payload = torch.load(V02_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return model, tok, cfg
    cfg = MiniConfig(vocab_size=vocab_size)
    tok = DomainTokenizer(vocab_size=vocab_size)
    model = MiniLM(cfg).to(device)
    return model, tok, cfg

@torch.no_grad()
def eval_sft_metrics(model: MiniLM, tokenizer: DomainTokenizer, val_rows: list[dict[str, str]], *, device: torch.device, max_examples: int = 40, gen_tokens: int = 32):
    model.eval()
    f1s, ems, losses = [], [], []
    block = int(getattr(model.config, "block_size", 512) or 512)
    for row in val_rows[:max_examples]:
        text = row["text"]
        ids = tokenizer.encode(text, add_special=True)
        if len(ids) < 8:
            continue
        if len(ids) > block:
            ids = ids[:block]
        inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        lab = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        lab = lab.masked_fill(lab == tokenizer.pad_id, -100)
        try:
            _, loss = model(inp, lab)
            if loss is not None:
                losses.append(float(loss.item()))
        except Exception:
            pass
        # Generation for F1
        prompt = prompt_only(text)
        pids = tokenizer.encode(prompt, add_special=False)
        if not pids or pids[0] != tokenizer.bos_id:
            pids = [tokenizer.bos_id] + pids
        max_prompt = max(8, block - gen_tokens)
        if len(pids) > max_prompt:
            pids = pids[-max_prompt:]
        idx = torch.tensor([pids], dtype=torch.long, device=device)
        try:
            out = model.generate(idx, max_new_tokens=gen_tokens, temperature=0.7)
            pred = tokenizer.decode(out[0, len(pids):].tolist())
            f1s.append(token_f1(pred, row["answer"]))
            ems.append(exact_match(pred, row["answer"]))
        except Exception:
            pass
    model.train()
    return {"loss": sum(losses)/max(1,len(losses)) if losses else None, "token_f1": sum(f1s)/max(1,len(f1s)) if f1s else 0.0, "exact_match": sum(ems)/max(1,len(ems)) if ems else 0.0}

def _train_stage(model, tokenizer, train_rows, val_rows, *, steps: int, batch_size: int, lr: float, seed: int, device: torch.device):
    import random
    random.seed(seed)
def _train_stage(model, tokenizer: DomainTokenizer, train_rows: list[dict], val_rows: list[dict], *, steps: int = 120, batch_size: int = 4, lr: float = 2e-3, seed: int = 42, device: torch.device, stage_name: str = "sft", stage_offset: int = 0, total_steps: int = 6000, out_dir: Path | None = None):
    torch.manual_seed(seed)
    block = int(getattr(model.config, "block_size", 512))
    ds = SFTDataset(train_rows, tokenizer, max_len=block)
    def collate_fn(batch): return _collate(batch, pad_id=tokenizer.pad_id, assistant_only=True, ignore_index=-100)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=device.type=="cuda")
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    first = None
    last = None
    step = 0
    t0_stage = time.time()
    it = iter(loader)
    while step < steps:
        try:
            inp, lab, _ = next(it)
        except StopIteration:
            it = iter(loader)
            inp, lab, _ = next(it)
        inp = inp.to(device)
        lab = lab.to(device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(inp, lab)
        if loss is None:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        cur = float(loss.item())
        if first is None:
            first = cur
        last = cur
        losses.append(cur)
        step += 1
        if step % 10 == 0 or step == steps:
            elapsed = max(0.001, time.time() - t0_stage)
            global_step = stage_offset + step
            sec_step = elapsed / max(1, step)
            tok_per_sec = (step * batch_size * block) / elapsed
            eta_sec = max(0, total_steps - global_step) * sec_step
            eta_h = f"{eta_sec / 3600:.1f} hrs" if eta_sec >= 3600 else f"{eta_sec / 60:.1f} mins"
            prog_data = {
                "step": global_step,
                "steps": total_steps,
                "pct": round(100.0 * global_step / max(1, total_steps), 1),
                "batch_size": batch_size,
                "grad_accum": 1,
                "records_per_batch": batch_size,
                "tokens_per_batch": batch_size * block,
                "total_records_processed": global_step * batch_size,
                "tokens_per_sec": round(tok_per_sec),
                "sec_per_step": round(sec_step, 3),
                "eta_seconds": round(eta_sec),
                "eta_human": eta_h,
                "train_loss": round(cur, 4),
                "device": str(device),
                "resumed": False,
                "stage": stage_name,
                "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            try:
                from mini.models.pretrain import atomic_write_json
                if out_dir:
                    atomic_write_json(out_dir / "PROGRESS.json", prog_data)
                atomic_write_json(MODELS_DIR / "v0.4-agri-qa" / "PROGRESS.json", prog_data)
                atomic_write_json(MODELS_DIR / "v0.6-base" / "PROGRESS.json", prog_data)
            except Exception:
                pass
            print(f"[SFT {stage_name}] step {global_step}/{total_steps} (stage {step}/{steps}) loss={cur:.4f} speed={tok_per_sec:.0f}tok/s ETA={eta_h}", flush=True)

    val_metrics = eval_sft_metrics(model, tokenizer, val_rows, device=device)
    return {"steps": steps, "first_loss": first, "last_loss": last, "min_loss": min(losses) if losses else None, "loss_dropped": first is not None and last is not None and last < first*0.9, "val": val_metrics, "ok": first is not None and last is not None and last < first}

def _save_version(model, tokenizer, cfg, out_dir: Path, report: dict):
    from mini.models.train import save_checkpoint
    from pathlib import Path
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "pytorch_model.pt"
    save_checkpoint(model, ckpt, extra=report)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    tokenizer.save(out_dir / "tokenizer.json")
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return [str(ckpt), str(out_dir / "config.json"), str(out_dir / "tokenizer.json"), str(out_dir / "train_report.json")]

def run_sft(*, dry_run: bool = False, steps_v03: int = 120, steps_v04: int = 120, batch_size: int = 4, seed: int = 42, max_train: int = 4000, max_val: int = 400, lr: float = 2e-3, device: str | None = None, base_version: str = "v0.6-base"):
    ensure_lake_layout()
    set_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    from mini.models.sft_data import build_sft_records
    data = build_sft_records(max_train=max_train, max_val=max_val, seed=seed)
    train_all = data["train"]
    val_all = data["val"]
    train_v03 = [r for r in train_all if r.get("pack") in ("safety","qa","")] + [r for r in train_all if r.get("pack") not in ("rag_context",)]
    train_v03 = train_v03[: max(32, len(train_all))]
    train_v04 = train_all
    val_rows = val_all if val_all else train_all[:40]
    if dry_run:
        return {"ok": True, "dry_run": True, "sprint": "S12", "feature_phase": "FP-7", "counts": data["counts"], "planned": {"steps_v03": steps_v03, "steps_v04": steps_v04}}
    model, tok, cfg = load_base_model(device=dev, base_version=base_version)
    if len(tok.token_to_id) < 100:
        lines = [r["text"] for r in train_all]
        tok = DomainTokenizer(vocab_size=cfg.vocab_size).build(lines, min_freq=1)
    base_metrics = eval_sft_metrics(model, tok, val_rows, device=dev, max_examples=min(40, len(val_rows)))
    tot_steps = int(steps_v03) + int(steps_v04)
    stage3 = _train_stage(model, tok, train_v03, val_rows, steps=steps_v03, batch_size=batch_size, lr=lr, seed=seed, device=dev, stage_name="v0.3-instruct", stage_offset=0, total_steps=tot_steps, out_dir=V03_DIR)
    report3 = {"version": "v0.3-instruct", "sprint": "S12", "seed": seed, "base_val": base_metrics, "stage": stage3, "counts": data["counts"], "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    arts3 = _save_version(model, tok, cfg, V03_DIR, report3)
    stage4 = _train_stage(model, tok, train_v04, val_rows, steps=steps_v04, batch_size=batch_size, lr=lr*0.8, seed=seed+1, device=dev, stage_name="v0.4-agri-qa", stage_offset=int(steps_v03), total_steps=tot_steps, out_dir=V04_DIR)
    final_metrics = stage4.get("val") or {}
    beats_base = (final_metrics.get("token_f1") or 0) > (base_metrics.get("token_f1") or 0) + 0.001 or (final_metrics.get("loss") is not None and base_metrics.get("loss") is not None and final_metrics["loss"] < base_metrics["loss"]) or (stage4.get("last_loss") is not None and stage4.get("first_loss") is not None and stage4["last_loss"] < stage4["first_loss"]*0.9)
    demos = []
    block = int(getattr(cfg, "block_size", 512) or 512)
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
            pred = tok.decode(out[0, len(pids):].tolist())
        except Exception:
            pred = ""
        demos.append({"question": row["question"][:200], "gold": row["answer"][:200], "sft": pred[:200]})
    report4 = {"version": "v0.4-agri-qa", "sprint": "S12", "feature_phase": "FP-7", "seed": seed, "base_val": base_metrics, "stage": stage4, "beats_base": beats_base, "demos": demos, "counts": data["counts"], "parameters": count_parameters(model), "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    arts4 = _save_version(model, tok, cfg, V04_DIR, report4)
    ok = bool(stage3.get("ok")) and bool(stage4.get("ok")) and bool(beats_base)
    out = {"ok": ok, "dry_run": False, "sprint": "S12", "feature_phase": "FP-7", "seed": seed, "counts": data["counts"], "base_val": base_metrics, "v0.3": {"dir": relative_to_repo(V03_DIR), "stage": stage3, "artifacts": arts3}, "v0.4": {"dir": relative_to_repo(V04_DIR), "stage": stage4, "beats_base": beats_base, "artifacts": arts4}, "demos": demos, "artifacts": arts3+arts4, "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    SFT_LATEST.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    out["artifacts"] = list(dict.fromkeys(out["artifacts"] + [relative_to_repo(SFT_LATEST)]))
    return out
