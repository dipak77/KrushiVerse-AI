"""Trainer worker for Pretrain & 3-stage SFT of MiniLMPro v3-18M Pro."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from mini.model import MiniLMPro, ProConfig
from mini.workers.tokenizer import KrushiTokenizer


class TextDataset(Dataset):
    def __init__(self, path: str, block_size: int, tok: KrushiTokenizer):
        self.block_size = block_size
        self.tokens = []
        p = Path(path)
        if p.exists():
            if p.suffix == ".bin":
                data = p.read_bytes()
                try:
                    text = data.decode("utf-8")
                    self.tokens = tok.encode(text)
                except Exception:
                    import array
                    a = array.array("H")
                    a.frombytes(data)
                    self.tokens = list(a)
            else:
                text = p.read_text(encoding="utf-8", errors="ignore")
                self.tokens = tok.encode(text)
        else:
            self.tokens = list(range(100, 100 + block_size * 50))

    def __len__(self):
        return max(1, len(self.tokens) // self.block_size)

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.tokens[start : start + self.block_size + 1]
        if len(chunk) < self.block_size + 1:
            chunk = chunk + [0] * (self.block_size + 1 - len(chunk))
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class SFTDataset(Dataset):
    def __init__(self, path: str, tok: KrushiTokenizer, block_size: int):
        self.samples = []
        self.block_size = block_size
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            self.samples.append(json.loads(line))
                        except Exception:
                            pass
        else:
            self.samples = [
                {"instruction": "सोयाबीनला खत द्या", "response": "सोयाबीनला युरिया व एसएसपी द्या."}
            ]
        self.tok = tok

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = item.get("instruction") or item.get("prompt") or ""
        resp = item.get("response") or item.get("answer") or ""

        p_ids = self.tok.encode(prompt, add_bos=True)
        r_ids = self.tok.encode(resp, add_eos=True)

        full_ids = (p_ids + r_ids)[: self.block_size + 1]
        if len(full_ids) < self.block_size + 1:
            full_ids = full_ids + [0] * (self.block_size + 1 - len(full_ids))

        x = torch.tensor(full_ids[:-1], dtype=torch.long)
        y = torch.tensor(full_ids[1:], dtype=torch.long)

        mask = torch.zeros(self.block_size, dtype=torch.float32)
        p_len = min(len(p_ids), self.block_size)
        resp_len = min(len(r_ids), self.block_size - p_len)
        mask[p_len : p_len + resp_len] = 1.0

        return x, y, mask


class Trainer:
    def __init__(self, config_path: str = "configs/config_v3_18M_pro.json"):
        self.cfg = ProConfig.from_json(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MiniLMPro(self.cfg).to(self.device)
        self.tok = KrushiTokenizer()

        if self.cfg.dropout > 0:
            self.model.enable_grad_checkpoint()

    def _save_ckpt(self, path: str, extra: dict | None = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        d = {"model": self.model.state_dict(), "cfg": self.cfg.__dict__}
        if extra:
            d.update(extra)
        torch.save(d, path)

    def _load_ckpt(self, path: str) -> int:
        if not os.path.exists(path):
            return 0
        d = torch.load(path, map_location=self.device)
        self.model.load_state_dict(d["model"], strict=False)
        return d.get("step", 0)

    def pretrain(
        self,
        data_path: str = "data/lake_balanced_50MB.bin",
        steps: int = 15000,
        batch: int = 8,
        accum: int = 4,
        lr: float = 3e-4,
        wd: float = 0.01,
        warmup: int = 1000,
        save_every: int = 500,
        eval_every: int = 500,
        ckpt: str = "artifacts/pretrain_18M.pt",
        resume: bool = True,
    ):
        ds = TextDataset(data_path, self.cfg.block_size, self.tok)
        dl = DataLoader(
            ds, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
        )
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: min(1.0, s / max(1, warmup))
            * (0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))),
        )
        scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

        start = 0
        if resume and os.path.exists(ckpt):
            start = self._load_ckpt(ckpt)
            print(f"↻ resume from step {start}")

        if self.cfg.use_torch_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"(torch.compile skipped: {e})")

        self.model.train()
        it = iter(dl)
        t0 = time.time()

        for step in range(start, steps):
            accum_loss = 0.0
            for _ in range(accum):
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(dl)
                    x, y = next(it)
                x, y = x.to(self.device), y.to(self.device)

                with autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                    logits, loss = self.model(x, y)
                    loss = loss / accum

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += loss.item()

            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

            sched.step()
            opt.zero_grad(set_to_none=True)

            if step % 50 == 0:
                lr_now = sched.get_last_lr()[0]
                tps = (step - start + 1) * batch * accum / max(time.time() - t0, 1)
                print(
                    f"step {step:5d} | loss {accum_loss:6.3f} | lr {lr_now:.2e} | tps {tps:7.0f}"
                )

            if (step + 1) % save_every == 0:
                self._save_ckpt(ckpt, {"step": step + 1})
                print(f"  ✓ saved → {ckpt}")

        self._save_ckpt(ckpt, {"step": steps})

    def sft(
        self,
        jsonl_path: str,
        steps: int,
        stage: str,
        ckpt: str,
        lr: float = 1e-4,
        batch: int = 8,
        accum: int = 4,
        resume: bool = True,
    ):
        ds = SFTDataset(jsonl_path, self.tok, self.cfg.block_size)
        dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)

        if resume and os.path.exists(ckpt):
            self._load_ckpt(ckpt)

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

        it = iter(dl)
        t0 = time.time()

        for step in range(steps):
            accum_loss = 0.0
            for _ in range(accum):
                try:
                    x, y, mask = next(it)
                except StopIteration:
                    it = iter(dl)
                    x, y, mask = next(it)
                x, y, mask = (
                    x.to(self.device),
                    y.to(self.device),
                    mask.to(self.device).bool(),
                )

                with autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                    logits, _ = self.model(x)
                    logits_flat = logits[mask]
                    y_flat = y[mask]
                    if logits_flat.numel() > 0:
                        loss = F.cross_entropy(logits_flat, y_flat) / accum
                    else:
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)), y.view(-1)
                        ) / accum

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += loss.item()

            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

            opt.zero_grad(set_to_none=True)

            if step % 50 == 0:
                print(
                    f"[{stage}] step {step:4d} | loss {accum_loss:6.3f} | {(step+1)*batch*accum/max(time.time()-t0, 1):.0f} tps"
                )

            if (step + 1) % 500 == 0:
                self._save_ckpt(ckpt, {"stage": stage, "step": step + 1})

        self._save_ckpt(ckpt, {"stage": stage, "step": steps})


if __name__ == "__main__":
    t = Trainer("configs/config_v3_18M_pro.json")
    t.pretrain("data/lake_balanced_50MB.bin")
    t.sft("data/sft_stage1_instruct.json", 3000, "s1", "artifacts/sft_s1.pt")
    t.sft("data/sft_stage2_rag.json", 3000, "s2", "artifacts/sft_s2.pt")
    t.sft("data/sft_stage3_reason.json", 3000, "s3", "artifacts/sft_s3_final.pt")
