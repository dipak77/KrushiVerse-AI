"""MiniLMPro: GQA + RoPE + RMSNorm + SwiGLU + QK-norm + Gradient Checkpointing.

Targeted for RTX 2050 4GB (18M Pro Premium architecture).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


@dataclass
class ProConfig:
    vocab_size: int = 8192
    n_embd: int = 320
    n_layer: int = 12
    n_head: int = 8
    n_kv_head: int = 4
    head_dim: int = 40
    n_hidden: int = 864
    block_size: int = 1024
    block_size_ext: int = 2048
    rope_scaling_factor: float = 2.0
    dropout: float = 0.02
    bias: bool = False
    weight_tying: bool = True
    rope_theta: float = 50000.0
    qk_norm: bool = True
    use_flash_attn: bool = True
    use_torch_compile: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> ProConfig:
        p = Path(path)
        if not p.exists():
            alt = Path("configs") / p.name
            if alt.exists():
                p = alt
            else:
                alt2 = Path("mini") / p.name
                if alt2.exists():
                    p = alt2
        data = json.loads(p.read_text(encoding="utf-8"))
        hp = data.get("hyperparameters", data)
        rope_sf = 2.0
        if isinstance(hp.get("rope_scaling"), dict):
            rope_sf = float(hp["rope_scaling"].get("factor", 2.0))
        elif "rope_scaling_factor" in hp:
            rope_sf = float(hp["rope_scaling_factor"])

        return cls(
            vocab_size=hp.get("vocab_size", 8192),
            n_embd=hp.get("n_embd", 320),
            n_layer=hp.get("n_layer", 12),
            n_head=hp.get("n_head", 8),
            n_kv_head=hp.get("n_kv_head", 4),
            head_dim=hp.get("head_dim", 40),
            n_hidden=hp.get("n_hidden", 864),
            block_size=hp.get("block_size", 1024),
            block_size_ext=hp.get("block_size_ext", 2048),
            rope_scaling_factor=rope_sf,
            dropout=hp.get("dropout", 0.02),
            bias=hp.get("bias", False),
            weight_tying=hp.get("weight_tying", True),
            rope_theta=hp.get("rope_theta", 50000.0),
            qk_norm=hp.get("qk_norm", True),
            use_flash_attn=hp.get("use_flash_attn", True),
            use_torch_compile=hp.get("use_torch_compile", True),
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        norm = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (self.weight * (x_f * norm)).to(dtype)


def precompute_rope(
    dim: int, max_seq: int, theta: float = 50000.0, scaling_factor: float = 2.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    if scaling_factor > 1.0:
        inv_freq = inv_freq / scaling_factor
    t = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, None, :, :]  # [1, 1, seq, dim]
    sin = emb.sin()[None, None, :, :]
    return cos, sin


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    seq_len = q.size(2)
    c = cos[:, :, :seq_len, :].to(q.device)
    s = sin[:, :, :seq_len, :].to(q.device)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_out = (q * c) + (rotate_half(q) * s)
    k_out = (k * c) + (rotate_half(k) * s)
    return q_out, k_out


class Attention(nn.Module):
    def __init__(self, cfg: ProConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_head // self.n_kv_head

        self.wq = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=cfg.bias)
        self.wk = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.wv = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.wo = nn.Linear(self.n_head * self.head_dim, cfg.n_embd, bias=cfg.bias)

        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.attn_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = apply_rope(q, k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if hasattr(F, "scaled_dot_product_attention") and self.cfg.use_flash_attn:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.cfg.dropout if self.training else 0.0
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.tril(torch.ones((T, T), device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ProConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.n_embd, cfg.n_hidden, bias=cfg.bias)
        self.w2 = nn.Linear(cfg.n_hidden, cfg.n_embd, bias=cfg.bias)
        self.w3 = nn.Linear(cfg.n_embd, cfg.n_hidden, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ProConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.n_embd)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.ffn(self.n2(x))
        return x


class MiniLMPro(nn.Module):
    def __init__(self, cfg: ProConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.weight_tying:
            self.head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(
            cfg.head_dim, cfg.block_size_ext, cfg.rope_theta, cfg.rope_scaling_factor
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def enable_grad_checkpoint(self):
        for b in self.blocks:
            b.__class__ = type("GCBlock", (Block,), {"forward": _gc_fwd})

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = self.cos[:, :, :T, :]
        sin = self.sin[:, :, :T, :]

        for blk in self.blocks:
            x = blk(x, cos, sin)

        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0
            )
        return logits, loss


def _gc_fwd(self, x, cos, sin):
    return torch.utils.checkpoint.checkpoint(
        Block.forward, self, x, cos, sin, use_reentrant=False
    )


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    cfg = ProConfig.from_json("configs/config_v3_18M_pro.json")
    m = MiniLMPro(cfg)
    print(f"Params: {count_params(m)/1e6:.2f}M")
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = m(x, x)
    print(f"Logits: {logits.shape}, Loss: {loss.item():.3f}")
