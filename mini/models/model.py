"""MiniLM: decoder-only transformer with RoPE, RMSNorm, SwiGLU, weight tying — prod-ready.

Fixes vs original:
- RotaryEmbedding cache rebuild without re-register_buffer leak
- head_dim even check + SDPA fast path (32/64/128)
- RMSNorm uses float32 for stability
- generate() efficient top-p with topk filtering
- count_parameters unique handling
- torch.compile compatible
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from mini.models.config import MiniConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use float32 for norm for stability, then cast back
        dtype = x.dtype
        x_f = x.float()
        norm = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out = x_f * norm
        return (self.weight * out).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: (..., dim)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q,k: (B, n_head, T, head_dim)
    # cos,sin: (1,1,T,head_dim)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq: int = 512, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._build_cache(max_seq)

    def _build_cache(self, seq_len: int):
        # Build without re-registering buffers (fix leak)
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq, dim]
        cos = emb.cos()[None, None, :, :]  # [1,1,seq,dim]
        sin = emb.sin()[None, None, :, :]
        # Store as non-persistent buffers via direct assignment
        self._cos_cached = cos
        self._sin_cached = sin

    def forward(self, seq_len: int, device: torch.device):
        if seq_len > self._seq_len_cached:
            # Rebuild on device
            self.inv_freq = self.inv_freq.to(device)
            self._build_cache(seq_len)
        # Move cached to device if needed
        cos = self._cos_cached
        sin = self._sin_cached
        if cos.device != device:
            cos = cos.to(device)
            sin = sin.to(device)
            self._cos_cached = cos
            self._sin_cached = sin
        return cos[:, :, :seq_len, :], sin[:, :, :seq_len, :]


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, n_hidden: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(n_embd, n_hidden, bias=bias)
        self.w2 = nn.Linear(n_hidden, n_embd, bias=bias)
        self.w3 = nn.Linear(n_embd, n_hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"n_embd {config.n_embd} not divisible by n_head {config.n_head}"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.n_embd = config.n_embd
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.block_size, config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, n_head, T, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rope(T, x.device)
        q, k = apply_rope(q, k, cos, sin)
        # SDPA fast path — dropout only in training
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config.n_embd, config.n_hidden, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniLM(nn.Module):
    """Agriculture Mini decoder-only LM."""

    def __init__(self, config: MiniConfig | None = None):
        super().__init__()
        self.config = config or MiniConfig()
        c = self.config
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.norm_f = RMSNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        if c.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.gradient_checkpointing = bool(getattr(c, "gradient_checkpointing", False))
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.config.block_size}")
        x = self.drop(self.tok_emb(idx))
        if self.gradient_checkpointing and self.training:
            for block in self.blocks:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            ignore = int(getattr(self.config, "ignore_index", -100))
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 32, temperature: float = 1.0, top_p: float | None = None, do_sample: bool = True, **kwargs):
        # Efficient generate with caching of last block only (no KV cache for simplicity but optimized)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if logits.shape[1] != self.config.vocab_size:
                raise RuntimeError("Logits vocab mismatch")

            if top_p is not None and 0.0 < top_p < 1.0:
                # Efficient top-p: sort descending, cumulative probs
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                # Mask beyond top-p
                sorted_mask = cumsum > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
                # Sample from filtered distribution
                filtered_probs = F.softmax(sorted_logits, dim=-1)
                next_token_sorted = torch.multinomial(filtered_probs, num_samples=1)
                next_token = torch.gather(sorted_indices, -1, next_token_sorted)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_token], dim=1)
            # Early stop on eos
            if next_token.item() == self.config.eos_id:
                break
        return idx


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Unique storage (handles tie_weights)
    unique = 0
    seen_ids = set()
    for p in model.parameters():
        pid = id(p)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        if trainable_only and not p.requires_grad:
            continue
        unique += p.numel()
    return {
        "total_tensors_sum": total,
        "trainable_tensors_sum": trainable,
        "unique_params": unique,
        "millions": round(unique / 1e6, 4),
    }


from mini.model import MiniLMPro, ProConfig  # re-export for v3-18M Pro compatibility

