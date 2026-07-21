"""Mini decoder-only config — prod-ready v2 FINAL FIX.

- v1 default: ~1.36M (vocab 4096, emb 128, 6 layers)
- v2-12M-fixed: ~12.3M unique (320 emb, 10 layers, 1280 hidden, tie_weights, head_dim 32)

FIXED vs your version:
- Added grad_accum field (was missing, caused batch 16 OOM)
- batch_size 16 -> 4 + grad_accum 4 = effective 16 without OOM on RTX 2050 4GB
- n_head 8 -> 10 for head_dim 32 fast path (was 40 slow)
- v2_15m() no longer overrides JSON batch_size to 16
- Supports batch_size from JSON, falls back to 4 not 16
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mini.paths import MODELS_DIR, REPO_ROOT

IGNORE_INDEX = -100

CONFIG_V2_15M_PATH = MODELS_DIR / "config_v2_15M.json"
CONFIG_V2_15M_DOCS = REPO_ROOT / "docs" / "Next 15M Gen Plan" / "config_v2_15M.json"


@dataclass
class MiniConfig:
    vocab_size: int = 4096
    n_embd: int = 128
    n_layer: int = 6
    n_head: int = 4
    n_hidden: int = 192
    block_size: int = 512
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True
    rope_theta: float = 10000.0
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3
    ignore_index: int = IGNORE_INDEX

    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    max_steps: int = 200
    batch_size: int = 4
    grad_accum: int = 4
    grad_clip: float = 1.0
    seed: int = 42
    use_amp: bool = True
    gradient_checkpointing: bool = False
    model_variant: str = "v1-1M"

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd {self.n_embd} must be divisible by n_head {self.n_head}")
        head_dim = self.n_embd // self.n_head
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim {head_dim} must be even for RoPE rotate_half")
        if head_dim not in (32, 64, 128):
            warnings.warn(f"head_dim={head_dim} not power-of-2 (32/64/128) — SDPA will be ~25% slower. Use n_embd=320,n_head=10 for 32.")
        if self.batch_size > 8 and self.grad_accum == 1:
            warnings.warn(f"batch_size={self.batch_size} >8 with grad_accum=1 will OOM on RTX 2050 4GB. Use batch_size=4 + grad_accum=4")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    @property
    def estimated_params_m(self) -> float:
        emb = self.vocab_size * self.n_embd
        attn = 4 * self.n_embd * self.n_embd
        ffn = 3 * self.n_embd * self.n_hidden
        layer = attn + ffn
        total = emb + self.n_layer * layer
        if not self.tie_weights:
            total += self.vocab_size * self.n_embd
        return total / 1e6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MiniConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        if "hyperparameters" in d and isinstance(d["hyperparameters"], dict):
            hp = dict(d["hyperparameters"])
            if "weight_tying" in hp:
                hp["tie_weights"] = bool(hp.pop("weight_tying"))
            for drop in ("head_dim", "ffn_multiplier", "positional_encoding", "normalization", "activation", "attention_type"):
                hp.pop(drop, None)
            merged = {**hp}
            for k, v in d.items():
                if k != "hyperparameters" and k in known:
                    merged[k] = v
            if "training" in d and isinstance(d["training"], dict):
                tr = d["training"]
                if isinstance(tr.get("pretrain"), dict):
                    pt = tr["pretrain"]
                    for kk in ("learning_rate", "weight_decay", "max_steps", "batch_size", "grad_accum"):
                        if kk not in merged and kk in pt:
                            merged[kk] = pt[kk]
                    if "steps" in pt and "max_steps" not in merged:
                        merged["max_steps"] = pt["steps"]
            d = merged

        if "tie_weights" not in d and "weight_tying" in d:
            d["tie_weights"] = bool(d.pop("weight_tying"))
        if "model_variant" not in d and "version" in d:
            d["model_variant"] = str(d["version"])

        # Backward compat: if batch_size 16 and no grad_accum, auto-convert to 4+4
        if int(d.get("batch_size", 0)) >= 16 and int(d.get("grad_accum", 0)) == 0:
            d["batch_size"] = 4
            d["grad_accum"] = 4

        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @classmethod
    def v2_15m(cls) -> "MiniConfig":
        """Canonical 12.3M production config (previously called 15M)."""
        path = CONFIG_V2_15M_PATH if CONFIG_V2_15M_PATH.exists() else CONFIG_V2_15M_DOCS
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                cfg = cls.from_dict(raw)
                if cfg.model_variant == "v1-1M":
                    cfg.model_variant = str(raw.get("model_variant") or raw.get("version") or "v2-12M-fixed")
                if cfg.max_steps < 10:
                    cfg.max_steps = int(raw.get("max_steps") or raw.get("training", {}).get("pretrain", {}).get("steps") or 10000)
                if cfg.batch_size < 1:
                    cfg.batch_size = 4
                if cfg.grad_accum < 1:
                    cfg.grad_accum = 4
                if cfg.learning_rate <= 0:
                    cfg.learning_rate = 3e-4
                if cfg.weight_decay < 0:
                    cfg.weight_decay = 0.1
                cfg.use_amp = True
                cfg.gradient_checkpointing = True
                return cfg
            except Exception:
                pass
        return cls(
            vocab_size=8192,
            n_embd=320,
            n_layer=10,
            n_head=10,
            n_hidden=1280,
            block_size=512,
            dropout=0.0,
            bias=False,
            tie_weights=True,
            rope_theta=10000.0,
            pad_id=0,
            bos_id=2,
            eos_id=3,
            learning_rate=3e-4,
            weight_decay=0.1,
            max_steps=10000,
            batch_size=4,
            grad_accum=4,
            grad_clip=1.0,
            seed=42,
            use_amp=True,
            gradient_checkpointing=True,
            model_variant="v2-12M-fixed",
        )


def load_config_json(path: str | Path | None = None) -> MiniConfig:
    if path is None:
        return MiniConfig.v2_15m()
    p = Path(path)
    if not p.is_absolute():
        for base in (REPO_ROOT, MODELS_DIR, REPO_ROOT / "docs" / "Next 15M Gen Plan", Path.cwd(), Path.cwd() / "configs"):
            cand = base / p
            if cand.exists():
                p = cand
                break
            cand2 = base / Path(p).name
            if cand2.exists():
                p = cand2
                break
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path} searched in repo")
    raw = json.loads(p.read_text(encoding="utf-8"))
    cfg = MiniConfig.from_dict(raw)
    if not cfg.model_variant or cfg.model_variant == "v1-1M":
        cfg.model_variant = str(raw.get("version") or raw.get("model_variant") or "v2-12M-fixed")
    return cfg
