"""Mini decoder-only config (~1M parameters target)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MiniConfig:
    """emb 128, layers 6, heads 4, hidden 256, seq 512 → ~0.8–1.5M params.

    Default vocab_size=4096 keeps the total near ~1M with emb=128 (tied).
    Full 32k tokenizer remains available; embedding can be resized in S11.
    """

    vocab_size: int = 4096
    n_embd: int = 128
    n_layer: int = 6
    n_head: int = 4
    n_hidden: int = 192  # SwiGLU intermediate (keeps total ~1.0–1.5M with vocab 4096)
    block_size: int = 512
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True
    rope_theta: float = 10000.0
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3

    # training defaults
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    max_steps: int = 200
    batch_size: int = 8
    grad_clip: float = 1.0
    seed: int = 42
    use_amp: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MiniConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        return cls(**{k: v for k, v in d.items() if k in known})
