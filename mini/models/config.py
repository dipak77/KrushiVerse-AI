"""Mini decoder-only config.

v1.0 default: ~1.36M (vocab 4096, emb 128, 6 layers).
v2.0-15M: load via ``MiniConfig.v2_15m()`` or ``load_config_json(...)``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mini.paths import MODELS_DIR, REPO_ROOT

# Standard PyTorch ignore index for masked LM labels (assistant-only SFT / pads)
IGNORE_INDEX = -100

CONFIG_V2_15M_PATH = MODELS_DIR / "config_v2_15M.json"
CONFIG_V2_15M_DOCS = REPO_ROOT / "docs" / "Next 15M Gen Plan" / "config_v2_15M.json"


@dataclass
class MiniConfig:
    """Decoder-only Mini LM hyperparameters.

    Defaults remain v1.0 (~1.36M) for backward compatibility.
    Use ``MiniConfig.v2_15m()`` for the 15M production target.
    """

    vocab_size: int = 4096
    n_embd: int = 128
    n_layer: int = 6
    n_head: int = 4
    n_hidden: int = 192  # SwiGLU intermediate
    block_size: int = 512
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True
    rope_theta: float = 10000.0
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3
    ignore_index: int = IGNORE_INDEX

    # training defaults (v1-oriented; v2 overrides via JSON / factory)
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    max_steps: int = 200
    batch_size: int = 8
    grad_clip: float = 1.0
    seed: int = 42
    use_amp: bool = True
    gradient_checkpointing: bool = False
    model_variant: str = "v1-1M"  # informational

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MiniConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        # Map nested JSON hyperparameter blocks
        if "hyperparameters" in d and isinstance(d["hyperparameters"], dict):
            hp = dict(d["hyperparameters"])
            # drop non-config keys
            for drop in ("head_dim", "ffn_multiplier", "positional_encoding", "normalization", "activation", "attention_type", "weight_tying"):
                if drop == "weight_tying" and "weight_tying" in hp:
                    hp["tie_weights"] = bool(hp.pop("weight_tying"))
                else:
                    hp.pop(drop, None)
            d = {**hp, **{k: v for k, v in d.items() if k != "hyperparameters"}}
        if "tie_weights" not in d and "weight_tying" in d:
            d["tie_weights"] = bool(d.pop("weight_tying"))
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @classmethod
    def v2_15m(cls) -> "MiniConfig":
        """15M production config (canonical)."""
        path = CONFIG_V2_15M_PATH if CONFIG_V2_15M_PATH.exists() else CONFIG_V2_15M_DOCS
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            cfg = cls.from_dict(raw)
            cfg.model_variant = "v2-15M"
            # Training defaults for RTX 2050 laptop override
            cfg.learning_rate = float(
                ((raw.get("training") or {}).get("pretrain") or {}).get("learning_rate") or 3e-4
            )
            cfg.weight_decay = float(
                ((raw.get("training") or {}).get("pretrain") or {}).get("weight_decay") or 0.01
            )
            cfg.use_amp = True
            cfg.gradient_checkpointing = True
            cfg.batch_size = 4  # safe laptop default; accum in trainer
            cfg.max_steps = int(
                ((raw.get("training") or {}).get("pretrain") or {}).get("steps") or 15000
            )
            return cfg
        # Hardcoded fallback if JSON missing
        return cls(
            vocab_size=8192,
            n_embd=320,
            n_layer=10,
            n_head=8,
            n_hidden=864,
            block_size=1024,
            dropout=0.0,
            bias=False,
            tie_weights=True,
            learning_rate=3e-4,
            weight_decay=0.01,
            max_steps=15000,
            batch_size=4,
            use_amp=True,
            gradient_checkpointing=True,
            model_variant="v2-15M",
        )


def load_config_json(path: str | Path | None = None) -> MiniConfig:
    """Load MiniConfig from a v2 JSON (or any hyperparameter JSON)."""
    if path is None:
        return MiniConfig.v2_15m()
    p = Path(path)
    if not p.is_absolute():
        for base in (REPO_ROOT, MODELS_DIR, REPO_ROOT / "docs" / "Next 15M Gen Plan"):
            cand = base / p
            if cand.exists():
                p = cand
                break
    raw = json.loads(p.read_text(encoding="utf-8"))
    cfg = MiniConfig.from_dict(raw)
    cfg.model_variant = str(raw.get("version") or cfg.model_variant)
    return cfg
