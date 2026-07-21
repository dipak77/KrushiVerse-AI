"""Mini LM generation with optional RAG context (Sprint 15)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mini.eval.harness import generate_answer, load_checkpoint, resolve_model_dir
from mini.models.sft_data import SYSTEM_RAG
from mini.models.train import set_seed
from mini.paths import MODELS_DIR


def pick_model_dir(version: str = "auto") -> Path:
    if version and version != "auto":
        return resolve_model_dir(version)
    # Prefer v0.6-base (v2-12M-fixed), serve/v0.5-quant, v0.4-agri-qa
    candidates = [
        MODELS_DIR / "v0.6-base",
        MODELS_DIR / "serve" / "v0.5-quant",
        MODELS_DIR / "v0.5-quant" / "fp32",
        MODELS_DIR / "v0.4-agri-qa",
        MODELS_DIR / "v0.3-instruct",
        MODELS_DIR / "v0.2-base",
    ]
    for c in candidates:
        if (c / "pytorch_model.pt").exists():
            return c
    return resolve_model_dir("v0.6-base")


def mini_generate(
    user_prompt: str,
    *,
    version: str = "auto",
    max_new_tokens: int = 48,
    temperature: float = 0.7,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dir = pick_model_dir(version)
    model, tok, cfg, load_meta = load_checkpoint(model_dir, device=dev)
    text, latency_ms = generate_answer(
        model,
        tok,
        user_prompt,
        device=dev,
        system=SYSTEM_RAG,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return {
        "text": (text or "").strip(),
        "latency_ms": latency_ms,
        "model_dir": str(model_dir),
        "load": load_meta,
        "config_block_size": getattr(cfg, "block_size", None),
        "engine": "mini_lm",
    }
