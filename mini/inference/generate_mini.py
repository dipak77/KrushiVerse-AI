"""Mini LM generation with optional RAG context (Sprint 15 - FIXED v2-12M-fixed)."""

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
    # FIXED: Prioritize SFT instruction-tuned v0.4-agri-qa over base - Gemini was correct here
    candidates = [
        MODELS_DIR / "v0.4-agri-qa",          # SFT best - instruction tuned
        MODELS_DIR / "serve" / "v0.5-quant",    # Quant served
        MODELS_DIR / "v0.5-quant" / "fp32",     # Quant fp32
        MODELS_DIR / "v0.3-instruct",           # Fallback instruct
        MODELS_DIR / "v0.6-base",               # Base only - should be last
        MODELS_DIR / "v0.2-base",
    ]
    for c in candidates:
        if (c / "pytorch_model.pt").exists():
            return c
    return resolve_model_dir("v0.4-agri-qa")  # FIXED: was v0.4 (ambiguous)


def mini_generate(
    user_prompt: str,
    *,
    version: str = "auto",
    max_new_tokens: int = 256,  # FIXED: was 48 - too short, feels like matching. Need 256 for ChatGPT quality
    temperature: float = 0.7,
    top_p: float = 0.9,          # ADDED
    do_sample: bool = True,       # ADDED
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dir = pick_model_dir(version)
    model, tok, cfg, load_meta = load_checkpoint(model_dir, device=dev)
    
    # FIXED: Use SYSTEM_RAG with instruction, not just raw prompt
    # v2-12M-fixed: block_size 512, so prompt must be < 256 tokens to leave 256 for generation
    system_prompt = SYSTEM_RAG if hasattr(SYSTEM_RAG, "__len__") else "You are KrushiVerse-AI, expert agriculture assistant for Maharashtra farmers. Answer in Marathi if query is Marathi."
    
    text, latency_ms = generate_answer(
        model,
        tok,
        user_prompt,
        device=dev,
        system=system_prompt,
        max_new_tokens=max_new_tokens,  # Now 256 not 48
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )
    return {
        "text": (text or "").strip(),
        "latency_ms": latency_ms,
        "model_dir": str(model_dir),
        "load": load_meta,
        "config_block_size": getattr(cfg, "block_size", 512),  # FIXED: default 512 not None
        "engine": "mini_lm",
        "model_variant": "v2-12M-fixed",
    }
