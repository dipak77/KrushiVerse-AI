"""Artifact path conventions for the Mini data lake and factory outputs.

Rule: never train directly from raw/. Only versioned standard records
under training/validation/test (or datasets/versions/) may feed trainers.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root (parent of mini/)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Mini factory root
MINI_ROOT = REPO_ROOT / "mini"

# Data lake (Phase 3)
LAKE_ROOT = REPO_ROOT / "data" / "lake"
LAKE_RAW = LAKE_ROOT / "raw"
LAKE_PROCESSED = LAKE_ROOT / "processed"
LAKE_TRAINING = LAKE_ROOT / "training"
LAKE_VALIDATION = LAKE_ROOT / "validation"
LAKE_TEST = LAKE_ROOT / "test"
LAKE_QUARANTINE = LAKE_ROOT / "quarantine"

# Domain folders under raw/processed
LAKE_DOMAINS = (
    "weather",
    "soil",
    "crop",
    "market",
    "government",
    "disease",
    "pest",
    "fertilizer",
    "irrigation",
    "finance",
    "machinery",
    "images",
    "seed",  # bootstrap seed from existing platform KB
    "general",
)

# Factory artifact roots
MODELS_DIR = MINI_ROOT / "models"
TOKENIZER_DIR = MINI_ROOT / "tokenizer"
DATASETS_DIR = MINI_ROOT / "datasets"
EVAL_DIR = MINI_ROOT / "eval"
INFERENCE_DIR = MINI_ROOT / "inference"
ORCHESTRATOR_DIR = MINI_ROOT / "orchestrator"
TAXONOMY_DIR = MINI_ROOT / "taxonomy"
SERVE_DIR = MINI_ROOT / "serve"
EVENTS_LOG = ORCHESTRATOR_DIR / "events.jsonl"
RUNS_DIR = MINI_ROOT / "runs"

SCHEMA_VERSION = "1.0"


def ensure_lake_layout() -> list[Path]:
    """Create the data lake directory tree. Returns paths created or existing."""
    created: list[Path] = []
    roots = [
        LAKE_ROOT,
        LAKE_RAW,
        LAKE_PROCESSED,
        LAKE_TRAINING,
        LAKE_VALIDATION,
        LAKE_TEST,
        LAKE_QUARANTINE,
        MODELS_DIR,
        TOKENIZER_DIR,
        DATASETS_DIR,
        EVAL_DIR,
        INFERENCE_DIR,
        RUNS_DIR,
        TAXONOMY_DIR,
    ]
    for p in roots:
        p.mkdir(parents=True, exist_ok=True)
        created.append(p)

    for domain in LAKE_DOMAINS:
        for base in (LAKE_RAW, LAKE_PROCESSED):
            d = base / domain
            d.mkdir(parents=True, exist_ok=True)
            created.append(d)
            gitkeep = d / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.write_text("", encoding="utf-8")

    for split in (LAKE_TRAINING, LAKE_VALIDATION, LAKE_TEST, LAKE_QUARANTINE):
        gk = split / ".gitkeep"
        if not gk.exists():
            gk.write_text("", encoding="utf-8")

    return created


def run_dir(run_id: str) -> Path:
    """Per-run artifact directory under mini/runs/{run_id}/."""
    p = RUNS_DIR / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
