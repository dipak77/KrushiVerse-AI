"""Published Mini model version matrix (Sprint 17)."""

from __future__ import annotations

from typing import Any

from mini.paths import MODELS_DIR, TOKENIZER_DIR


VERSION_MATRIX: list[dict[str, Any]] = [
    {
        "tag": "v0.1-tokenizer",
        "meaning": "Domain SentencePiece vocab 30–50k",
        "sprint": "S9",
        "path": "mini/tokenizer/v0.1",
        "status_key": "tokenizer",
    },
    {
        "tag": "v0.2-base",
        "meaning": "Domain pretrained Mini ~1M",
        "sprint": "S11",
        "path": "mini/models/v0.2-base",
        "status_key": "v02",
    },
    {
        "tag": "v0.3-instruct",
        "meaning": "Instruction + safety SFT",
        "sprint": "S12",
        "path": "mini/models/v0.3-instruct",
        "status_key": "v03",
    },
    {
        "tag": "v0.4-agri-qa",
        "meaning": "Agri-QA + RAG-context SFT",
        "sprint": "S12",
        "path": "mini/models/v0.4-agri-qa",
        "status_key": "v04",
    },
    {
        "tag": "v0.5-quant",
        "meaning": "INT8/INT4 package + serve bundle",
        "sprint": "S14",
        "path": "mini/models/v0.5-quant",
        "status_key": "v05",
    },
    {
        "tag": "v0.5-reasoning-lite",
        "meaning": "Optional short rationale package tag",
        "sprint": "S14",
        "path": "mini/models/serve/v0.5-reasoning-lite",
        "status_key": "v05r",
    },
    {
        "tag": "v0.6-rag-coupled",
        "meaning": "Serving path with grounded RAG+Mini (code+infer)",
        "sprint": "S15",
        "path": "mini/inference",
        "status_key": "v06",
        "code_only": True,
    },
    {
        "tag": "v0.8-multi-agent",
        "meaning": "Planner Mini synthesizer + USE_MINI_LLM",
        "sprint": "S16",
        "path": "app/llm/mini_bridge.py",
        "status_key": "v08",
        "code_only": True,
    },
    {
        "tag": "v0.9-prod-beta",
        "meaning": "Hardened release candidate tooling",
        "sprint": "S17",
        "path": "mini/release",
        "status_key": "v09",
        "code_only": True,
    },
    {
        "tag": "v1.0-mini",
        "meaning": "Mini v1.0 program release",
        "sprint": "S17",
        "path": "mini/",
        "status_key": "v10",
        "code_only": True,
    },
]


def probe_local_artifacts() -> dict[str, Any]:
    """Check which local (gitignored) artifacts exist on this machine."""
    from pathlib import Path

    def exists(rel: str) -> bool:
        p = Path(rel)
        if not p.is_absolute():
            # repo-relative
            from mini.paths import REPO_ROOT

            p = REPO_ROOT / rel
        if p.is_file():
            return p.exists()
        if p.is_dir():
            # dir "present" if any weight/config or non-empty
            if not p.exists():
                return False
            return any(p.iterdir()) if p.is_dir() else True
        return p.exists()

    present = {}
    for row in VERSION_MATRIX:
        present[row["status_key"]] = {
            "tag": row["tag"],
            "path": row["path"],
            "exists": exists(row["path"]),
            "code_only": bool(row.get("code_only")),
        }
    # extra probes
    present["serve_root"] = {
        "tag": "serve/",
        "path": str(MODELS_DIR / "serve"),
        "exists": (MODELS_DIR / "serve").exists(),
    }
    present["tokenizer_latest"] = {
        "tag": "TOKENIZER_LATEST",
        "path": str(TOKENIZER_DIR / "TOKENIZER_LATEST.json"),
        "exists": (TOKENIZER_DIR / "TOKENIZER_LATEST.json").exists(),
    }
    return present


def build_version_matrix_report() -> dict[str, Any]:
    local = probe_local_artifacts()
    rows = []
    for v in VERSION_MATRIX:
        loc = local.get(v["status_key"]) or {}
        rows.append(
            {
                **v,
                "local_present": bool(loc.get("exists")),
            }
        )
    return {
        "title": "KrushiVerseAI Mini — Model Version Matrix",
        "versions": rows,
        "local_probe": local,
        "notes": [
            "Weight checkpoints are local-only (gitignored); code paths are versioned in git.",
            "Promote only after W-EVAL gates + release checklist (S17 RC gate).",
        ],
    }
