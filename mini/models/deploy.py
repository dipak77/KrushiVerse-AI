"""Package deploy module for Mini models (Sprint 14 / E5)."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from mini.paths import MODELS_DIR, SERVE_DIR, ensure_lake_layout

SEMVER_POLICY = {
    "format": "v{major}.{minor}.{patch}",
    "examples": ["v0.5-quant", "v0.5-reasoning-lite", "v0.4-agri-qa"],
    "policy": "semantic_versioning"
}

def run_deploy(*, dry_run: bool = False, source_version: str = "v0.4", tag: str = "v0.5-quant", force: bool = False, include_quant: bool = True, reasoning_lite: bool = True) -> dict[str, Any]:
    ensure_lake_layout()
    src_dir = MODELS_DIR / source_version
    serve_path = MODELS_DIR / "serve" / tag
    serve_path.mkdir(parents=True, exist_ok=True)

    # Write package files
    card = {"model_name": "KrushiVerse-AI Mini", "version": tag, "source": source_version, "status": "deployed"}
    (serve_path / "MODEL_CARD.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (serve_path / "LICENSE.txt").write_text("Apache-2.0 License for KrushiVerse-AI", encoding="utf-8")
    manifest = {"tag": tag, "source": source_version, "created_at": datetime.now(timezone.utc).isoformat()}
    (serve_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    tags_written = [tag]
    if reasoning_lite:
        lite_path = MODELS_DIR / "serve" / "v0.5-reasoning-lite"
        lite_path.mkdir(parents=True, exist_ok=True)
        (lite_path / "MODEL_CARD.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
        (lite_path / "LICENSE.txt").write_text("Apache-2.0 License for KrushiVerse-AI", encoding="utf-8")
        (lite_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tags_written.append("v0.5-reasoning-lite")

    reg_file = MODELS_DIR / "VERSION_REGISTRY.json"
    latest_file = MODELS_DIR / "DEPLOY_LATEST.json"
    reg_data = {"latest": tag, "versions": tags_written}
    reg_file.write_text(json.dumps(reg_data, indent=2), encoding="utf-8")
    latest_file.write_text(json.dumps(reg_data, indent=2), encoding="utf-8")

    report = {
        "ok": True,
        "sprint": "S14",
        "dry_run": dry_run,
        "source_version": source_version,
        "tag": tag,
        "serve_path": str(serve_path),
        "tags_written": tags_written,
        "artifacts": [str(serve_path), str(reg_file), str(latest_file)],
    }
    return report
