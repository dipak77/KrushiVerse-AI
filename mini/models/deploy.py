"""Version packaging + registry for Mini (Sprint 14 / W-DEPLOY package-only).

Does not start a server — publishes a local serve package + registry entry:
- model card, license, data provenance
- semantic version policy
- optional v0.5-reasoning-lite tag (short rationale traces note)
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mini.eval.harness import resolve_model_dir
from mini.paths import MODELS_DIR, ensure_lake_layout, relative_to_repo

DEPLOY_LATEST = MODELS_DIR / "DEPLOY_LATEST.json"
REGISTRY_PATH = MODELS_DIR / "VERSION_REGISTRY.json"
SERVE_ROOT = MODELS_DIR / "serve"

# Semantic versioning policy (documented + enforced lightly in registry)
SEMVER_POLICY = {
    "scheme": "semver-ish Mini tags",
    "format": "vMAJOR.MINOR[-label]",
    "rules": [
        "MAJOR: incompatible serving/schema change",
        "MINOR: new training stage or capability (base, instruct, agri-qa, quant, reasoning-lite)",
        "label: optional capability tag (e.g. quant, reasoning-lite)",
        "Never overwrite an existing registry tag without --force",
        "Promote only after W-EVAL gates pass (soft check; not hard-blocked in package mode)",
    ],
    "examples": [
        "v0.2-base",
        "v0.3-instruct",
        "v0.4-agri-qa",
        "v0.5-quant",
        "v0.5-reasoning-lite",
    ],
}

LICENSE_TEXT = """KrushiVerseAI Mini Model Artifacts
Copyright (c) KrushiVerseAI contributors.

These model weights and derived packages are provided for research and
demonstration of the Mini agriculture assistant pipeline.

Data provenance: training/eval content draws from the local Mini data lake
(synthetic expert QA packs, curated gold sets, platform knowledge seeds).
Do not treat outputs as licensed agronomic prescriptions. Follow official
label rates and local agricultural officer guidance for chemical use.

License: Apache-2.0 for accompanying source code in this repository unless
otherwise noted. Model weights are released under the same terms for this
project unless a separate model card states otherwise.
"""


def load_registry() -> dict[str, Any]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "schema_version": "1.0",
        "policy": SEMVER_POLICY,
        "versions": {},
        "latest": None,
        "updated_at": None,
    }


def save_registry(reg: dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    reg["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def build_model_card(
    *,
    tag: str,
    source_version: str,
    quant_report: dict[str, Any] | None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model_name": "KrushiVerseAI-Mini",
        "version_tag": tag,
        "approx_parameters": "~1.36M",
        "architecture": "decoder-only Transformer (RoPE, RMSNorm, SwiGLU)",
        "languages": ["en", "mr", "hi"],
        "domain": "Maharashtra agriculture assistant (Krushi Mitra)",
        "source_checkpoint": source_version,
        "intended_use": "Offline/demo agri QA with RAG coupling (later sprints); not medical/legal advice",
        "limitations": [
            "Small model — may hallucinate; use grounded mode when available",
            "Not a substitute for soil tests or official pesticide labels",
            "Quantized variants may trade quality for size/latency",
        ],
        "data_provenance": {
            "lake": "data/lake (local)",
            "qa_synth": "W-QASYNTH packs",
            "gold_eval": "mini/eval/gold_sets.py curated items",
            "kb_seed": "platform knowledge seeds / KG triples",
        },
        "license": "Apache-2.0 (code); see LICENSE.txt in package",
        "quantization": (quant_report or {}).get("comparison"),
        "evaluation": "See mini/eval/EVAL_LATEST.json when present",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extras": extras or {},
    }


def _copy_tree_files(src: Path, dst: Path, names: list[str]) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in names:
        sp = src / name
        if sp.exists() and sp.is_file():
            shutil.copy2(sp, dst / name)
            copied.append(name)
    return copied


def package_version(
    *,
    tag: str,
    source_dir: Path,
    include_quant: bool = True,
    reasoning_lite: bool = False,
) -> dict[str, Any]:
    """Create serve/<tag>/ package from source + optional quant artifacts."""
    dest = SERVE_ROOT / tag
    dest.mkdir(parents=True, exist_ok=True)

    copied = _copy_tree_files(
        source_dir,
        dest,
        ["config.json", "tokenizer.json", "pytorch_model.pt", "train_report.json"],
    )

    quant_src = MODELS_DIR / "v0.5-quant"
    quant_copied: list[str] = []
    if include_quant and quant_src.exists():
        qdest = dest / "quant"
        qdest.mkdir(parents=True, exist_ok=True)
        for sub in ("fp32", "int8", "int4"):
            s = quant_src / sub
            if s.is_dir():
                t = qdest / sub
                t.mkdir(parents=True, exist_ok=True)
                for f in s.iterdir():
                    if f.is_file():
                        shutil.copy2(f, t / f.name)
                        quant_copied.append(f"{sub}/{f.name}")
        qr = quant_src / "QUANT_REPORT.json"
        if qr.exists():
            shutil.copy2(qr, dest / "QUANT_REPORT.json")
            quant_copied.append("QUANT_REPORT.json")

    reasoning_note = None
    if reasoning_lite:
        reasoning_note = {
            "tag": "v0.5-reasoning-lite",
            "description": (
                "Optional short chain-of-thought style traces for training/eval demos only. "
                "Serving should strip rationales for farmers by default; keep answers concise."
            ),
            "example_trace_format": "Reason: <1-2 steps>\\nAnswer: <final>",
            "enabled": True,
        }
        (dest / "REASONING_LITE.json").write_text(
            json.dumps(reasoning_note, indent=2), encoding="utf-8"
        )

    return {
        "dest": dest,
        "copied": copied,
        "quant_copied": quant_copied,
        "reasoning_lite": reasoning_note,
    }


def run_deploy(
    *,
    dry_run: bool = False,
    source_version: str = "v0.4",
    tag: str = "v0.5-quant",
    force: bool = False,
    include_quant: bool = True,
    reasoning_lite: bool = True,
) -> dict[str, Any]:
    ensure_lake_layout()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SERVE_ROOT.mkdir(parents=True, exist_ok=True)

    src = resolve_model_dir(source_version)
    # Prefer packaged fp32 from quant stage if present
    quant_fp32 = MODELS_DIR / "v0.5-quant" / "fp32"
    if include_quant and (quant_fp32 / "pytorch_model.pt").exists():
        package_src = quant_fp32
    elif (src / "pytorch_model.pt").exists():
        package_src = src
    else:
        package_src = src

    quant_report = None
    ql = MODELS_DIR / "QUANT_LATEST.json"
    if ql.exists():
        try:
            quant_report = json.loads(ql.read_text(encoding="utf-8"))
        except Exception:
            quant_report = None

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "sprint": "S14",
            "feature_phase": "E5-deploy-pkg",
            "tag": tag,
            "source_version": source_version,
            "package_src": relative_to_repo(package_src) if package_src.exists() else str(package_src),
            "policy": SEMVER_POLICY,
            "include_quant": include_quant,
            "reasoning_lite": reasoning_lite,
        }

    reg = load_registry()
    if tag in (reg.get("versions") or {}) and not force:
        return {
            "ok": False,
            "dry_run": False,
            "sprint": "S14",
            "error": f"Tag '{tag}' already in registry; use force=True to overwrite",
            "registry": relative_to_repo(REGISTRY_PATH),
        }

    pkg = package_version(
        tag=tag,
        source_dir=package_src,
        include_quant=include_quant,
        reasoning_lite=reasoning_lite,
    )
    dest: Path = pkg["dest"]

    card = build_model_card(
        tag=tag,
        source_version=source_version,
        quant_report=quant_report,
        extras={
            "reasoning_lite": bool(reasoning_lite),
            "package_files": pkg["copied"],
            "quant_files": pkg["quant_copied"],
        },
    )
    (dest / "MODEL_CARD.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    (dest / "LICENSE.txt").write_text(LICENSE_TEXT, encoding="utf-8")
    (dest / "PROVENANCE.json").write_text(
        json.dumps(card["data_provenance"], indent=2),
        encoding="utf-8",
    )
    (dest / "SEMVER_POLICY.json").write_text(json.dumps(SEMVER_POLICY, indent=2), encoding="utf-8")

    # serve manifest
    manifest = {
        "tag": tag,
        "path": relative_to_repo(dest),
        "files": sorted(p.name for p in dest.iterdir() if p.is_file()),
        "has_quant": bool(pkg["quant_copied"]),
        "reasoning_lite": bool(reasoning_lite),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # also publish optional reasoning-lite alias entry when requested
    tags_written = [tag]
    if reasoning_lite and tag != "v0.5-reasoning-lite":
        alias = "v0.5-reasoning-lite"
        alias_pkg = package_version(
            tag=alias,
            source_dir=package_src,
            include_quant=include_quant,
            reasoning_lite=True,
        )
        adest: Path = alias_pkg["dest"]
        acard = build_model_card(
            tag=alias,
            source_version=source_version,
            quant_report=quant_report,
            extras={"alias_of": tag, "reasoning_lite": True},
        )
        (adest / "MODEL_CARD.json").write_text(json.dumps(acard, indent=2, ensure_ascii=False), encoding="utf-8")
        (adest / "LICENSE.txt").write_text(LICENSE_TEXT, encoding="utf-8")
        (adest / "manifest.json").write_text(
            json.dumps(
                {
                    "tag": alias,
                    "path": relative_to_repo(adest),
                    "alias_of": tag,
                    "reasoning_lite": True,
                    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tags_written.append(alias)
        reg.setdefault("versions", {})[alias] = {
            "tag": alias,
            "path": relative_to_repo(adest),
            "source_version": source_version,
            "reasoning_lite": True,
            "alias_of": tag,
        }

    reg.setdefault("versions", {})[tag] = {
        "tag": tag,
        "path": relative_to_repo(dest),
        "source_version": source_version,
        "has_quant": bool(pkg["quant_copied"]),
        "reasoning_lite": bool(reasoning_lite),
        "model_card": relative_to_repo(dest / "MODEL_CARD.json"),
        "created_at": manifest["created_at"],
    }
    reg["latest"] = tag
    reg["policy"] = SEMVER_POLICY
    save_registry(reg)

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {
        "ok": True,
        "dry_run": False,
        "sprint": "S14",
        "feature_phase": "E5-deploy-pkg",
        "tag": tag,
        "tags_written": tags_written,
        "source_version": source_version,
        "package_src": relative_to_repo(package_src),
        "serve_path": relative_to_repo(dest),
        "registry": relative_to_repo(REGISTRY_PATH),
        "manifest": manifest,
        "model_card": card,
        "policy": SEMVER_POLICY,
        "artifacts": [
            relative_to_repo(dest / "MODEL_CARD.json"),
            relative_to_repo(dest / "LICENSE.txt"),
            relative_to_repo(dest / "manifest.json"),
            relative_to_repo(REGISTRY_PATH),
        ],
        "created_at": created,
    }
    DEPLOY_LATEST.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    report["artifacts"] = list(dict.fromkeys(report["artifacts"] + [relative_to_repo(DEPLOY_LATEST)]))
    DEPLOY_LATEST.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report
