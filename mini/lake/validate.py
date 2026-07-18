"""Raw lake file validation (Sprint 3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mini.paths import LAKE_DOMAINS, LAKE_QUARANTINE, LAKE_RAW, relative_to_repo


# Domain → required top-level keys (any one of the groups can pass)
DOMAIN_KEY_HINTS: dict[str, list[str]] = {
    "crop": ["crops", "advisories", "varieties"],
    "disease": ["diseases_and_pests", "diseases", "crops"],
    "pest": ["diseases_and_pests", "pests", "crops"],
    "soil": ["soil_types", "fertilizer_recommendations"],
    "fertilizer": ["fertilizer_recommendations", "soil_types"],
    "government": ["schemes"],
    "market": ["markets", "records"],
    "irrigation": ["practices"],
    "seed": ["varieties"],
    "weather": ["zones", "records"],
    "general": ["nodes", "edges", "advisories", "sources", "meta"],
    "finance": ["schemes", "records"],
    "machinery": ["records", "items"],
    "images": [],  # optional
}


def iter_raw_data_files() -> list[Path]:
    files: list[Path] = []
    for domain in LAKE_DOMAINS:
        dpath = LAKE_RAW / domain
        if not dpath.exists():
            continue
        for p in sorted(dpath.rglob("*")):
            if not p.is_file():
                continue
            if p.name in (".gitkeep",) or p.name.endswith(".meta.json"):
                continue
            if p.suffix.lower() not in (".json", ".jsonl", ".txt", ".csv"):
                continue
            files.append(p)
    return files


def _domain_of(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(LAKE_RAW.resolve())
        return rel.parts[0] if rel.parts else "general"
    except ValueError:
        return "general"


def validate_json_payload(data: Any, domain: str) -> list[str]:
    """Return list of errors (empty = valid)."""
    errors: list[str] = []
    if data is None:
        return ["payload is null"]
    if isinstance(data, list):
        if len(data) == 0:
            errors.append("empty list payload")
        return errors
    if not isinstance(data, dict):
        return [f"expected object or list, got {type(data).__name__}"]
    if len(data) == 0:
        return ["empty object payload"]

    hints = DOMAIN_KEY_HINTS.get(domain, [])
    if hints:
        if not any(k in data for k in hints):
            # soft: allow if any list/dict content present
            has_content = any(isinstance(v, (list, dict)) and v for v in data.values())
            if not has_content:
                errors.append(
                    f"domain '{domain}' missing expected keys {hints} and no nested content"
                )
    return errors


def validate_file(path: Path) -> dict[str, Any]:
    domain = _domain_of(path)
    result: dict[str, Any] = {
        "path": relative_to_repo(path),
        "domain": domain,
        "ok": False,
        "errors": [],
        "bytes": path.stat().st_size if path.exists() else 0,
        "record_estimate": 0,
    }
    if path.stat().st_size == 0:
        result["errors"].append("empty file")
        return result

    if path.suffix.lower() == ".json":
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except Exception as e:
                result["errors"].append(f"encoding error: {e}")
                return result
        except Exception as e:
            result["errors"].append(f"read error: {e}")
            return result

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            result["errors"].append(f"invalid JSON: {e}")
            return result

        result["errors"].extend(validate_json_payload(data, domain))
        result["record_estimate"] = estimate_records(data)
        result["ok"] = len(result["errors"]) == 0
        return result

    # non-json: accept non-empty utf-8 text
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            result["errors"].append("empty text content")
        else:
            result["ok"] = True
            result["record_estimate"] = 1
    except Exception as e:
        result["errors"].append(str(e))
    return result


def estimate_records(data: Any) -> int:
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in (
            "crops",
            "diseases_and_pests",
            "markets",
            "schemes",
            "advisories",
            "varieties",
            "practices",
            "zones",
            "soil_types",
            "fertilizer_recommendations",
            "sources",
            "nodes",
            "records",
        ):
            if isinstance(data.get(key), list):
                return len(data[key])
        return 1
    return 0


def quarantine_file(path: Path, reason: str) -> Path:
    """Copy invalid file into quarantine with reason sidecar."""
    domain = _domain_of(path)
    dest_dir = LAKE_QUARANTINE / domain
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    # avoid overwrite collisions
    if dest.exists():
        stem, suf = path.stem, path.suffix
        i = 1
        while dest.exists():
            dest = dest_dir / f"{stem}__q{i}{suf}"
            i += 1
    dest.write_bytes(path.read_bytes())
    (dest.with_suffix(dest.suffix + ".reason.json")).write_text(
        json.dumps(
            {
                "source": relative_to_repo(path),
                "reason": reason,
                "domain": domain,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dest


def run_validation(
    *,
    dry_run: bool = False,
    quarantine: bool = True,
) -> dict[str, Any]:
    files = iter_raw_data_files()
    results = []
    valid = 0
    invalid = 0
    quarantined: list[str] = []

    for path in files:
        vr = validate_file(path)
        results.append(vr)
        if vr["ok"]:
            valid += 1
        else:
            invalid += 1
            if quarantine and not dry_run:
                qpath = quarantine_file(path, "; ".join(vr["errors"]))
                quarantined.append(relative_to_repo(qpath))

    return {
        "ok": invalid == 0 or valid > 0,  # pipeline continues if any valid
        "files_scanned": len(files),
        "valid": valid,
        "invalid": invalid,
        "quarantined": quarantined,
        "results": results,
    }
