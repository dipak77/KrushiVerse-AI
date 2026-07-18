"""Validate → Clean → Dedup processing pipeline for lake/raw → lake/processed (Sprint 3)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mini.lake.dedup import dedupe_payload
from mini.lake.quality import clean_value
from mini.lake.validate import iter_raw_data_files, run_validation, validate_file
from mini.paths import (
    LAKE_PROCESSED,
    LAKE_RAW,
    LAKE_ROOT,
    ensure_lake_layout,
    relative_to_repo,
)


def _processed_dest(raw_path: Path) -> Path:
    rel = raw_path.resolve().relative_to(LAKE_RAW.resolve())
    return LAKE_PROCESSED / rel


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_clean_stage(*, dry_run: bool = False, only_valid: bool = True) -> dict[str, Any]:
    """Clean valid raw JSON files into processed/ (overwrite)."""
    ensure_lake_layout()
    files = iter_raw_data_files()
    cleaned = 0
    skipped = 0
    failed = 0
    details: list[dict[str, Any]] = []

    for path in files:
        if path.suffix.lower() != ".json":
            skipped += 1
            continue
        vr = validate_file(path)
        if only_valid and not vr["ok"]:
            skipped += 1
            details.append(
                {
                    "path": relative_to_repo(path),
                    "action": "skipped_invalid",
                    "errors": vr["errors"],
                }
            )
            continue
        try:
            data = _load_json(path)
            cleaned_data = clean_value(data)
            dest = _processed_dest(path)
            if not dry_run:
                _write_json(dest, cleaned_data)
                meta = {
                    "source": relative_to_repo(path),
                    "stage": "clean",
                    "cleaned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                dest.with_suffix(dest.suffix + ".clean.meta.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )
            cleaned += 1
            details.append(
                {
                    "path": relative_to_repo(path),
                    "dest": relative_to_repo(dest),
                    "action": "cleaned",
                }
            )
        except Exception as e:
            failed += 1
            details.append(
                {"path": relative_to_repo(path), "action": "error", "error": str(e)}
            )

    return {
        "ok": failed == 0,
        "cleaned": cleaned,
        "skipped": skipped,
        "failed": failed,
        "details": details,
    }


def iter_processed_json_files() -> list[Path]:
    files: list[Path] = []
    if not LAKE_PROCESSED.exists():
        return files
    for p in sorted(LAKE_PROCESSED.rglob("*.json")):
        if p.name.endswith(".meta.json") or ".clean.meta." in p.name or ".dedup.meta." in p.name:
            continue
        files.append(p)
    return files


def run_dedup_stage(*, dry_run: bool = False, near_threshold: float = 0.92) -> dict[str, Any]:
    """Deduplicate records inside processed JSON files."""
    ensure_lake_layout()
    files = iter_processed_json_files()
    # If processed empty, clean first is expected — still ok with 0
    total_exact = 0
    total_near = 0
    total_input = 0
    total_kept = 0
    updated = 0
    details: list[dict[str, Any]] = []

    for path in files:
        try:
            data = _load_json(path)
            deduped, stats = dedupe_payload(data, near_threshold=near_threshold)
            file_exact = sum(s.get("exact_removed", 0) for s in stats.values())
            file_near = sum(s.get("near_removed", 0) for s in stats.values())
            file_in = sum(s.get("input", 0) for s in stats.values())
            file_kept = sum(s.get("kept", 0) for s in stats.values())
            total_exact += file_exact
            total_near += file_near
            total_input += file_in
            total_kept += file_kept

            changed = file_exact + file_near > 0
            if changed and not dry_run:
                _write_json(path, deduped)
                path.with_suffix(path.suffix + ".dedup.meta.json").write_text(
                    json.dumps(
                        {
                            "stage": "dedup",
                            "stats": stats,
                            "deduped_at": datetime.now(timezone.utc).strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            ),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                updated += 1
            details.append(
                {
                    "path": relative_to_repo(path),
                    "stats": stats,
                    "changed": changed,
                }
            )
        except Exception as e:
            details.append({"path": relative_to_repo(path), "error": str(e)})

    return {
        "ok": True,
        "files": len(files),
        "files_updated": updated,
        "records_input": total_input,
        "records_kept": total_kept,
        "exact_removed": total_exact,
        "near_removed": total_near,
        "near_threshold": near_threshold,
        "details": details,
    }


def run_quality_pipeline(
    *,
    dry_run: bool = False,
    quarantine: bool = True,
    near_threshold: float = 0.92,
) -> dict[str, Any]:
    """Full Sprint 3 pipeline: validate → clean → dedup + quality report."""
    ensure_lake_layout()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    validation = run_validation(dry_run=dry_run, quarantine=quarantine)
    cleaning = run_clean_stage(dry_run=dry_run, only_valid=True)
    dedup = run_dedup_stage(dry_run=dry_run, near_threshold=near_threshold)

    report = {
        "run_id": run_id,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": dry_run,
        "ok": bool(validation.get("ok") and cleaning.get("ok") and dedup.get("ok")),
        "validation": validation,
        "cleaning": cleaning,
        "dedup": dedup,
    }

    if not dry_run:
        reports = LAKE_ROOT / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        out = reports / f"quality_{run_id}.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        latest = LAKE_ROOT / "QUALITY_LATEST.json"
        latest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["report_paths"] = [relative_to_repo(out), relative_to_repo(latest)]
    else:
        report["report_paths"] = []

    return report
