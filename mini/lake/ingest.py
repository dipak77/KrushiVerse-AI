"""Ingest engine: copy/fetch sources into data/lake/raw with manifests (Sprint 2).

Rules:
- Never write training records here (raw only).
- Idempotent: same content hash → skip copy.
- Every batch writes a JSON manifest with SHA-256 hashes and run_id.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from mini.lake.registry import SourceRegistry, SourceSpec, load_source_registry
from mini.paths import (
    LAKE_DOMAINS,
    LAKE_RAW,
    LAKE_ROOT,
    REPO_ROOT,
    ensure_lake_layout,
    relative_to_repo,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class IngestFileResult(BaseModel):
    source_id: str
    domain: str
    source_path: str | None = None
    dest_path: str | None = None
    sha256: str | None = None
    bytes: int | None = None
    action: str  # copied | skipped_identical | missing | error | fetched | skipped_disabled | skipped_env
    message: str = ""


class IngestReport(BaseModel):
    run_id: str
    started_at: str
    finished_at: str | None = None
    dry_run: bool = False
    ok: bool = True
    sources_considered: int = 0
    files_copied: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    results: list[IngestFileResult] = Field(default_factory=list)
    manifest_paths: list[str] = Field(default_factory=list)
    lake_raw: str = ""
    errors: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class IngestEngine:
    def __init__(self, registry: SourceRegistry | None = None):
        self.registry = registry or load_source_registry()

    def run(
        self,
        *,
        dry_run: bool = False,
        source_ids: list[str] | None = None,
        include_http: bool = True,
    ) -> IngestReport:
        ensure_lake_layout()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        report = IngestReport(
            run_id=run_id,
            started_at=started,
            dry_run=dry_run,
            lake_raw=relative_to_repo(LAKE_RAW),
        )

        sources = self.registry.enabled_sources()
        if source_ids:
            sources = [s for s in sources if s.id in source_ids]
        report.sources_considered = len(sources)

        batch_entries: list[dict[str, Any]] = []

        for src in sources:
            if src.type == "local_file":
                file_results = self._ingest_local_file(src, dry_run=dry_run)
            elif src.type == "http_api" and include_http:
                file_results = self._ingest_http_api(src, dry_run=dry_run)
            else:
                file_results = [
                    IngestFileResult(
                        source_id=src.id,
                        domain=src.domain,
                        action="skipped_disabled",
                        message=f"type={src.type} not ingested in this run",
                    )
                ]

            for fr in file_results:
                report.results.append(fr)
                if fr.action in ("copied", "fetched"):
                    report.files_copied += 1
                    batch_entries.append(fr.model_dump(mode="json"))
                elif fr.action.startswith("skipped"):
                    report.files_skipped += 1
                elif fr.action in ("missing", "error"):
                    report.files_failed += 1
                    report.errors.append(f"{src.id}: {fr.message}")
                    report.ok = False

        finished = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        report.finished_at = finished

        if not dry_run:
            manifest_path = LAKE_ROOT / "manifests" / f"ingest_{run_id}.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest = {
                "batch_id": run_id,
                "worker_id": "W-INGEST",
                "created_at": finished,
                "schema_version": "1.0",
                "registry_version": self.registry.version,
                "entries": batch_entries,
                "summary": {
                    "copied": report.files_copied,
                    "skipped": report.files_skipped,
                    "failed": report.files_failed,
                    "sources_considered": report.sources_considered,
                },
                "ok": report.ok,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            report.manifest_paths.append(relative_to_repo(manifest_path))

            latest = LAKE_ROOT / "INGEST_LATEST.json"
            latest.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
            report.manifest_paths.append(relative_to_repo(latest))

        return report

    def _ingest_local_file(self, src: SourceSpec, *, dry_run: bool) -> list[IngestFileResult]:
        path = src.resolve_path()
        if path is None or not path.exists():
            return [
                IngestFileResult(
                    source_id=src.id,
                    domain=src.domain,
                    source_path=src.path,
                    action="missing",
                    message=f"Source file not found: {src.path}",
                )
            ]

        digest = sha256_file(path)
        size = path.stat().st_size
        results: list[IngestFileResult] = []

        for domain in src.all_domains():
            if domain not in LAKE_DOMAINS:
                results.append(
                    IngestFileResult(
                        source_id=src.id,
                        domain=domain,
                        action="error",
                        message=f"Unknown lake domain: {domain}",
                    )
                )
                continue

            dest_dir = LAKE_RAW / domain / src.id
            dest = dest_dir / path.name
            rel_dest = relative_to_repo(dest)
            rel_src = relative_to_repo(path)

            if dest.exists():
                try:
                    existing = sha256_file(dest)
                except OSError:
                    existing = ""
                if existing == digest:
                    results.append(
                        IngestFileResult(
                            source_id=src.id,
                            domain=domain,
                            source_path=rel_src,
                            dest_path=rel_dest,
                            sha256=digest,
                            bytes=size,
                            action="skipped_identical",
                            message="Idempotent skip — content hash matches",
                        )
                    )
                    continue

            if dry_run:
                results.append(
                    IngestFileResult(
                        source_id=src.id,
                        domain=domain,
                        source_path=rel_src,
                        dest_path=rel_dest,
                        sha256=digest,
                        bytes=size,
                        action="copied",
                        message="dry-run: would copy",
                    )
                )
                continue

            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            # write per-file sidecar meta
            meta = {
                "source_id": src.id,
                "source_name": src.name,
                "domain": domain,
                "origin": src.origin,
                "license": src.license,
                "sha256": digest,
                "bytes": size,
                "ingested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source_path": rel_src,
            }
            (dest_dir / f"{path.name}.meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            results.append(
                IngestFileResult(
                    source_id=src.id,
                    domain=domain,
                    source_path=rel_src,
                    dest_path=rel_dest,
                    sha256=digest,
                    bytes=size,
                    action="copied",
                    message="Copied to lake/raw",
                )
            )
        return results

    def _ingest_http_api(self, src: SourceSpec, *, dry_run: bool) -> list[IngestFileResult]:
        if not src.env_satisfied():
            return [
                IngestFileResult(
                    source_id=src.id,
                    domain=src.domain,
                    action="skipped_env",
                    message=f"Missing env: {src.requires_env}",
                )
            ]

        # Sprint 2: only implement data.gov.in agmarknet snapshot when configured
        if src.id == "datagov_agmarknet":
            return self._ingest_agmarknet(src, dry_run=dry_run)

        return [
            IngestFileResult(
                source_id=src.id,
                domain=src.domain,
                action="skipped_disabled",
                message="HTTP source handler not implemented for this id in Sprint 2",
            )
        ]

    def _ingest_agmarknet(self, src: SourceSpec, *, dry_run: bool) -> list[IngestFileResult]:
        try:
            from app.live_feeds.opendata_client import opendata_client

            payload = opendata_client.fetch_commodity_prices(
                state="Maharashtra", commodity=None, limit=50
            )
            raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            digest = sha256_bytes(raw)
            dest_dir = LAKE_RAW / src.domain / src.id
            dest = dest_dir / "agmarknet_maharashtra_latest.json"
            rel_dest = relative_to_repo(dest)

            if dest.exists() and sha256_file(dest) == digest:
                return [
                    IngestFileResult(
                        source_id=src.id,
                        domain=src.domain,
                        dest_path=rel_dest,
                        sha256=digest,
                        bytes=len(raw),
                        action="skipped_identical",
                        message=f"Idempotent skip (mode={payload.get('mode')})",
                    )
                ]

            if dry_run:
                return [
                    IngestFileResult(
                        source_id=src.id,
                        domain=src.domain,
                        dest_path=rel_dest,
                        sha256=digest,
                        bytes=len(raw),
                        action="fetched",
                        message=f"dry-run: would write mode={payload.get('mode')}",
                    )
                ]

            dest_dir.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            return [
                IngestFileResult(
                    source_id=src.id,
                    domain=src.domain,
                    dest_path=rel_dest,
                    sha256=digest,
                    bytes=len(raw),
                    action="fetched",
                    message=f"Wrote Agmarknet snapshot mode={payload.get('mode')} count={payload.get('count')}",
                )
            ]
        except Exception as e:
            return [
                IngestFileResult(
                    source_id=src.id,
                    domain=src.domain,
                    action="error",
                    message=str(e),
                )
            ]


def lake_tree_summary(max_files_per_domain: int = 20) -> dict[str, Any]:
    """List files currently under lake/raw for demo/status."""
    ensure_lake_layout()
    tree: dict[str, list[dict[str, Any]]] = {}
    total = 0
    for domain in LAKE_DOMAINS:
        dpath = LAKE_RAW / domain
        files = []
        if dpath.exists():
            for p in sorted(dpath.rglob("*")):
                if p.is_file() and p.name not in (".gitkeep",) and not p.name.endswith(".meta.json"):
                    files.append(
                        {
                            "path": relative_to_repo(p),
                            "bytes": p.stat().st_size,
                        }
                    )
                    total += 1
                    if len(files) >= max_files_per_domain:
                        break
        if files:
            tree[domain] = files
    latest = LAKE_ROOT / "INGEST_LATEST.json"
    return {
        "lake_raw": relative_to_repo(LAKE_RAW),
        "file_count": total,
        "domains_with_data": list(tree.keys()),
        "tree": tree,
        "latest_ingest": relative_to_repo(latest) if latest.exists() else None,
    }
