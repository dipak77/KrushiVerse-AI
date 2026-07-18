"""W-INGEST worker implementation (Sprint 2)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.lake.ingest import IngestEngine, lake_tree_summary
from mini.lake.registry import load_source_registry
from mini.paths import relative_to_repo
from mini.workers.base import BaseWorker, register_worker


@register_worker
class IngestWorker(BaseWorker):
    worker_id = "W-INGEST"
    name = "Ingest"
    description = "Pull sources into lake/raw/{domain}/ with manifests (idempotent)"
    epic = "E1"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        source_ids = kwargs.get("source_ids")
        include_http = kwargs.get("include_http", True)
        if isinstance(source_ids, str):
            source_ids = [s.strip() for s in source_ids.split(",") if s.strip()]

        engine = IngestEngine(load_source_registry())
        report = engine.run(
            dry_run=dry_run,
            source_ids=source_ids,
            include_http=bool(include_http),
        )
        tree = lake_tree_summary() if not dry_run else {"note": "skipped in dry_run"}

        return WorkerResult(
            worker_id=self.worker_id,
            ok=report.ok,
            dry_run=dry_run,
            message=(
                f"Ingest {'dry-run ' if dry_run else ''}complete: "
                f"copied={report.files_copied} skipped={report.files_skipped} failed={report.files_failed}"
            ),
            artifacts=report.manifest_paths,
            metrics={
                "run_id": report.run_id,
                "sources_considered": report.sources_considered,
                "files_copied": report.files_copied,
                "files_skipped": report.files_skipped,
                "files_failed": report.files_failed,
                "registry": engine.registry.summary(),
                "lake_tree": tree,
            },
            errors=report.errors,
        )
