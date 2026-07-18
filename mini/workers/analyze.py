"""W-ANALYZE worker — coverage intelligence dashboard (Sprint 5)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.lake.analyze import run_analysis
from mini.workers.base import BaseWorker, register_worker


@register_worker
class AnalyzeWorker(BaseWorker):
    worker_id = "W-ANALYZE"
    name = "Analyze"
    description = "Coverage/quality dashboard: missingness, balance, taxonomy gaps"
    epic = "E2"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        report = run_analysis(dry_run=dry_run)
        summary = report.get("summary") or {}
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(report.get("ok")),
            dry_run=dry_run,
            message=(
                f"Analyzed {summary.get('total_records', 0)} records; "
                f"gaps={summary.get('gap_count', 0)}; "
                f"dup_rate={summary.get('duplicate_rate_pct', 0)}%"
            ),
            artifacts=report.get("artifacts") or [],
            metrics=report,
            errors=[] if report.get("ok") else ["No standard records available to analyze"],
        )
