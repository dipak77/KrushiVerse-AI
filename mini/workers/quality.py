"""W-VALIDATE / W-CLEAN / W-DEDUP workers (Sprint 3)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.lake.process import run_clean_stage, run_dedup_stage, run_quality_pipeline
from mini.lake.validate import run_validation
from mini.workers.base import BaseWorker, register_worker


@register_worker
class ValidateWorker(BaseWorker):
    worker_id = "W-VALIDATE"
    name = "Validate"
    description = "Schema/type validation of raw batches; quarantine invalid files"
    epic = "E1"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        quarantine = kwargs.get("quarantine", True)
        report = run_validation(dry_run=dry_run, quarantine=bool(quarantine))
        return WorkerResult(
            worker_id=self.worker_id,
            ok=True,  # validation findings are metrics; worker itself succeeded
            dry_run=dry_run,
            message=(
                f"Validated {report['files_scanned']} files: "
                f"valid={report['valid']} invalid={report['invalid']} "
                f"quarantined={len(report.get('quarantined') or [])}"
            ),
            metrics=report,
            errors=[],
        )


@register_worker
class CleanWorker(BaseWorker):
    worker_id = "W-CLEAN"
    name = "Clean"
    description = "HTML/boilerplate strip, unicode & whitespace normalization → processed/"
    epic = "E1"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        report = run_clean_stage(dry_run=dry_run, only_valid=True)
        return WorkerResult(
            worker_id=self.worker_id,
            ok=report.get("ok", False),
            dry_run=dry_run,
            message=(
                f"Cleaned={report.get('cleaned', 0)} skipped={report.get('skipped', 0)} "
                f"failed={report.get('failed', 0)}"
            ),
            metrics=report,
            errors=[
                d.get("error", "")
                for d in report.get("details") or []
                if d.get("action") == "error"
            ],
        )


@register_worker
class DedupWorker(BaseWorker):
    worker_id = "W-DEDUP"
    name = "Deduplicate"
    description = "Exact + near-duplicate removal on processed records"
    epic = "E1"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        thr = float(kwargs.get("near_threshold", 0.92))
        report = run_dedup_stage(dry_run=dry_run, near_threshold=thr)
        return WorkerResult(
            worker_id=self.worker_id,
            ok=report.get("ok", False),
            dry_run=dry_run,
            message=(
                f"Dedup files={report.get('files', 0)} exact_removed={report.get('exact_removed', 0)} "
                f"near_removed={report.get('near_removed', 0)}"
            ),
            metrics=report,
        )


@register_worker
class QualityPipelineWorker(BaseWorker):
    """Convenience worker: validate + clean + dedup with quality report."""

    worker_id = "W-QUALITY"
    name = "Quality Pipeline"
    description = "Run validate → clean → dedup and write QUALITY_LATEST.json"
    epic = "E1"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        thr = float(kwargs.get("near_threshold", 0.92))
        quarantine = kwargs.get("quarantine", True)
        report = run_quality_pipeline(
            dry_run=dry_run,
            quarantine=bool(quarantine),
            near_threshold=thr,
        )
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(report.get("ok")),
            dry_run=dry_run,
            message="Quality pipeline complete",
            artifacts=report.get("report_paths") or [],
            metrics=report,
        )
