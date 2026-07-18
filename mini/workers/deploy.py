"""W-DEPLOY — package-only publish to local serve path + registry (Sprint 14)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.models.deploy import run_deploy
from mini.workers.base import BaseWorker, register_worker


@register_worker
class DeployWorker(BaseWorker):
    worker_id = "W-DEPLOY"
    name = "Deploy"
    description = "Package Mini version: model card, license, registry (S14 package-only)"
    epic = "E5"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        report = run_deploy(
            dry_run=dry_run,
            source_version=str(kwargs.get("source_version") or kwargs.get("version") or "v0.4"),
            tag=str(kwargs.get("tag") or "v0.5-quant"),
            force=bool(kwargs.get("force") or False),
            include_quant=bool(kwargs.get("include_quant", True)),
            reasoning_lite=bool(kwargs.get("reasoning_lite", True)),
        )
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(report.get("ok")),
            dry_run=dry_run,
            message=(
                f"DEPLOY tag={report.get('tag')} "
                f"src={report.get('source_version')} "
                f"serve={report.get('serve_path')} "
                f"tags={report.get('tags_written')}"
            ),
            artifacts=report.get("artifacts") or [],
            metrics=report,
            errors=[] if report.get("ok") else [str(report.get("error") or "deploy package failed")],
        )
