"""W-QASYNTH — expert QA synthesis factory (Sprint 6)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.lake.qa_synth import run_qa_synth
from mini.workers.base import BaseWorker, register_worker


@register_worker
class QASynthWorker(BaseWorker):
    worker_id = "W-QASYNTH"
    name = "QA Synthesis"
    description = "Generate multilingual expert QA packs (≥10k train / ≥1k val) from facts + taxonomy"
    epic = "E2"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        target = int(kwargs.get("target_min_total", 12000))
        report = run_qa_synth(dry_run=dry_run, target_min_total=target)
        counts = report.get("counts") or {}
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(report.get("ok")),
            dry_run=dry_run,
            message=(
                f"Synth QA total={counts.get('total', 0)} "
                f"train={counts.get('train', 0)} val={counts.get('val', 0)} "
                f"test={counts.get('test', 0)} version={report.get('version')}"
            ),
            artifacts=report.get("artifacts") or [],
            metrics=report,
            errors=[]
            if report.get("ok")
            else [
                f"Targets not met: train={counts.get('train')} (need ≥10000), "
                f"val={counts.get('val')} (need ≥1000)"
            ],
        )
