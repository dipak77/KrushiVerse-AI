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
    description = "Generate multilingual expert QA packs (S7: ≥50k train, ≥8 categories, ≥20% non-EN)"
    epic = "E2"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        target = int(kwargs.get("target_min_total", 62500))
        report = run_qa_synth(dry_run=dry_run, target_min_total=target)
        counts = report.get("counts") or {}
        tm = report.get("targets_met") or {}
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(report.get("ok")),
            dry_run=dry_run,
            message=(
                f"Synth QA total={counts.get('total', 0)} "
                f"train={counts.get('train', 0)} val={counts.get('val', 0)} "
                f"non_en={report.get('non_english_pct')}% "
                f"cats={len(report.get('by_category') or {})} version={report.get('version')}"
            ),
            artifacts=report.get("artifacts") or [],
            metrics=report,
            errors=[]
            if report.get("ok")
            else [
                f"Targets not met: train={counts.get('train')} (need ≥50000), "
                f"val={counts.get('val')} (need ≥1000), "
                f"categories={tm.get('categories')}, non_en={tm.get('non_english_pct')}"
            ],
        )
