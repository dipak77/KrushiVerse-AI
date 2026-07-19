"""W-RELEASE — Mini v1.0 RC gate + checklist sign-off (Sprint 17)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.release.rc_gate import run_release
from mini.workers.base import BaseWorker, register_worker


@register_worker
class ReleaseWorker(BaseWorker):
    worker_id = "W-RELEASE"
    name = "Release Gate"
    description = "v1.0 RC: checklist, eval gate, load smoke, version matrix (S17)"
    epic = "E6"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        report = run_release(
            dry_run=dry_run,
            run_eval=bool(kwargs.get("run_eval", True)),
            run_smoke=bool(kwargs.get("run_smoke", True)),
            eval_version=str(kwargs.get("eval_version") or "v0.4"),
            smoke_rounds=int(kwargs.get("smoke_rounds") or 2),
            seed=int(kwargs.get("seed") or 42),
        )
        ok = bool(report.get("ok")) if not dry_run else True
        gates = report.get("gates") or {}
        return WorkerResult(
            worker_id=self.worker_id,
            ok=ok,
            dry_run=dry_run,
            message=(
                f"RELEASE v1.0 ok={ok} checklist={gates.get('checklist_ok')} "
                f"eval={gates.get('eval_ok')} smoke={gates.get('smoke_ok')}"
            ),
            artifacts=report.get("artifacts") or [],
            metrics=report,
            errors=[] if ok else ["Release gate failed — see RELEASE_LATEST.json"],
        )
