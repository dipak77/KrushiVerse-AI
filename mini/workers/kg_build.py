"""W-KGBUILD — automated knowledge graph builder (Sprint 8)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.lake.kg_build import run_kg_build
from mini.workers.base import BaseWorker, register_worker


@register_worker
class KGBuildWorker(BaseWorker):
    worker_id = "W-KGBUILD"
    name = "Knowledge Graph Builder"
    description = "Build/update agri knowledge graph from standard entities (S8: ≥200 nodes, ≥400 edges)"
    epic = "E3"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        write_seed = bool(kwargs.get("write_platform_seed", False))
        include_districts = bool(kwargs.get("include_districts", True))
        report = run_kg_build(
            dry_run=dry_run,
            write_platform_seed=write_seed,
            include_districts=include_districts,
        )
        counts = report.get("counts") or {}
        tm = report.get("targets_met") or {}
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(report.get("ok")),
            dry_run=dry_run,
            message=(
                f"KG nodes={counts.get('nodes', 0)} edges={counts.get('edges', 0)} "
                f"version={report.get('version')} "
                f"(seed {report.get('seed_nodes')}/{report.get('seed_edges')})"
            ),
            artifacts=report.get("artifacts") or [],
            metrics=report,
            errors=[]
            if report.get("ok")
            else [
                f"Targets not met: nodes={counts.get('nodes')} (need ≥200), "
                f"edges={counts.get('edges')} (need ≥400); met={tm}"
            ],
        )
