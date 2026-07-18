"""W-AGENT — light specialist agent notes for Mini context (Sprint 15)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.inference.agents_wrap import collect_agent_notes
from mini.workers.base import BaseWorker, register_worker


@register_worker
class AgentWorker(BaseWorker):
    worker_id = "W-AGENT"
    name = "Agent Router"
    description = "Intent-selected specialist notes for Mini context (S15 light wrap)"
    epic = "E6"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        query = str(kwargs.get("query") or "Cotton pink bollworm management")
        intents = kwargs.get("intents")
        if isinstance(intents, str):
            intents = [intents]
        if dry_run:
            return WorkerResult(
                worker_id=self.worker_id,
                ok=True,
                dry_run=True,
                message=f"W-AGENT dry-run query={query[:80]}",
                metrics={"sprint": "S15", "dry_run": True},
            )
        pack = collect_agent_notes(
            query,
            intents=list(intents) if intents else ["disease"],
            crop=kwargs.get("crop") or "Cotton",
            location=str(kwargs.get("location") or "Pune"),
            enable=True,
        )
        ok = True  # agent wrap is best-effort; empty notes still ok
        return WorkerResult(
            worker_id=self.worker_id,
            ok=ok,
            dry_run=False,
            message=f"AGENT notes={len(pack.get('notes') or [])} crop={pack.get('crop')}",
            metrics={"sprint": "S15", "feature_phase": "FP-8", **pack},
        )
