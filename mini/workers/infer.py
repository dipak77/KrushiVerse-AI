"""W-INFER — intent → retrieve → Mini → validate (Sprint 15 / FP-8)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.inference.pipeline import run_infer
from mini.workers.base import BaseWorker, register_worker


@register_worker
class InferWorker(BaseWorker):
    worker_id = "W-INFER"
    name = "Inference"
    description = "Intent → RAG → Mini → validate → answer (S15 grounded mode)"
    epic = "E6"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        report = run_infer(
            query=kwargs.get("query"),
            dry_run=dry_run,
            mode=str(kwargs.get("mode") or "grounded"),
            crop=kwargs.get("crop"),
            location=str(kwargs.get("location") or "Pune"),
            version=str(kwargs.get("version") or "auto"),
            enable_web=bool(kwargs.get("enable_web", False)),
            enable_tools=bool(kwargs.get("enable_tools", True)),
            enable_agents=bool(kwargs.get("enable_agents", True)),
            use_platform_rag=bool(kwargs.get("use_platform_rag", True)),
            max_new_tokens=int(kwargs.get("max_new_tokens") or 40),
            min_grounding=float(kwargs.get("min_grounding") or 0.08),
            seed=int(kwargs.get("seed") or 42),
            top_k=int(kwargs.get("top_k") or 6),
        )
        ok = bool(report.get("ok")) if not dry_run else True
        return WorkerResult(
            worker_id=self.worker_id,
            ok=ok,
            dry_run=dry_run,
            message=(
                f"INFER mode={report.get('mode')} engine={report.get('engine')} "
                f"sources={report.get('n_sources')} fallback={report.get('used_fallback')} "
                f"ok={ok}"
            ),
            artifacts=report.get("artifacts") or [],
            metrics=report,
            errors=[] if ok else list((report.get("validation") or {}).get("reasons") or ["infer_failed"]),
        )
