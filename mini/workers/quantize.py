"""W-QUANT — INT8/INT4 export + size/latency benchmarks (Sprint 14)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.models.quantize import run_quantize
from mini.workers.base import BaseWorker, register_worker


@register_worker
class QuantizeWorker(BaseWorker):
    worker_id = "W-QUANT"
    name = "Quantize"
    description = "INT8/INT4 export, disk budgets, CPU latency p95 (S14)"
    epic = "E5"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        report = run_quantize(
            dry_run=dry_run,
            version=str(kwargs.get("version") or "v0.4"),
            include_int4=bool(kwargs.get("include_int4", True)),
            seed=int(kwargs.get("seed") or 42),
            int8_budget_bytes=int(kwargs.get("int8_budget_bytes") or 4 * 1024 * 1024),
            int4_budget_bytes=int(kwargs.get("int4_budget_bytes") or 2 * 1024 * 1024),
            latency_runs=int(kwargs.get("latency_runs") or 6),
        )
        cmp_ = report.get("comparison") or {}
        int8 = cmp_.get("int8") or {}
        fp32 = cmp_.get("fp32") or {}
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(report.get("ok")),
            dry_run=dry_run,
            message=(
                f"QUANT src={report.get('source_version')} "
                f"fp32={fp32.get('weight_mb')}MB → int8={int8.get('weight_mb')}MB "
                f"ratio={int8.get('ratio_vs_fp32')} "
                f"int8_ok={int8.get('within_budget')} "
                f"p95_int8={((int8.get('latency') or {}).get('p95_ms'))}ms"
            ),
            artifacts=report.get("artifacts") or [],
            metrics=report,
            errors=[]
            if report.get("ok")
            else [
                f"Quant acceptance failed: within_budget={int8.get('within_budget')} "
                f"smaller={((report.get('acceptance') or {}).get('int8_smaller_than_fp32'))}"
            ],
        )
