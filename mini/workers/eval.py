"""W-EVAL — evaluation harness + promotion gates (Sprint 13)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.eval.harness import run_eval
from mini.workers.base import BaseWorker, register_worker


@register_worker
class EvalWorker(BaseWorker):
    worker_id = "W-EVAL"
    name = "Evaluate"
    description = "Gold QA + probes + gates scorecard (S13: HTML/JSON report, non-zero on fail)"
    epic = "E5"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        overrides = {}
        for k in (
            "min_token_f1",
            "min_rouge_l",
            "max_hallucination_rate",
            "max_latency_p95_ms",
            "max_val_loss",
        ):
            if kwargs.get(k) is not None:
                overrides[k] = kwargs[k]

        report = run_eval(
            dry_run=dry_run,
            version=str(kwargs.get("version") or "v0.4"),
            gate_profile=str(kwargs.get("gate_profile") or kwargs.get("profile") or "default"),
            seed=int(kwargs.get("seed") or 42),
            max_new_tokens=int(kwargs.get("max_new_tokens") or 28),
            max_gold=int(kwargs["max_gold"]) if kwargs.get("max_gold") is not None else None,
            gate_overrides=overrides or None,
        )
        gates = report.get("gates") or {}
        qa = report.get("qa") or {}
        ok = bool(report.get("ok")) if not dry_run else True
        failed = gates.get("failed_gates") or []
        return WorkerResult(
            worker_id=self.worker_id,
            ok=ok,
            dry_run=dry_run,
            message=(
                f"EVAL {report.get('version')} profile={report.get('gate_profile') or kwargs.get('gate_profile')} "
                f"ok={ok} f1={qa.get('token_f1')} rouge_l={qa.get('rouge_l')} "
                f"hall={((report.get('probes') or {}).get('hallucination_rate'))} "
                f"failed_gates={failed or 'none'}"
            ),
            artifacts=report.get("artifacts") or [],
            metrics=report,
            errors=[] if ok else [f"Gate failures: {', '.join(failed) or 'unknown'}"],
        )
