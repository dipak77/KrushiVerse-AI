"""W-SFT — instruction + agri-QA fine-tuning (Sprint 12)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.models.sft import run_sft
from mini.workers.base import BaseWorker, register_worker


@register_worker
class SFTWorker(BaseWorker):
    worker_id = "W-SFT"
    name = "Supervised Fine-Tune"
    description = "Instruction + multilingual agri-QA SFT (S12: v0.3-instruct, v0.4-agri-qa)"
    epic = "E4"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        report = run_sft(
            dry_run=dry_run,
            steps_v03=int(kwargs.get("steps_v03") or kwargs.get("steps") or 120),
            steps_v04=int(kwargs.get("steps_v04") or kwargs.get("steps") or 120),
            batch_size=int(kwargs.get("batch_size") or 4),
            seed=int(kwargs.get("seed") or 42),
            max_train=int(kwargs.get("max_train") or 4000),
            max_val=int(kwargs.get("max_val") or 400),
            lr=float(kwargs.get("lr") or 2e-3),
        )
        base = report.get("base_val") or {}
        v4 = ((report.get("v0.4") or {}).get("stage") or {}).get("val") or {}
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(report.get("ok")),
            dry_run=dry_run,
            message=(
                f"SFT v0.3+v0.4 seed={report.get('seed')} "
                f"base_f1={base.get('token_f1')} → sft_f1={v4.get('token_f1')} "
                f"beats_base={((report.get('v0.4') or {}).get('beats_base'))} "
                f"train={((report.get('counts') or {}).get('train'))}"
            ),
            artifacts=report.get("artifacts") or [],
            metrics=report,
            errors=[]
            if report.get("ok")
            else [
                f"SFT acceptance failed: base_f1={base.get('token_f1')}, "
                f"sft_f1={v4.get('token_f1')}, beats_base={((report.get('v0.4') or {}).get('beats_base'))}"
            ],
        )
