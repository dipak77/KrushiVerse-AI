"""W-PRETRAIN — Mini architecture + train harness skeleton (Sprint 10)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.models.train import run_pretrain_skeleton
from mini.workers.base import BaseWorker, register_worker


@register_worker
class PretrainWorker(BaseWorker):
    worker_id = "W-PRETRAIN"
    name = "Pretrain"
    description = "Mini ~1M architecture + overfit smoke harness (S10 skeleton; full pretrain S11)"
    epic = "E4"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        steps = int(kwargs.get("overfit_steps", 50))
        batch = int(kwargs.get("batch_size", 8))
        seq = int(kwargs.get("seq_len", 64))
        report = run_pretrain_skeleton(
            dry_run=dry_run,
            overfit_steps=steps,
            batch_size=batch,
            seq_len=seq,
            save_ckpt=bool(kwargs.get("save_ckpt", True)),
        )
        params = report.get("parameters") or {}
        smoke = report.get("overfit_smoke") or {}
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(report.get("ok")),
            dry_run=dry_run,
            message=(
                f"Mini params={params.get('unique_params')} "
                f"({params.get('millions')}M) in_range={report.get('in_range')} "
                f"overfit {smoke.get('first_loss')}→{smoke.get('last_loss')} "
                f"dropped={smoke.get('loss_dropped')}"
            ),
            artifacts=report.get("artifacts") or [],
            metrics=report,
            errors=[]
            if report.get("ok")
            else [
                f"Param range or overfit smoke failed: in_range={report.get('in_range')}, "
                f"loss_dropped={smoke.get('loss_dropped')}"
            ],
        )
