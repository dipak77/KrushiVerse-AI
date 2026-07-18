"""W-PRETRAIN — Mini domain pretraining (Sprint 11 / FP-6)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.models.pretrain import run_pretrain_s11
from mini.workers.base import BaseWorker, register_worker


@register_worker
class PretrainWorker(BaseWorker):
    worker_id = "W-PRETRAIN"
    name = "Pretrain"
    description = "Domain pretrain Mini ~1M on agri text (S11 v0.2-base; val PPL + seed)"
    epic = "E4"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        mode = str(kwargs.get("mode") or "domain")
        steps = int(kwargs.get("steps") or kwargs.get("overfit_steps") or 200)
        batch = int(kwargs.get("batch_size") or 8)
        block = int(kwargs.get("block_size") or kwargs.get("seq_len") or 128)
        seed = int(kwargs.get("seed") or 42)
        vocab = int(kwargs.get("vocab_size") or 4096)
        max_qa = int(kwargs.get("max_qa") or 25_000)

        report = run_pretrain_s11(
            dry_run=dry_run,
            mode=mode,
            steps=steps,
            batch_size=batch,
            block_size=block,
            vocab_size=vocab,
            seed=seed,
            overfit_steps=int(kwargs.get("smoke_steps") or 30),
            max_qa=max_qa,
        )
        domain = report.get("domain") or {}
        val = domain.get("val") or {}
        train = domain.get("train") or {}
        smoke = report.get("overfit_smoke") or {}
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(report.get("ok")),
            dry_run=dry_run,
            message=(
                f"S11 pretrain mode={mode} seed={seed} steps={steps} "
                f"train_loss {train.get('first_loss')}→{train.get('last_loss')} "
                f"val_ppl {val.get('ppl_start', {}).get('ppl')}→{val.get('ppl_end', {}).get('ppl')} "
                f"improved={val.get('ppl_improved')} ckpt={domain.get('checkpoint')}"
            ),
            artifacts=report.get("artifacts") or [],
            metrics=report,
            errors=[]
            if report.get("ok")
            else [
                f"Pretrain targets not met: ppl_improved={val.get('ppl_improved')}, "
                f"smoke={smoke.get('loss_dropped')}, in_range={report.get('in_range')}"
            ],
        )
