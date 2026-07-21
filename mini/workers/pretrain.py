"""W-PRETRAIN — Mini domain pretraining (S11 v1 / S20 v2-15M)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.models.pretrain import run_pretrain_s11
from mini.workers.base import BaseWorker, register_worker


def _ppl(val: dict[str, Any], key: str) -> Any:
    block = val.get(key) or {}
    if isinstance(block, dict):
        return block.get("ppl")
    return None


@register_worker
class PretrainWorker(BaseWorker):
    worker_id = "W-PRETRAIN"
    name = "Pretrain"
    description = "Domain pretrain Mini (v1 ~1M → v0.2 | v2 ~15M → v0.6-base)"
    epic = "E4"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        variant = str(kwargs.get("variant") or "v1")
        is_v2 = variant.lower().startswith("v2") or variant.lower() in {"15m", "v0.6"}
        mode = str(kwargs.get("mode") or "domain")
        steps = int(kwargs.get("steps") or kwargs.get("overfit_steps") or (10_000 if is_v2 else 200))
        batch = int(kwargs.get("batch_size") or (4 if is_v2 else 8))
        block = int(kwargs.get("block_size") or kwargs.get("seq_len") or (512 if is_v2 else 128))
        seed = int(kwargs.get("seed") or 42)
        vocab = int(kwargs.get("vocab_size") or (8192 if is_v2 else 4096))
        max_qa = int(kwargs.get("max_qa") or 40_000)
        version = str(kwargs.get("version") or ("v0.6-base" if is_v2 else "v0.2-base"))

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
            variant="v2-15M" if is_v2 else "v1",
            version=version,
            grad_accum=int(kwargs.get("grad_accum") or (4 if is_v2 else 1)),
            use_fp16=bool(kwargs.get("fp16") or kwargs.get("use_fp16") or True),
            grad_checkpoint=bool(kwargs.get("grad_checkpoint") or kwargs.get("gradient_checkpointing") or True),
            lr=float(kwargs["lr"]) if kwargs.get("lr") is not None else None,
            eval_every=int(kwargs["eval_every"]) if kwargs.get("eval_every") is not None else None,
            config_path=kwargs.get("config") or kwargs.get("config_path"),
        )
        domain = report.get("domain") or {}
        val = domain.get("val") or {}
        train = domain.get("train") or {}
        smoke = report.get("overfit_smoke") or {}
        sprint = report.get("sprint") or "S11"
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(report.get("ok")),
            dry_run=dry_run,
            message=(
                f"{sprint} pretrain variant={report.get('variant') or variant} "
                f"seed={seed} steps={domain.get('steps') or steps} "
                f"loss {train.get('first_loss')}→{train.get('last_loss')} "
                f"ppl {_ppl(val, 'ppl_start')}→{_ppl(val, 'ppl_end')} "
                f"ckpt={domain.get('checkpoint')}"
            ),
            artifacts=report.get("artifacts") or [],
            metrics=report,
            errors=[]
            if report.get("ok")
            else [
                f"Pretrain targets not met: ppl_improved={val.get('ppl_improved')}, "
                f"loss_dropped={train.get('loss_dropped')}, smoke={smoke.get('loss_dropped')}"
            ],
        )
