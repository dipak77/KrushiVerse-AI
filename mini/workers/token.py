"""W-TOKEN — domain SentencePiece tokenizer trainer (Sprint 9)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.tokenizer.train import DEFAULT_VOCAB, run_tokenizer_train
from mini.workers.base import BaseWorker, register_worker


@register_worker
class TokenizerWorker(BaseWorker):
    worker_id = "W-TOKEN"
    name = "Tokenizer"
    description = "Train domain SentencePiece (v1 32k BPE or v2 8k unigram)"
    epic = "E4"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        vocab = int(kwargs.get("vocab_size", DEFAULT_VOCAB))
        version = str(kwargs.get("version") or "v0.1")
        train_baseline = bool(kwargs.get("train_baseline", True))
        max_qa = int(kwargs.get("max_qa_lines", 80_000))
        model_type = str(kwargs.get("model_type") or ("unigram" if version.startswith("v2") else "bpe"))
        report = run_tokenizer_train(
            dry_run=dry_run,
            vocab_size=vocab,
            version=version,
            train_baseline=train_baseline,
            max_qa_lines=max_qa,
            model_type=model_type,
        )
        train = report.get("train") or {}
        fert = report.get("fertility") or {}
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(report.get("ok")),
            dry_run=dry_run,
            message=(
                f"Tokenizer {report.get('version')} vocab={train.get('actual_vocab_size')} "
                f"fert_domain={fert.get('domain_mean')} fert_base={fert.get('baseline_mean')} "
                f"improved={fert.get('improved')}"
            ),
            artifacts=report.get("artifacts") or [],
            metrics=report,
            errors=[]
            if report.get("ok")
            else [
                f"Tokenizer targets not met: vocab={train.get('actual_vocab_size')} "
                f"(need 30–50k), fertility_improved={fert.get('improved')}"
            ],
        )
