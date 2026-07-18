"""Register all Sprint-0 worker stubs (full implementations land in later sprints)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.paths import (
    LAKE_RAW,
    LAKE_PROCESSED,
    LAKE_TRAINING,
    ensure_lake_layout,
    relative_to_repo,
)
from mini.workers.base import BaseWorker, register_worker


def _stub_result(worker_id: str, dry_run: bool, message: str, **metrics: Any) -> WorkerResult:
    return WorkerResult(
        worker_id=worker_id,
        ok=True,
        dry_run=dry_run,
        message=message,
        metrics=metrics,
    )


@register_worker
class IngestWorker(BaseWorker):
    worker_id = "W-INGEST"
    name = "Ingest"
    description = "Pull sources into lake/raw/{domain}/ with manifests"
    epic = "E1"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        paths = ensure_lake_layout() if not dry_run else []
        return _stub_result(
            self.worker_id,
            dry_run,
            "Sprint 0 stub: lake layout ready; full ingest in Sprint 2",
            lake_paths=len(paths),
            raw_root=relative_to_repo(LAKE_RAW),
        )


@register_worker
class ValidateWorker(BaseWorker):
    worker_id = "W-VALIDATE"
    name = "Validate"
    description = "Schema/type validation of raw batches"
    epic = "E1"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(self.worker_id, dry_run, "Sprint 0 stub: validation scheduled Sprint 3")


@register_worker
class CleanWorker(BaseWorker):
    worker_id = "W-CLEAN"
    name = "Clean"
    description = "Text cleaning and encoding normalization"
    epic = "E1"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(self.worker_id, dry_run, "Sprint 0 stub: clean pipeline Sprint 3")


@register_worker
class DedupWorker(BaseWorker):
    worker_id = "W-DEDUP"
    name = "Deduplicate"
    description = "Exact and near-duplicate removal"
    epic = "E1"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(self.worker_id, dry_run, "Sprint 0 stub: dedup Sprint 3")


@register_worker
class NormalizeWorker(BaseWorker):
    worker_id = "W-NORMALIZE"
    name = "Normalize"
    description = "Canonical crop/region/units via taxonomy"
    epic = "E1"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(self.worker_id, dry_run, "Sprint 0 stub: normalize Sprint 4")


@register_worker
class LangDetectWorker(BaseWorker):
    worker_id = "W-LANGDETECT"
    name = "Language Detect"
    description = "Tag language mr/hi/en/mixed"
    epic = "E1"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(self.worker_id, dry_run, "Sprint 0 stub: language detect Sprint 4")


@register_worker
class StandardWorker(BaseWorker):
    worker_id = "W-STANDARD"
    name = "Standardize"
    description = "Emit Schema v1 StandardRecord parquet/JSONL"
    epic = "E2"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(
            self.worker_id,
            dry_run,
            "Sprint 0 stub: StandardRecord contract locked; emit in Sprint 4",
            schema_version="1.0",
            training_root=relative_to_repo(LAKE_TRAINING),
        )


@register_worker
class AnalyzeWorker(BaseWorker):
    worker_id = "W-ANALYZE"
    name = "Analyze"
    description = "Coverage and quality dashboards after ingest"
    epic = "E2"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(self.worker_id, dry_run, "Sprint 0 stub: analyze Sprint 5")


@register_worker
class QASynthWorker(BaseWorker):
    worker_id = "W-QASYNTH"
    name = "QA Synthesis"
    description = "Generate expert QA packs from structured facts"
    epic = "E2"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(self.worker_id, dry_run, "Sprint 0 stub: QA synth Sprint 6")


@register_worker
class KGBuildWorker(BaseWorker):
    worker_id = "W-KGBUILD"
    name = "Knowledge Graph Builder"
    description = "Build/update agri knowledge graph from standard entities"
    epic = "E3"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(self.worker_id, dry_run, "Sprint 0 stub: KG builder Sprint 8")


@register_worker
class TokenizerWorker(BaseWorker):
    worker_id = "W-TOKEN"
    name = "Tokenizer"
    description = "Train domain SentencePiece tokenizer 30–50k"
    epic = "E4"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(self.worker_id, dry_run, "Sprint 0 stub: tokenizer Sprint 9")


@register_worker
class PretrainWorker(BaseWorker):
    worker_id = "W-PRETRAIN"
    name = "Pretrain"
    description = "Domain pretraining of ~1M Mini base model"
    epic = "E4"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(self.worker_id, dry_run, "Sprint 0 stub: pretrain Sprint 11")


@register_worker
class SFTWorker(BaseWorker):
    worker_id = "W-SFT"
    name = "Supervised Fine-Tune"
    description = "Instruction and agri-QA fine-tuning"
    epic = "E4"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(self.worker_id, dry_run, "Sprint 0 stub: SFT Sprint 12")


@register_worker
class EvalWorker(BaseWorker):
    worker_id = "W-EVAL"
    name = "Evaluate"
    description = "Metrics, gates, hallucination probes"
    epic = "E5"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(self.worker_id, dry_run, "Sprint 0 stub: eval Sprint 13")


@register_worker
class QuantizeWorker(BaseWorker):
    worker_id = "W-QUANT"
    name = "Quantize"
    description = "INT8/INT4 export for deployment"
    epic = "E5"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(self.worker_id, dry_run, "Sprint 0 stub: quantize Sprint 14")


@register_worker
class DeployWorker(BaseWorker):
    worker_id = "W-DEPLOY"
    name = "Deploy"
    description = "Publish model version to serving path"
    epic = "E5"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(self.worker_id, dry_run, "Sprint 0 stub: deploy Sprint 16")


@register_worker
class RAGWorker(BaseWorker):
    worker_id = "W-RAG"
    name = "RAG Retriever"
    description = "Wrap platform advanced RAG for Mini context packs"
    epic = "E6"
    status = "partial"  # platform already has advanced RAG

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(
            self.worker_id,
            dry_run,
            "Partial: platform advanced_rag ready; Mini coupling Sprint 15",
            platform_module="app.knowledge.advanced_rag",
        )


@register_worker
class AgentWorker(BaseWorker):
    worker_id = "W-AGENT"
    name = "Agent Router"
    description = "Wrap platform planner agents as tool workers"
    epic = "E6"
    status = "partial"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(
            self.worker_id,
            dry_run,
            "Partial: platform planner agents ready; Mini synthesizer Sprint 16",
            platform_module="app.agents.planner",
        )


@register_worker
class InferWorker(BaseWorker):
    worker_id = "W-INFER"
    name = "Inference"
    description = "Intent → retrieve → Mini → validate → answer"
    epic = "E6"
    status = "stub"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        return _stub_result(self.worker_id, dry_run, "Sprint 0 stub: inference chain Sprint 15")


@register_worker
class BootstrapWorker(BaseWorker):
    """Sprint 0 concrete worker: ensure lake layout + write events."""

    worker_id = "W-BOOTSTRAP"
    name = "Bootstrap Factory"
    description = "Create lake dirs, verify contracts, write bootstrap marker"
    epic = "E0"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        from mini.paths import LAKE_RAW, LAKE_ROOT, ensure_lake_layout, relative_to_repo
        import json

        if dry_run:
            return WorkerResult(
                worker_id=self.worker_id,
                ok=True,
                dry_run=True,
                message="Dry-run: would create lake layout and bootstrap marker",
                metrics={"would_create_lake": True},
            )

        paths = ensure_lake_layout()
        marker = LAKE_ROOT / "BOOTSTRAP.json"
        payload = {
            "sprint": "S0",
            "feature_phase": "FP-0",
            "schema_version": "1.0",
            "message": "KrushiVerseAI Mini factory bootstrapped",
            "domains": sorted({p.name for p in LAKE_RAW.iterdir() if p.is_dir()}),
        }
        marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return WorkerResult(
            worker_id=self.worker_id,
            ok=True,
            dry_run=False,
            message="Lake layout ensured; bootstrap marker written",
            artifacts=[relative_to_repo(marker)],
            metrics={"paths_touched": len(paths), "domains": len(payload["domains"])},
        )
