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


# W-INGEST → mini.workers.ingest
# W-VALIDATE / W-CLEAN / W-DEDUP / W-QUALITY → mini.workers.quality


# W-NORMALIZE / W-LANGDETECT / W-STANDARD / W-STANDARDIZE → mini.workers.standardize


@register_worker
class TaxonomyWorker(BaseWorker):
    worker_id = "W-TAXONOMY"
    name = "Taxonomy Validate"
    description = "Validate frozen taxonomy integrity and platform KB coverage"
    epic = "E0"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        from mini.taxonomy.service import taxonomy_service
        import json
        from mini.paths import TAXONOMY_DIR, relative_to_repo

        report = taxonomy_service.validate()
        artifacts: list[str] = []
        if not dry_run:
            TAXONOMY_DIR.mkdir(parents=True, exist_ok=True)
            out = TAXONOMY_DIR / "validation_report.json"
            out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            artifacts.append(relative_to_repo(out))
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(report.get("ok")),
            dry_run=dry_run,
            message="Taxonomy validation " + ("passed" if report.get("ok") else "FAILED"),
            artifacts=artifacts,
            metrics={
                "version": report.get("taxonomy_version"),
                "integrity_ok": report.get("integrity", {}).get("ok"),
                "coverage_ok": report.get("platform_coverage", {}).get("ok"),
                "errors": (report.get("integrity", {}).get("errors") or [])
                + (report.get("platform_coverage", {}).get("errors") or []),
            },
            errors=(report.get("integrity", {}).get("errors") or [])
            + (report.get("platform_coverage", {}).get("errors") or []),
        )


# W-ANALYZE → mini.workers.analyze


# W-QASYNTH → mini.workers.qa_synth
# W-KGBUILD → mini.workers.kg_build


# W-TOKEN → mini.workers.token


# W-PRETRAIN → mini.workers.pretrain
# W-SFT → mini.workers.sft
# W-EVAL → mini.workers.eval


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


# Import real workers so they register into WORKER_REGISTRY
from mini.workers.ingest import IngestWorker  # noqa: E402,F401
from mini.workers.quality import (  # noqa: E402,F401
    CleanWorker,
    DedupWorker,
    QualityPipelineWorker,
    ValidateWorker,
)
from mini.workers.standardize import (  # noqa: E402,F401
    LangDetectWorker,
    NormalizeWorker,
    StandardWorker,
    StandardizePipelineWorker,
)
from mini.workers.analyze import AnalyzeWorker  # noqa: E402,F401
from mini.workers.qa_synth import QASynthWorker  # noqa: E402,F401
from mini.workers.kg_build import KGBuildWorker  # noqa: E402,F401
from mini.workers.token import TokenizerWorker  # noqa: E402,F401
from mini.workers.pretrain import PretrainWorker  # noqa: E402,F401
from mini.workers.sft import SFTWorker  # noqa: E402,F401
from mini.workers.eval import EvalWorker  # noqa: E402,F401


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
