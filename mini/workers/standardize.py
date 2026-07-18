"""W-NORMALIZE, W-LANGDETECT, W-STANDARD workers (Sprint 4)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.lake.langdetect import detect_language, detect_language_pair, language_distribution
from mini.lake.standardize import (
    coverage_stats,
    extract_standard_records_from_processed,
    export_standard_dataset,
    run_standardize_pipeline,
)
from mini.taxonomy.service import taxonomy_service
from mini.workers.base import BaseWorker, register_worker


@register_worker
class NormalizeWorker(BaseWorker):
    worker_id = "W-NORMALIZE"
    name = "Normalize"
    description = "Canonical crop/region/category via taxonomy on processed facts"
    epic = "E1"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        # Full batch normalize is embedded in extract_standard_records (crop resolve, category)
        # This worker reports normalize metrics on extracted records.
        records = extract_standard_records_from_processed()
        with_crop = sum(1 for r in records if r.crop)
        cats = {}
        for r in records:
            cats[r.category.value] = cats.get(r.category.value, 0) + 1
        sample = kwargs.get("text")
        sample_metrics = {}
        if sample:
            sample_metrics = {
                "sample_crop": taxonomy_service.resolve_crop(str(sample)),
                "sample_categories": taxonomy_service.detect_category(str(sample)),
            }
        return WorkerResult(
            worker_id=self.worker_id,
            ok=True,
            dry_run=dry_run,
            message=(
                f"Normalized {len(records)} candidate records; "
                f"with_crop={with_crop} taxonomy={taxonomy_service.version}"
            ),
            metrics={
                "records": len(records),
                "with_crop": with_crop,
                "by_category": cats,
                "taxonomy_version": taxonomy_service.version,
                **sample_metrics,
            },
        )


@register_worker
class LangDetectWorker(BaseWorker):
    worker_id = "W-LANGDETECT"
    name = "Language Detect"
    description = "Tag language mr/hi/en/mixed on standardized records"
    epic = "E1"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        text = kwargs.get("text")
        if text:
            lang = detect_language(str(text))
            return WorkerResult(
                worker_id=self.worker_id,
                ok=True,
                dry_run=dry_run,
                message=f"Detected language: {lang.value}",
                metrics={"language": lang.value, "text_preview": str(text)[:80]},
            )

        records = extract_standard_records_from_processed()
        dist = language_distribution([r.to_training_dict() for r in records])
        known = sum(v for k, v in dist.items() if k not in ("unknown",))
        total = sum(dist.values()) or 1
        return WorkerResult(
            worker_id=self.worker_id,
            ok=True,
            dry_run=dry_run,
            message=f"Language tags on {total} records; known={known}",
            metrics={
                "distribution": dist,
                "known_pct": round(100.0 * known / total, 2),
                "total": total,
            },
        )


@register_worker
class StandardWorker(BaseWorker):
    worker_id = "W-STANDARD"
    name = "Standardize"
    description = "Emit Schema v1 StandardRecord JSONL + parquet with train/val/test splits"
    epic = "E2"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        report = run_standardize_pipeline(dry_run=dry_run)
        cov = report.get("coverage") or {}
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(report.get("ok")),
            dry_run=dry_run,
            message=(
                f"Standard records total={report.get('counts', {}).get('total', 0)} "
                f"lang_pct={cov.get('language_pct')} cat_pct={cov.get('category_pct')} "
                f"version={report.get('version')}"
            ),
            artifacts=report.get("artifacts") or [],
            metrics=report,
            errors=[]
            if report.get("ok")
            else [
                f"Coverage below 90%: language={cov.get('language_pct')}% "
                f"category={cov.get('category_pct')}%"
            ],
        )


@register_worker
class StandardizePipelineWorker(BaseWorker):
    """Sprint 4 convenience: normalize metrics + lang detect + standard export."""

    worker_id = "W-STANDARDIZE"
    name = "Standardize Pipeline"
    description = "Normalize + language tag + export Schema v1 dataset version"
    epic = "E2"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        records = extract_standard_records_from_processed()
        stats = coverage_stats(records)
        export = export_standard_dataset(records, dry_run=dry_run)
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(export.get("ok")),
            dry_run=dry_run,
            message="Standardize pipeline complete",
            artifacts=export.get("artifacts") or [],
            metrics={
                "normalize_records": len(records),
                "coverage": stats,
                "export": export,
            },
        )
