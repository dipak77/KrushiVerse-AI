"""Worker DAG definitions for Mini factory pipelines."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from mini.contracts import PipelineResult, WorkerResult
from mini.paths import EVENTS_LOG, ORCHESTRATOR_DIR, ensure_lake_layout, run_dir
from mini.workers.base import get_worker, list_workers

# Named pipelines: ordered worker IDs
PIPELINES: dict[str, list[str]] = {
    "bootstrap": ["W-BOOTSTRAP"],
    "taxonomy": ["W-TAXONOMY"],
    "sprint1": ["W-BOOTSTRAP", "W-TAXONOMY", "W-NORMALIZE"],
    "sprint2": ["W-BOOTSTRAP", "W-TAXONOMY", "W-INGEST"],
    "sprint3": ["W-BOOTSTRAP", "W-INGEST", "W-VALIDATE", "W-CLEAN", "W-DEDUP"],
    "sprint4": [
        "W-BOOTSTRAP",
        "W-INGEST",
        "W-VALIDATE",
        "W-CLEAN",
        "W-DEDUP",
        "W-NORMALIZE",
        "W-LANGDETECT",
        "W-STANDARD",
    ],
    "standardize": ["W-STANDARDIZE"],
    "sprint5": ["W-STANDARDIZE", "W-ANALYZE"],
    "sprint6": ["W-QUALITY", "W-QASYNTH", "W-ANALYZE"],
    "sprint7": ["W-QUALITY", "W-QASYNTH", "W-ANALYZE"],
    "sprint8": ["W-STANDARDIZE", "W-KGBUILD", "W-ANALYZE"],
    "sprint9": ["W-TOKEN"],
    "sprint10": ["W-PRETRAIN"],
    "sprint11": ["W-PRETRAIN"],
    "pretrain": ["W-PRETRAIN"],
    "token": ["W-TOKEN"],
    "kgbuild": ["W-KGBUILD"],
    "qasynth": ["W-QASYNTH"],
    "analyze": ["W-ANALYZE"],
    "quality": ["W-QUALITY"],
    "ingest": ["W-BOOTSTRAP", "W-INGEST"],
    "dry-factory": [
        "W-BOOTSTRAP",
        "W-TAXONOMY",
        "W-INGEST",
        "W-VALIDATE",
        "W-CLEAN",
        "W-DEDUP",
        "W-NORMALIZE",
        "W-LANGDETECT",
        "W-STANDARD",
        "W-ANALYZE",
        "W-QASYNTH",
        "W-KGBUILD",
        "W-TOKEN",
        "W-PRETRAIN",
        "W-SFT",
        "W-EVAL",
        "W-QUANT",
        "W-DEPLOY",
        "W-RAG",
        "W-AGENT",
        "W-INFER",
    ],
    "ingest-pipeline": ["W-BOOTSTRAP", "W-INGEST", "W-VALIDATE", "W-CLEAN", "W-DEDUP"],
    "train": ["W-TOKEN", "W-PRETRAIN", "W-SFT", "W-EVAL"],
    "eval": ["W-EVAL"],
    "full": [
        "W-BOOTSTRAP",
        "W-TAXONOMY",
        "W-INGEST",
        "W-VALIDATE",
        "W-CLEAN",
        "W-DEDUP",
        "W-NORMALIZE",
        "W-LANGDETECT",
        "W-STANDARD",
        "W-ANALYZE",
        "W-QASYNTH",
        "W-KGBUILD",
        "W-TOKEN",
        "W-PRETRAIN",
        "W-SFT",
        "W-EVAL",
        "W-QUANT",
        "W-DEPLOY",
    ],
}


def _append_event(event: dict[str, Any]) -> None:
    ORCHESTRATOR_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_pipeline(
    pipeline: str,
    *,
    dry_run: bool = True,
    stop_on_error: bool = True,
) -> PipelineResult:
    """Run a named pipeline of workers in order."""
    if pipeline not in PIPELINES:
        raise KeyError(f"Unknown pipeline '{pipeline}'. Known: {sorted(PIPELINES)}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    worker_ids = PIPELINES[pipeline]
    steps: list[WorkerResult] = []
    ok = True

    if not dry_run:
        ensure_lake_layout()
        rd = run_dir(run_id)
    else:
        rd = None

    _append_event(
        {
            "event": "pipeline_start",
            "pipeline": pipeline,
            "run_id": run_id,
            "dry_run": dry_run,
            "workers": worker_ids,
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )

    for wid in worker_ids:
        worker = get_worker(wid)
        result = worker.run(dry_run=dry_run)
        steps.append(result)
        _append_event(
            {
                "event": "worker_finished",
                "pipeline": pipeline,
                "run_id": run_id,
                "worker_id": wid,
                "ok": result.ok,
                "dry_run": dry_run,
                "message": result.message,
                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        if not result.ok:
            ok = False
            if stop_on_error:
                break

    if rd is not None:
        summary_path = rd / "pipeline_result.json"
        summary = PipelineResult(
            pipeline=pipeline,
            ok=ok,
            dry_run=dry_run,
            run_id=run_id,
            steps=steps,
            message="ok" if ok else "failed",
        )
        summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    _append_event(
        {
            "event": "pipeline_end",
            "pipeline": pipeline,
            "run_id": run_id,
            "ok": ok,
            "dry_run": dry_run,
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )

    return PipelineResult(
        pipeline=pipeline,
        ok=ok,
        dry_run=dry_run,
        run_id=run_id,
        steps=steps,
        message="Pipeline completed successfully" if ok else "Pipeline failed",
    )


def describe_factory() -> dict[str, Any]:
    from mini import __feature_phase__, __sprint__, __version__
    from mini.paths import LAKE_ROOT, SCHEMA_VERSION

    return {
        "mini_version": __version__,
        "sprint": __sprint__,
        "feature_phase": __feature_phase__,
        "schema_version": SCHEMA_VERSION,
        "lake_root": str(LAKE_ROOT),
        "pipelines": {k: v for k, v in PIPELINES.items()},
        "workers": list_workers(),
        "worker_count": len(list_workers()),
    }
