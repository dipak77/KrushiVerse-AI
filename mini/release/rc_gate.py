"""Release candidate gate: eval + checklist + load smoke + matrix (Sprint 17)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mini.paths import EVAL_DIR, INFERENCE_DIR, MODELS_DIR, REPO_ROOT, ensure_lake_layout, relative_to_repo
from mini.release.checklist import build_checklist
from mini.release.load_smoke import run_load_smoke
from mini.release.scale_roadmap import build_scale_report
from mini.release.version_matrix import build_version_matrix_report

RELEASE_DIR = REPO_ROOT / "mini" / "release"
RELEASE_LATEST = RELEASE_DIR / "RELEASE_LATEST.json"
CHECKLIST_PATH = RELEASE_DIR / "CHECKLIST_SIGNED.json"
MATRIX_PATH = RELEASE_DIR / "VERSION_MATRIX.json"
SCALE_PATH = RELEASE_DIR / "SCALE_ROADMAP.json"
RUNBOOK_PATH = RELEASE_DIR / "RUNBOOK.md"


def ensure_runbook() -> Path:
    """Write/update runbook markdown (idempotent content)."""
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    body = """# KrushiVerseAI Mini v1.0 Runbook

## Overview

Mini is a ~1M-parameter agriculture assistant integrated with platform RAG and agents.

| Flag | Default | Effect |
|---|---|---|
| `USE_MINI_LLM` | `false` | When `true`, planner uses Mini as synthesizer |
| `MINI_DEFAULT_MODE` | `grounded` | Requires sources for answers |
| `MINI_MODEL_VERSION` | `auto` | Prefer serve/v0.5-quant → v0.4 → v0.3 → v0.2 |

## Health checks

```bash
# API
curl http://127.0.0.1:8000/api/mini/status
curl -X POST http://127.0.0.1:8000/api/mini/chat -H "Content-Type: application/json" \\
  -d "{\\"query\\":\\"pink bollworm cotton IPM\\",\\"crop\\":\\"Cotton\\",\\"mode\\":\\"grounded\\"}"

# Factory release gate
python -m mini.orchestrator release --execute
```

## Rollback

1. Set `USE_MINI_LLM=false` and restart API/Streamlit → classic template synthesizer.
2. Point `MINI_MODEL_VERSION` to a previous local tag (`v0.4`, `v0.3`).
3. Redeploy prior `mini/models/serve/<tag>` package if needed.

## Incident tips

- **Empty sources / grounded refusal:** check lake/KB and `W-RAG`; mini-local KB still serves cotton IPM demos.
- **Slow CPU latency:** reduce `MINI_MAX_NEW_TOKENS`; use INT8 package when available.
- **Unsafe advice:** validator + fallback templates; re-run `W-EVAL` probes.

## Artifacts (local-only weights)

- Checkpoints: `mini/models/v0.2-base` … `v0.5-quant`, `serve/`
- Reports: `EVAL_LATEST`, `INFER_LATEST`, `RELEASE_LATEST`, `CHECKLIST_SIGNED`

## Owners

See `SCALE_ROADMAP.json` owners and plan §8 deferred list.
"""
    RUNBOOK_PATH.write_text(body, encoding="utf-8")
    return RUNBOOK_PATH


def run_release(
    *,
    dry_run: bool = False,
    run_eval: bool = True,
    run_smoke: bool = True,
    eval_version: str = "v0.4",
    smoke_rounds: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    ensure_lake_layout()
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "sprint": "S17",
            "feature_phase": "FP-10",
            "release": "v1.0",
            "planned": {"run_eval": run_eval, "run_smoke": run_smoke, "eval_version": eval_version},
        }

    ensure_runbook()
    matrix = build_version_matrix_report()
    scale = build_scale_report()
    checklist = build_checklist()

    MATRIX_PATH.write_text(json.dumps(matrix, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    SCALE_PATH.write_text(json.dumps(scale, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    CHECKLIST_PATH.write_text(
        json.dumps({**checklist, "signed_at": created}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    eval_report: dict[str, Any] | None = None
    if run_eval:
        try:
            from mini.eval.harness import run_eval as _run_eval

            eval_report = _run_eval(
                dry_run=False,
                version=eval_version,
                gate_profile="default",
                seed=seed,
                max_new_tokens=16,
                max_gold=10,
            )
        except Exception as e:
            eval_report = {"ok": False, "error": str(e)}

    smoke: dict[str, Any] | None = None
    if run_smoke:
        try:
            smoke = run_load_smoke(rounds=smoke_rounds, max_new_tokens=12)
        except Exception as e:
            smoke = {"ok": False, "error": str(e)}

    # RC gate: checklist no hard fails; eval ok if run; smoke ok if run
    parts_ok = [bool(checklist.get("ok"))]
    if run_eval and eval_report is not None:
        parts_ok.append(bool(eval_report.get("ok")))
    if run_smoke and smoke is not None:
        parts_ok.append(bool(smoke.get("ok")))
    ok = all(parts_ok)

    report = {
        "ok": ok,
        "dry_run": False,
        "sprint": "S17",
        "feature_phase": "FP-10",
        "release": "v1.0",
        "release_name": "KrushiVerseAI Mini v1.0",
        "created_at": created,
        "checklist": {
            "ok": checklist.get("ok"),
            "summary": checklist.get("summary"),
            "path": relative_to_repo(CHECKLIST_PATH),
        },
        "version_matrix": {
            "n_versions": len(matrix.get("versions") or []),
            "path": relative_to_repo(MATRIX_PATH),
        },
        "scale_roadmap": {
            "decision": scale.get("decision"),
            "path": relative_to_repo(SCALE_PATH),
        },
        "runbook": relative_to_repo(RUNBOOK_PATH),
        "eval_gate": {
            "ran": run_eval,
            "ok": None if eval_report is None else bool(eval_report.get("ok")),
            "version": eval_version,
            "f1": ((eval_report or {}).get("qa") or {}).get("token_f1"),
            "p95_ms": ((eval_report or {}).get("qa") or {}).get("latency_ms_p95"),
        },
        "load_smoke": {
            "ran": run_smoke,
            "ok": None if smoke is None else bool(smoke.get("ok")),
            "latency": (smoke or {}).get("latency_ms"),
            "n_ok": (smoke or {}).get("n_ok"),
            "n_calls": (smoke or {}).get("n_calls"),
        },
        "gates": {
            "checklist_ok": bool(checklist.get("ok")),
            "eval_ok": True if not run_eval else bool((eval_report or {}).get("ok")),
            "smoke_ok": True if not run_smoke else bool((smoke or {}).get("ok")),
        },
        "artifacts": [
            relative_to_repo(CHECKLIST_PATH),
            relative_to_repo(MATRIX_PATH),
            relative_to_repo(SCALE_PATH),
            relative_to_repo(RUNBOOK_PATH),
        ],
    }
    RELEASE_LATEST.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    report["artifacts"] = list(dict.fromkeys(report["artifacts"] + [relative_to_repo(RELEASE_LATEST)]))
    RELEASE_LATEST.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report
