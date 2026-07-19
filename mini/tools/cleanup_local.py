"""Free disk occupied by disposable Mini factory test artifacts.

Primary target: mini/datasets/versions/ (often multi-GB from repeated synth/KG runs).

Usage:
  python -m mini.tools.cleanup_local --dry-run
  python -m mini.tools.cleanup_local --execute --keep-synth 1 --keep-other 1
  python -m mini.tools.cleanup_local --execute --smoke-ckpts --runs
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mini.paths import DATASETS_DIR, MODELS_DIR, REPO_ROOT, RUNS_DIR, relative_to_repo


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _mb(n: int) -> float:
    return round(n / (1024 * 1024), 2)


def _pinned_version_names() -> set[str]:
    """Keep any version folder still referenced by LATEST markers."""
    pinned: set[str] = set()
    for name in ("LATEST_VERSION.json", "QASYNTH_LATEST.json", "KG_LATEST.json"):
        p = DATASETS_DIR / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        blob = json.dumps(data)
        versions = DATASETS_DIR / "versions"
        if versions.exists():
            for d in versions.iterdir():
                if d.is_dir() and d.name in blob:
                    pinned.add(d.name)
        # common keys
        for key in ("version", "version_id", "dataset_version", "path"):
            v = data.get(key)
            if isinstance(v, str):
                pinned.add(Path(v).name)
    return pinned


def plan_versions_cleanup(*, keep_synth: int = 1, keep_other: int = 1) -> dict[str, Any]:
    versions = DATASETS_DIR / "versions"
    if not versions.exists():
        return {"ok": True, "delete": [], "keep": [], "bytes_freeable": 0, "message": "no versions dir"}

    pinned = _pinned_version_names()
    synth = sorted(
        [d for d in versions.iterdir() if d.is_dir() and "synth" in d.name.lower()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    other = sorted(
        [d for d in versions.iterdir() if d.is_dir() and "synth" not in d.name.lower()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    keep: set[Path] = set()
    for d in synth[: max(0, keep_synth)]:
        keep.add(d)
    for d in other[: max(0, keep_other)]:
        keep.add(d)
    for d in list(synth) + list(other):
        if d.name in pinned:
            keep.add(d)

    delete = [d for d in list(synth) + list(other) if d not in keep]
    freeable = sum(_dir_size(d) for d in delete)
    return {
        "ok": True,
        "pinned": sorted(pinned),
        "keep": [relative_to_repo(d) for d in sorted(keep, key=lambda p: p.name)],
        "delete": [relative_to_repo(d) for d in sorted(delete, key=lambda p: p.name)],
        "n_delete": len(delete),
        "n_keep": len(keep),
        "bytes_freeable": freeable,
        "mb_freeable": _mb(freeable),
        "paths": delete,
    }


def plan_smoke_ckpts() -> dict[str, Any]:
    ckpt_dir = MODELS_DIR / "checkpoints"
    if not ckpt_dir.exists():
        return {"delete": [], "bytes_freeable": 0, "paths": []}
    paths = [p for p in ckpt_dir.glob("*smoke*") if p.is_file()]
    freeable = sum(_dir_size(p) for p in paths)
    return {
        "delete": [relative_to_repo(p) for p in paths],
        "n_delete": len(paths),
        "bytes_freeable": freeable,
        "mb_freeable": _mb(freeable),
        "paths": paths,
    }


def plan_runs() -> dict[str, Any]:
    if not RUNS_DIR.exists():
        return {"delete": [], "bytes_freeable": 0, "paths": []}
    paths = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    freeable = sum(_dir_size(p) for p in paths)
    return {
        "delete": [relative_to_repo(p) for p in paths],
        "n_delete": len(paths),
        "bytes_freeable": freeable,
        "mb_freeable": _mb(freeable),
        "paths": paths,
    }


def execute_deletes(paths: list[Path]) -> list[str]:
    removed: list[str] = []
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                p.unlink()
            removed.append(relative_to_repo(p))
        except Exception as e:
            removed.append(f"FAIL {p}: {e}")
    return removed


def run_cleanup(
    *,
    dry_run: bool = True,
    keep_synth: int = 1,
    keep_other: int = 1,
    smoke_ckpts: bool = True,
    runs: bool = False,
) -> dict[str, Any]:
    ver = plan_versions_cleanup(keep_synth=keep_synth, keep_other=keep_other)
    smoke = plan_smoke_ckpts() if smoke_ckpts else {"paths": [], "bytes_freeable": 0, "delete": []}
    runsp = plan_runs() if runs else {"paths": [], "bytes_freeable": 0, "delete": []}

    all_paths: list[Path] = list(ver.get("paths") or []) + list(smoke.get("paths") or []) + list(
        runsp.get("paths") or []
    )
    total = int(ver.get("bytes_freeable") or 0) + int(smoke.get("bytes_freeable") or 0) + int(
        runsp.get("bytes_freeable") or 0
    )

    report: dict[str, Any] = {
        "ok": True,
        "dry_run": dry_run,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions": {k: v for k, v in ver.items() if k != "paths"},
        "smoke_ckpts": {k: v for k, v in smoke.items() if k != "paths"},
        "runs": {k: v for k, v in runsp.items() if k != "paths"},
        "total_mb_freeable": _mb(total),
        "total_bytes_freeable": total,
        "removed": [],
    }

    if not dry_run:
        report["removed"] = execute_deletes(all_paths)
        report["ok"] = not any(str(x).startswith("FAIL") for x in report["removed"])

    out = DATASETS_DIR / "CLEANUP_LATEST.json"
    try:
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        # strip non-serializable
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        report["report_path"] = relative_to_repo(out)
    except Exception:
        pass
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cleanup disposable Mini local artifacts")
    p.add_argument("--execute", action="store_true", help="Actually delete (default dry-run)")
    p.add_argument("--keep-synth", type=int, default=1, help="Newest synth version folders to keep")
    p.add_argument("--keep-other", type=int, default=1, help="Newest non-synth version folders to keep")
    p.add_argument("--smoke-ckpts", action="store_true", default=True, help="Delete *smoke* checkpoints")
    p.add_argument("--no-smoke-ckpts", action="store_true", help="Skip smoke checkpoint cleanup")
    p.add_argument("--runs", action="store_true", help="Also delete mini/runs/*")
    args = p.parse_args(argv)

    report = run_cleanup(
        dry_run=not args.execute,
        keep_synth=args.keep_synth,
        keep_other=args.keep_other,
        smoke_ckpts=not args.no_smoke_ckpts,
        runs=args.runs,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if report.get("dry_run"):
        print(f"\n[dry-run] Would free ~{report.get('total_mb_freeable')} MB. Re-run with --execute.")
    else:
        print(f"\n[execute] Freed target ~{report.get('total_mb_freeable')} MB (see removed list).")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
