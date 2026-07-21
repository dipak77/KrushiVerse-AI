"""Emit task & subtask progress, records processed/pending, and ETAs for terminal status reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from factory.monitor import collect_status
from factory.state import utc_now


def detailed_report(factory_dir: str | Path = "factory") -> str:
    status = collect_status(factory_dir)
    tasks = status.get("tasks") or []
    summary = status.get("summary") or {}
    
    total_tasks = len(tasks)
    completed_tasks = summary.get("COMPLETED", 0)
    overall_pct = (completed_tasks / total_tasks * 100.0) if total_tasks else 0.0

    lines = []
    lines.append(f"[{utc_now()}] KRUSHIVERSE-AI v3 AUTONOMOUS FACTORY PROGRESS REPORT")
    lines.append("=" * 110)
    lines.append(
        f"Overall Factory Progress: {completed_tasks}/{total_tasks} Tasks ({overall_pct:.1f}%) | "
        f"Total Processed Records/Steps: {status.get('total_processed_records', 0):,} | "
        f"Total Pending Records/Steps: {status.get('total_pending_records', 0):,}"
    )
    lines.append("-" * 110)
    lines.append(f"{'SPRINT':<7} {'TASK ID':<15} {'STATUS':<11} {'PROGRESS (%)':<13} {'PROCESSED':<12} {'PENDING':<12} {'ETA / DURATION':<20} {'SUBTASKS'}")
    lines.append("-" * 110)

    for t in tasks:
        sprint = str(t.get("sprint", "S18"))
        tid = str(t.get("id"))
        st = str(t.get("status"))
        pct = f"{t.get('progress_pct', 0.0):.1f}%"
        proc = f"{t.get('processed_records', 0):,}"
        pend = f"{t.get('pending_records', 0):,}"
        eta = str(t.get("eta", "N/A"))
        subtasks = " -> ".join(t.get("subtasks") or [])

        lines.append(f"{sprint:<7} {tid:<15} {st:<11} {pct:<13} {proc:<12} {pend:<12} {eta:<20} {subtasks}")

    lines.append("=" * 110)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print autonomous factory status update")
    parser.add_argument("--factory-dir", default="factory")
    args = parser.parse_args(argv)
    print(detailed_report(args.factory_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
