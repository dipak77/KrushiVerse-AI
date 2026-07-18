"""CLI entry: python -m mini.orchestrator <command>"""

from __future__ import annotations

import argparse
import json
import sys

from mini.orchestrator.dag import PIPELINES, describe_factory, run_pipeline
from mini.paths import ensure_lake_layout, relative_to_repo
from mini.workers.base import get_worker, list_workers


def cmd_list_workers(_: argparse.Namespace) -> int:
    workers = list_workers()
    print(f"KrushiVerseAI Mini workers ({len(workers)}):\n")
    print(f"{'ID':<16} {'STATUS':<10} {'EPIC':<6} NAME")
    print("-" * 72)
    for w in workers:
        print(f"{w['worker_id']:<16} {w['status']:<10} {w['epic']:<6} {w['name']}")
        print(f"  {w['description']}")
    return 0


def cmd_list_pipelines(_: argparse.Namespace) -> int:
    print("Available pipelines:\n")
    for name, steps in PIPELINES.items():
        print(f"  {name}: {' → '.join(steps)}")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    info = describe_factory()
    print(json.dumps(info, indent=2))
    return 0


def cmd_init_lake(_: argparse.Namespace) -> int:
    paths = ensure_lake_layout()
    print(f"Lake layout ready ({len(paths)} paths).")
    print(f"Root: {relative_to_repo(paths[0].parent) if paths else 'data/lake'}")
    return 0


def cmd_run_worker(args: argparse.Namespace) -> int:
    worker = get_worker(args.worker_id)
    result = worker.run(dry_run=args.dry_run)
    print(result.model_dump_json(indent=2))
    return 0 if result.ok else 1


def cmd_run_pipeline(args: argparse.Namespace) -> int:
    dry = not args.execute  # default dry-run unless --execute
    result = run_pipeline(args.pipeline, dry_run=dry)
    print(result.model_dump_json(indent=2))
    return 0 if result.ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mini.orchestrator",
        description="KrushiVerseAI Mini factory orchestrator (Sprint 0+)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list-workers", help="List all registered workers")
    s.set_defaults(func=cmd_list_workers)

    s = sub.add_parser("list-pipelines", help="List named pipelines")
    s.set_defaults(func=cmd_list_pipelines)

    s = sub.add_parser("status", help="Factory status JSON")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("init-lake", help="Create data lake directory layout")
    s.set_defaults(func=cmd_init_lake)

    s = sub.add_parser("run-worker", help="Run a single worker")
    s.add_argument("worker_id", help="e.g. W-BOOTSTRAP")
    s.add_argument("--dry-run", action="store_true", help="Do not write artifacts")
    s.set_defaults(func=cmd_run_worker)

    s = sub.add_parser("run", help="Run a named pipeline (default: dry-run)")
    s.add_argument(
        "pipeline",
        nargs="?",
        default="bootstrap",
        help=f"Pipeline name: {', '.join(PIPELINES)}",
    )
    s.add_argument(
        "--execute",
        action="store_true",
        help="Actually write artifacts (default is dry-run)",
    )
    s.set_defaults(func=cmd_run_pipeline)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
