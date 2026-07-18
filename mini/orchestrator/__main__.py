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


def cmd_taxonomy_validate(_: argparse.Namespace) -> int:
    from mini.taxonomy.service import taxonomy_service

    report = taxonomy_service.validate()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("ok") else 1


def cmd_taxonomy_summary(_: argparse.Namespace) -> int:
    from mini.taxonomy.service import taxonomy_service

    print(json.dumps(taxonomy_service.summary(), indent=2, ensure_ascii=False))
    return 0


def cmd_sources(_: argparse.Namespace) -> int:
    from mini.lake.registry import load_source_registry

    reg = load_source_registry()
    print(json.dumps({"summary": reg.summary(), "sources": [s.model_dump() for s in reg.sources]}, indent=2))
    return 0


def cmd_lake_status(_: argparse.Namespace) -> int:
    from mini.lake.ingest import lake_tree_summary

    print(json.dumps(lake_tree_summary(), indent=2))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from mini.workers.base import get_worker

    kwargs = {"include_http": not args.skip_http}
    if args.sources:
        kwargs["source_ids"] = args.sources
    result = get_worker("W-INGEST").run(dry_run=not args.execute, **kwargs)
    print(result.model_dump_json(indent=2))
    return 0 if result.ok else 1


def cmd_quality(args: argparse.Namespace) -> int:
    from mini.workers.base import get_worker

    result = get_worker("W-QUALITY").run(
        dry_run=not args.execute,
        quarantine=not args.no_quarantine,
        near_threshold=args.near_threshold,
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.ok else 1


def cmd_standardize(args: argparse.Namespace) -> int:
    from mini.workers.base import get_worker

    result = get_worker("W-STANDARDIZE").run(dry_run=not args.execute)
    print(result.model_dump_json(indent=2))
    return 0 if result.ok else 1


def cmd_analyze(args: argparse.Namespace) -> int:
    from mini.workers.base import get_worker

    result = get_worker("W-ANALYZE").run(dry_run=not args.execute)
    print(result.model_dump_json(indent=2))
    return 0 if result.ok else 1


def cmd_qasynth(args: argparse.Namespace) -> int:
    from mini.workers.base import get_worker

    result = get_worker("W-QASYNTH").run(
        dry_run=not args.execute,
        target_min_total=args.target,
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.ok else 1


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

    s = sub.add_parser("taxonomy-validate", help="Validate frozen taxonomy + KB coverage")
    s.set_defaults(func=cmd_taxonomy_validate)

    s = sub.add_parser("taxonomy-summary", help="Print taxonomy summary counts")
    s.set_defaults(func=cmd_taxonomy_summary)

    s = sub.add_parser("sources", help="List source registry")
    s.set_defaults(func=cmd_sources)

    s = sub.add_parser("lake-status", help="Show lake/raw file tree summary")
    s.set_defaults(func=cmd_lake_status)

    s = sub.add_parser("ingest", help="Run W-INGEST (default dry-run; use --execute to write)")
    s.add_argument("--execute", action="store_true", help="Write files to lake/raw")
    s.add_argument("--sources", nargs="*", help="Optional source ids to ingest")
    s.add_argument("--skip-http", action="store_true", help="Skip http_api sources")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("quality", help="Run validate→clean→dedup (default dry-run)")
    s.add_argument("--execute", action="store_true", help="Write processed/quarantine/reports")
    s.add_argument("--no-quarantine", action="store_true", help="Do not copy invalid files")
    s.add_argument("--near-threshold", type=float, default=0.92, help="Near-dup Jaccard threshold")
    s.set_defaults(func=cmd_quality)

    s = sub.add_parser("standardize", help="Export Schema v1 train/val/test JSONL+parquet")
    s.add_argument("--execute", action="store_true", help="Write lake splits and dataset version")
    s.set_defaults(func=cmd_standardize)

    s = sub.add_parser("analyze", help="Coverage/quality analysis report (W-ANALYZE)")
    s.add_argument("--execute", action="store_true", help="Write ANALYZE_LATEST.json + HTML")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("qasynth", help="Synthesize expert QA packs (W-QASYNTH)")
    s.add_argument("--execute", action="store_true", help="Write synth dataset version + review CSV")
    s.add_argument("--target", type=int, default=12000, help="Minimum total synth records target")
    s.set_defaults(func=cmd_qasynth)

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
