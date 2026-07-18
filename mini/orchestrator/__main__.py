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


def cmd_kgbuild(args: argparse.Namespace) -> int:
    from mini.workers.base import get_worker

    result = get_worker("W-KGBUILD").run(
        dry_run=not args.execute,
        write_platform_seed=bool(args.write_seed),
        include_districts=not args.no_districts,
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.ok else 1


def cmd_token(args: argparse.Namespace) -> int:
    from mini.workers.base import get_worker

    result = get_worker("W-TOKEN").run(
        dry_run=not args.execute,
        vocab_size=args.vocab_size,
        version=args.version,
        train_baseline=not args.no_baseline,
        max_qa_lines=args.max_qa,
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.ok else 1


def cmd_pretrain(args: argparse.Namespace) -> int:
    from mini.workers.base import get_worker

    result = get_worker("W-PRETRAIN").run(
        dry_run=not args.execute,
        mode=args.mode,
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        seed=args.seed,
        vocab_size=args.vocab_size,
        max_qa=args.max_qa,
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.ok else 1


def cmd_sft(args: argparse.Namespace) -> int:
    from mini.workers.base import get_worker

    result = get_worker("W-SFT").run(
        dry_run=not args.execute,
        steps_v03=args.steps_v03,
        steps_v04=args.steps_v04,
        batch_size=args.batch_size,
        seed=args.seed,
        max_train=args.max_train,
        max_val=args.max_val,
        lr=args.lr,
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.ok else 1


def cmd_eval(args: argparse.Namespace) -> int:
    """Run W-EVAL; non-zero exit when gates fail (acceptance)."""
    from mini.workers.base import get_worker

    result = get_worker("W-EVAL").run(
        dry_run=not args.execute,
        version=args.version,
        gate_profile=args.profile,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        max_gold=args.max_gold,
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.ok else 1


def cmd_quant(args: argparse.Namespace) -> int:
    from mini.workers.base import get_worker

    result = get_worker("W-QUANT").run(
        dry_run=not args.execute,
        version=args.version,
        include_int4=not args.no_int4,
        seed=args.seed,
        latency_runs=args.latency_runs,
    )
    print(result.model_dump_json(indent=2))
    return 0 if result.ok else 1


def cmd_deploy(args: argparse.Namespace) -> int:
    from mini.workers.base import get_worker

    result = get_worker("W-DEPLOY").run(
        dry_run=not args.execute,
        source_version=args.version,
        tag=args.tag,
        force=args.force,
        include_quant=not args.no_quant,
        reasoning_lite=not args.no_reasoning_lite,
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

    s = sub.add_parser("qasynth", help="Synthesize expert QA packs (W-QASYNTH, S7 ≥50k train)")
    s.add_argument("--execute", action="store_true", help="Write synth dataset version + review CSV")
    s.add_argument("--target", type=int, default=62500, help="Minimum total synth records target")
    s.set_defaults(func=cmd_qasynth)

    s = sub.add_parser("kgbuild", help="Build agri knowledge graph (W-KGBUILD, S8 ≥200 nodes / ≥400 edges)")
    s.add_argument("--execute", action="store_true", help="Write KG_LATEST + GraphML + triples")
    s.add_argument(
        "--write-seed",
        action="store_true",
        help="Also rewrite data/knowledge_graph.json for platform GraphRAG (local only)",
    )
    s.add_argument("--no-districts", action="store_true", help="Skip district GROWN_IN expansion")
    s.set_defaults(func=cmd_kgbuild)

    s = sub.add_parser("token", help="Train domain SentencePiece tokenizer (W-TOKEN, S9 30–50k)")
    s.add_argument("--execute", action="store_true", help="Write tokenizer/v0.1 artifacts")
    s.add_argument("--vocab-size", type=int, default=32000, help="Vocab size (30k–50k)")
    s.add_argument("--version", default="v0.1", help="Tokenizer version tag")
    s.add_argument("--no-baseline", action="store_true", help="Skip generic baseline fertility model")
    s.add_argument("--max-qa", type=int, default=80000, help="Max QA lines for corpus")
    s.set_defaults(func=cmd_token)

    s = sub.add_parser("pretrain", help="Domain pretrain Mini v0.2-base (W-PRETRAIN, S11)")
    s.add_argument("--execute", action="store_true", help="Train + write local v0.2-base checkpoint")
    s.add_argument("--mode", default="domain", choices=["domain", "smoke", "both"], help="Train mode")
    s.add_argument("--steps", type=int, default=200, help="Domain train steps")
    s.add_argument("--batch-size", type=int, default=8, help="Batch size")
    s.add_argument("--block-size", type=int, default=128, help="Packed context length")
    s.add_argument("--seed", type=int, default=42, help="RNG seed")
    s.add_argument("--vocab-size", type=int, default=4096, help="Model vocab size")
    s.add_argument("--max-qa", type=int, default=25000, help="Max QA lines for corpus")
    s.set_defaults(func=cmd_pretrain)

    s = sub.add_parser("sft", help="Instruction + agri-QA SFT (W-SFT, S12 v0.3/v0.4)")
    s.add_argument("--execute", action="store_true", help="Train + write local v0.3-instruct / v0.4-agri-qa")
    s.add_argument("--steps-v03", type=int, default=120, help="v0.3 instruct steps")
    s.add_argument("--steps-v04", type=int, default=120, help="v0.4 agri-QA steps")
    s.add_argument("--batch-size", type=int, default=4, help="Batch size")
    s.add_argument("--seed", type=int, default=42, help="RNG seed")
    s.add_argument("--max-train", type=int, default=4000, help="Max train SFT examples")
    s.add_argument("--max-val", type=int, default=400, help="Max val SFT examples")
    s.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    s.set_defaults(func=cmd_sft)

    s = sub.add_parser("eval", help="Evaluate Mini checkpoint with gates (W-EVAL, S13)")
    s.add_argument("--execute", action="store_true", help="Run eval + write HTML/JSON report")
    s.add_argument("--version", default="v0.4", help="Model version: v0.4, v0.3, v0.2")
    s.add_argument(
        "--profile",
        default="default",
        choices=["default", "strict", "prod", "promote"],
        help="Gate profile (strict may fail tiny models)",
    )
    s.add_argument("--seed", type=int, default=42, help="RNG seed")
    s.add_argument("--max-new-tokens", type=int, default=28, help="Generation length")
    s.add_argument("--max-gold", type=int, default=None, help="Optional cap on gold items")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("quant", help="Quantize Mini to INT8/INT4 (W-QUANT, S14)")
    s.add_argument("--execute", action="store_true", help="Write v0.5-quant artifacts + size report")
    s.add_argument("--version", default="v0.4", help="Source model version")
    s.add_argument("--no-int4", action="store_true", help="Skip INT4 pack")
    s.add_argument("--seed", type=int, default=42, help="RNG seed")
    s.add_argument("--latency-runs", type=int, default=6, help="CPU latency benchmark runs")
    s.set_defaults(func=cmd_quant)

    s = sub.add_parser("deploy", help="Package Mini version to serve/ + registry (W-DEPLOY, S14)")
    s.add_argument("--execute", action="store_true", help="Write serve package + VERSION_REGISTRY")
    s.add_argument("--version", default="v0.4", help="Source model version")
    s.add_argument("--tag", default="v0.5-quant", help="Registry/serve tag")
    s.add_argument("--force", action="store_true", help="Overwrite existing registry tag")
    s.add_argument("--no-quant", action="store_true", help="Skip bundling quant artifacts")
    s.add_argument("--no-reasoning-lite", action="store_true", help="Skip v0.5-reasoning-lite alias")
    s.set_defaults(func=cmd_deploy)

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
