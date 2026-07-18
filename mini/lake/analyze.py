"""Dataset coverage / quality analysis for the Mini factory (Sprint 5 / Phase 7).

Produces metrics after each standardize (or on-demand):
- missingness by field
- exact duplicate rate
- question/answer length histograms
- crop / language / category / split balance
- coverage gaps vs frozen taxonomy
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mini.lake.dedup import stable_hash
from mini.lake.ingest import lake_tree_summary
from mini.lake.standardize import extract_standard_records_from_processed
from mini.paths import (
    DATASETS_DIR,
    LAKE_ROOT,
    LAKE_TEST,
    LAKE_TRAINING,
    LAKE_VALIDATION,
    ensure_lake_layout,
    relative_to_repo,
)
from mini.taxonomy.domains import list_categories, list_crop_names_en, list_crop_stages
from mini.taxonomy.regions import list_mh_districts


def _load_jsonl_records() -> list[dict[str, Any]]:
    """Prefer exported standard records; fall back to live extraction."""
    paths = [
        LAKE_TRAINING / "standard_records.jsonl",
        LAKE_VALIDATION / "standard_records.jsonl",
        LAKE_TEST / "standard_records.jsonl",
        DATASETS_DIR / "LATEST_VERSION.json",
    ]
    records: list[dict[str, Any]] = []
    for p in (LAKE_TRAINING, LAKE_VALIDATION, LAKE_TEST):
        f = p / "standard_records.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if records:
        return records
    # live extract
    return [r.to_training_dict() for r in extract_standard_records_from_processed()]


def _histogram(values: list[int], bins: list[int] | None = None) -> dict[str, int]:
    bins = bins or [0, 50, 100, 200, 400, 800, 1600, 10_000]
    labels = []
    for i in range(len(bins) - 1):
        labels.append(f"{bins[i]}-{bins[i+1]-1}")
    labels.append(f"{bins[-1]}+")
    counts = {lab: 0 for lab in labels}
    for v in values:
        placed = False
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[labels[i]] += 1
                placed = True
                break
        if not placed:
            counts[labels[-1]] += 1
    return counts


def _percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    fields = [
        "id",
        "category",
        "subcategory",
        "crop",
        "language",
        "question",
        "answer",
        "source",
        "split",
        "schema_version",
        "region",
        "confidence",
        "verified",
    ]
    missing: dict[str, int] = {f: 0 for f in fields}
    q_lens: list[int] = []
    a_lens: list[int] = []
    by_lang: Counter[str] = Counter()
    by_cat: Counter[str] = Counter()
    by_crop: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_state: Counter[str] = Counter()
    hashes: list[str] = []
    confidences: list[float] = []

    for r in records:
        for f in fields:
            val = r.get(f)
            empty = val is None or val == "" or val == {} or val == []
            if empty:
                missing[f] += 1
        q = str(r.get("question") or "")
        a = str(r.get("answer") or "")
        q_lens.append(len(q))
        a_lens.append(len(a))
        by_lang[str(r.get("language") or "unknown")] += 1
        by_cat[str(r.get("category") or "unknown")] += 1
        crop = r.get("crop") or "(none)"
        by_crop[str(crop)] += 1
        by_split[str(r.get("split") or "unassigned")] += 1
        region = r.get("region") or {}
        if isinstance(region, dict):
            by_state[str(region.get("state") or "(none)")] += 1
        else:
            by_state["(none)"] += 1
        try:
            confidences.append(float(r.get("confidence") or 0))
        except (TypeError, ValueError):
            confidences.append(0.0)
        # exact dup fingerprint on Q+A
        hashes.append(stable_hash({"q": q.strip().lower(), "a": a.strip().lower()}))

    unique_hashes = len(set(hashes))
    exact_dups = total - unique_hashes if total else 0

    q_sorted = sorted(q_lens)
    a_sorted = sorted(a_lens)

    missing_pct = {
        f: round(100.0 * missing[f] / total, 2) if total else 0.0 for f in fields
    }

    return {
        "total_records": total,
        "missingness": missing,
        "missingness_pct": missing_pct,
        "duplicates": {
            "exact_qa_duplicates": exact_dups,
            "unique_qa": unique_hashes,
            "duplicate_rate_pct": round(100.0 * exact_dups / total, 2) if total else 0.0,
        },
        "length": {
            "question": {
                "min": q_sorted[0] if q_sorted else 0,
                "max": q_sorted[-1] if q_sorted else 0,
                "mean": round(sum(q_lens) / total, 1) if total else 0,
                "p50": _percentile(q_sorted, 0.5),
                "p90": _percentile(q_sorted, 0.9),
                "histogram": _histogram(q_lens),
            },
            "answer": {
                "min": a_sorted[0] if a_sorted else 0,
                "max": a_sorted[-1] if a_sorted else 0,
                "mean": round(sum(a_lens) / total, 1) if total else 0,
                "p50": _percentile(a_sorted, 0.5),
                "p90": _percentile(a_sorted, 0.9),
                "histogram": _histogram(a_lens),
            },
        },
        "balance": {
            "language": dict(by_lang.most_common()),
            "category": dict(by_cat.most_common()),
            "crop_top": dict(by_crop.most_common(25)),
            "crop_total_distinct": len(by_crop),
            "split": dict(by_split.most_common()),
            "state": dict(by_state.most_common()),
        },
        "confidence": {
            "mean": round(sum(confidences) / total, 3) if total else 0,
            "min": min(confidences) if confidences else 0,
            "max": max(confidences) if confidences else 0,
        },
    }


def taxonomy_coverage_gaps(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare dataset coverage to frozen taxonomy."""
    tax_crops = set(list_crop_names_en())
    tax_cats = set(list_categories())
    tax_stages = {s["id"] for s in list_crop_stages()}

    present_crops = {r.get("crop") for r in records if r.get("crop")}
    present_cats = {r.get("category") for r in records if r.get("category")}

    missing_crops = sorted(tax_crops - present_crops)
    extra_crops = sorted(present_crops - tax_crops)
    missing_cats = sorted(tax_cats - present_cats)
    # stages often not in StandardRecord yet — report as gap if never seen in subcategory
    present_subs = {r.get("subcategory") for r in records if r.get("subcategory")}
    missing_stages = sorted(tax_stages - present_subs)

    # district coverage is sparse by design
    mh = set(list_mh_districts())
    present_districts: set[str] = set()
    for r in records:
        reg = r.get("region") or {}
        if isinstance(reg, dict) and reg.get("district"):
            present_districts.add(str(reg["district"]))

    gaps = []
    for c in missing_crops:
        gaps.append({"type": "crop", "id": c, "severity": "medium"})
    for c in missing_cats:
        gaps.append({"type": "category", "id": c, "severity": "high"})
    for s in missing_stages[:20]:
        gaps.append({"type": "crop_stage_subcategory", "id": s, "severity": "low"})
    if "machinery" in missing_cats:
        pass  # already listed
    # finance often thin
    finance_count = sum(1 for r in records if r.get("category") == "finance")
    if finance_count < 5:
        gaps.append(
            {
                "type": "volume",
                "id": "finance",
                "severity": "medium",
                "detail": f"only {finance_count} finance records",
            }
        )

    return {
        "taxonomy_crops": len(tax_crops),
        "taxonomy_categories": len(tax_cats),
        "present_crops": len(present_crops),
        "present_categories": len(present_cats),
        "missing_crops": missing_crops,
        "extra_crops": extra_crops,
        "missing_categories": missing_cats,
        "missing_stage_subcategories_sample": missing_stages[:15],
        "mh_districts_catalog": len(mh),
        "districts_in_data": sorted(present_districts),
        "gaps": gaps,
        "gap_count": len(gaps),
    }


def lake_layer_stats() -> dict[str, Any]:
    raw = lake_tree_summary()
    processed = 0
    from mini.paths import LAKE_PROCESSED

    if LAKE_PROCESSED.exists():
        processed = sum(
            1
            for p in LAKE_PROCESSED.rglob("*.json")
            if p.is_file() and not p.name.endswith(".meta.json") and ".meta." not in p.name
        )
    return {
        "raw_files": raw.get("file_count", 0),
        "raw_domains": raw.get("domains_with_data") or [],
        "processed_json_files": processed,
        "has_training_jsonl": (LAKE_TRAINING / "standard_records.jsonl").exists(),
        "has_validation_jsonl": (LAKE_VALIDATION / "standard_records.jsonl").exists(),
        "has_test_jsonl": (LAKE_TEST / "standard_records.jsonl").exists(),
        "latest_dataset": relative_to_repo(DATASETS_DIR / "LATEST_VERSION.json")
        if (DATASETS_DIR / "LATEST_VERSION.json").exists()
        else None,
    }


def run_analysis(*, dry_run: bool = False) -> dict[str, Any]:
    """Full analysis report for W-ANALYZE."""
    ensure_lake_layout()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    records = _load_jsonl_records()
    record_metrics = analyze_records(records)
    gaps = taxonomy_coverage_gaps(records)
    layers = lake_layer_stats()

    # Source quality heuristic
    sources = Counter(str(r.get("source") or "unknown") for r in records)
    source_quality = {
        "distinct_sources": len(sources),
        "top_sources": dict(sources.most_common(15)),
    }

    report = {
        "run_id": run_id,
        "sprint": "S5",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": dry_run,
        "ok": record_metrics["total_records"] > 0,
        "lake_layers": layers,
        "records": record_metrics,
        "taxonomy_gaps": gaps,
        "source_quality": source_quality,
        "summary": {
            "total_records": record_metrics["total_records"],
            "language_balance": record_metrics["balance"]["language"],
            "category_balance": record_metrics["balance"]["category"],
            "gap_count": gaps["gap_count"],
            "duplicate_rate_pct": record_metrics["duplicates"]["duplicate_rate_pct"],
            "missing_crop_pct": record_metrics["missingness_pct"].get("crop", 0),
            "missing_language_pct": record_metrics["missingness_pct"].get("language", 0),
        },
    }

    artifacts: list[str] = []
    if not dry_run:
        reports = LAKE_ROOT / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        out = reports / f"analyze_{run_id}.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        latest = LAKE_ROOT / "ANALYZE_LATEST.json"
        latest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        # also under mini/datasets for factory visibility
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        ds_latest = DATASETS_DIR / "ANALYZE_LATEST.json"
        ds_latest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts = [
            relative_to_repo(out),
            relative_to_repo(latest),
            relative_to_repo(ds_latest),
        ]
        # simple HTML dashboard
        html_path = reports / f"analyze_{run_id}.html"
        html_path.write_text(_render_html_report(report), encoding="utf-8")
        html_latest = LAKE_ROOT / "ANALYZE_LATEST.html"
        html_latest.write_text(_render_html_report(report), encoding="utf-8")
        artifacts.extend([relative_to_repo(html_path), relative_to_repo(html_latest)])

    report["artifacts"] = artifacts
    return report


def _render_html_report(report: dict[str, Any]) -> str:
    s = report.get("summary") or {}
    gaps = (report.get("taxonomy_gaps") or {}).get("gaps") or []
    gap_rows = "".join(
        f"<tr><td>{g.get('type')}</td><td>{g.get('id')}</td><td>{g.get('severity')}</td>"
        f"<td>{g.get('detail', '')}</td></tr>"
        for g in gaps[:50]
    )
    lang = s.get("language_balance") or {}
    cat = s.get("category_balance") or {}
    lang_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in lang.items())
    cat_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in cat.items())
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>KrushiVerse Mini Analysis {report.get('run_id')}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;background:#0f1419;color:#e7ecf3}}
h1,h2{{color:#7dd87d}} table{{border-collapse:collapse;width:100%;margin:1rem 0}}
td,th{{border:1px solid #333;padding:.4rem .6rem;text-align:left}}
th{{background:#1a2332}} .card{{background:#1a2332;padding:1rem;border-radius:8px;margin:1rem 0}}
.muted{{color:#9aa7b8}}
</style></head><body>
<h1>KrushiVerseAI Mini — Data Analysis</h1>
<p class="muted">run_id={report.get('run_id')} · sprint={report.get('sprint')} · {report.get('finished_at')}</p>
<div class="card">
<h2>Summary</h2>
<ul>
<li>Total records: <b>{s.get('total_records')}</b></li>
<li>Duplicate rate: <b>{s.get('duplicate_rate_pct')}%</b></li>
<li>Missing crop %: <b>{s.get('missing_crop_pct')}</b></li>
<li>Missing language %: <b>{s.get('missing_language_pct')}</b></li>
<li>Taxonomy gaps: <b>{s.get('gap_count')}</b></li>
</ul>
</div>
<div class="card"><h2>Language balance</h2>
<table><tr><th>Language</th><th>Count</th></tr>{lang_rows}</table></div>
<div class="card"><h2>Category balance</h2>
<table><tr><th>Category</th><th>Count</th></tr>{cat_rows}</table></div>
<div class="card"><h2>Coverage gaps (sample)</h2>
<table><tr><th>Type</th><th>Id</th><th>Severity</th><th>Detail</th></tr>{gap_rows or '<tr><td colspan=4>None</td></tr>'}</table></div>
</body></html>"""
