"""Normalize processed lake facts → Schema v1 StandardRecords (Sprint 4)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from mini.contracts import Category, DataSplit, LanguageCode, Region, StandardRecord
from mini.lake.dedup import extract_record_lists
from mini.lake.langdetect import detect_language_pair
from mini.lake.process import iter_processed_json_files
from mini.paths import (
    DATASETS_DIR,
    LAKE_PROCESSED,
    LAKE_ROOT,
    LAKE_TEST,
    LAKE_TRAINING,
    LAKE_VALIDATION,
    SCHEMA_VERSION,
    ensure_lake_layout,
    relative_to_repo,
)
from mini.taxonomy.aliases import resolve_crop_name
from mini.taxonomy.service import taxonomy_service


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"std_{h}"


def _assign_split(record_id: str) -> DataSplit:
    """Deterministic 80/10/10 train/val/test split."""
    n = int(hashlib.md5(record_id.encode("utf-8")).hexdigest(), 16) % 100
    if n < 80:
        return DataSplit.TRAIN
    if n < 90:
        return DataSplit.VAL
    return DataSplit.TEST


def _domain_to_category(domain: str, list_key: str) -> Category:
    mapping = {
        "crop": Category.CROP,
        "disease": Category.DISEASE,
        "pest": Category.PEST,
        "soil": Category.SOIL,
        "fertilizer": Category.FERTILIZER,
        "government": Category.SCHEME,
        "market": Category.MARKET,
        "irrigation": Category.IRRIGATION,
        "seed": Category.SEED,
        "weather": Category.WEATHER,
        "finance": Category.FINANCE,
        "machinery": Category.MACHINERY,
        "general": Category.GENERAL,
    }
    if list_key in ("diseases_and_pests", "diseases"):
        return Category.DISEASE
    if list_key in ("schemes",):
        return Category.SCHEME
    if list_key in ("advisories",):
        return Category.ADVISORY
    if list_key in ("markets",):
        return Category.MARKET
    if list_key in ("varieties",):
        return Category.SEED
    if list_key in ("practices",):
        return Category.IRRIGATION
    if list_key in ("zones",):
        return Category.WEATHER
    if list_key in ("soil_types",):
        return Category.SOIL
    if list_key in ("fertilizer_recommendations",):
        return Category.FERTILIZER
    if list_key in ("crops",):
        return Category.CROP
    return mapping.get(domain, Category.GENERAL)


def _qa_from_item(item: dict, list_key: str, domain: str, source_path: str) -> list[StandardRecord]:
    """Generate one or more StandardRecords (Q/A pairs) from a structured fact."""
    records: list[StandardRecord] = []
    category = _domain_to_category(domain, list_key)
    source = f"lake_processed:{source_path}"

    def add(q: str, a: str, crop: str | None = None, sub: str | None = None, conf: float = 0.75):
        q, a = (q or "").strip(), (a or "").strip()
        if len(q) < 5 or len(a) < 5:
            return
        lang = detect_language_pair(q, a)
        rid = _stable_id(list_key, q[:80], a[:80], source_path)
        crop_name = resolve_crop_name(crop) if crop else None
        if not crop_name and crop:
            crop_name = crop
        # taxonomy category detection as fallback subcategory signal
        cats = taxonomy_service.detect_category(f"{q} {a}")
        rec = StandardRecord(
            id=rid,
            category=category,
            subcategory=sub or (cats[0] if cats else list_key),
            crop=crop_name,
            region=Region(state="Maharashtra", country="India"),
            language=lang,
            question=q,
            answer=a,
            source=source,
            confidence=conf,
            verified=False,
            split=_assign_split(rid),
            schema_version=SCHEMA_VERSION,
            metadata={
                "list_key": list_key,
                "domain": domain,
                "source_path": source_path,
            },
        )
        records.append(rec)

    if list_key == "crops":
        name = item.get("name_en") or item.get("crop") or "crop"
        name_mr = item.get("name_mr") or ""
        body = (
            f"Crop {name} ({name_mr}). Season: {item.get('season')}. "
            f"Soil: {', '.join(item.get('ideal_soil') or [])}. "
            f"Pests: {', '.join(item.get('major_pests') or [])}. "
            f"Diseases: {', '.join(item.get('major_diseases') or [])}. "
            f"Source: {item.get('source', 'platform KB')}."
        )
        add(f"What should I know about growing {name}?", body, crop=name, sub="crop_guide")
        if name_mr:
            add(f"{name_mr} पीक माहिती काय आहे?", body, crop=name, sub="crop_guide")

    elif list_key in ("diseases_and_pests", "diseases", "pests"):
        name = item.get("name_en") or item.get("name") or "pest/disease"
        crop = item.get("crop_en") or item.get("crop")
        symptoms = item.get("symptoms_en") or item.get("symptoms") or ""
        organic = item.get("organic_control_en") or ""
        chemical = item.get("chemical_control_en") or ""
        ans = (
            f"{name} on {crop}. Symptoms: {symptoms}. "
            f"Organic: {organic}. Chemical: {chemical}."
        )
        add(f"How do I manage {name} in {crop}?", ans, crop=crop, sub="disease_pest")
        name_mr = item.get("name_mr")
        if name_mr:
            add(
                f"{name_mr} नियंत्रण कसे करावे?",
                item.get("symptoms_mr") or ans,
                crop=crop,
                sub="disease_pest",
            )

    elif list_key == "schemes":
        name = item.get("name_en") or item.get("name") or "scheme"
        benefits = item.get("benefits_en") or item.get("benefits") or ""
        elig = item.get("eligibility_en") or ""
        ans = f"{name}. Benefits: {benefits}. Eligibility: {elig}. Portal: {item.get('portal', '')}"
        add(f"What is the {name} government scheme?", ans, sub="scheme")
        if item.get("name_mr"):
            add(
                f"{item.get('name_mr')} योजना काय आहे?",
                item.get("benefits_mr") or ans,
                sub="scheme",
            )

    elif list_key == "markets":
        crop = item.get("crop") or item.get("commodity")
        mandi = item.get("mandi") or item.get("market") or ""
        modal = item.get("modal_price_rs_quintal")
        ans = (
            f"{crop} at {mandi}, {item.get('district')}: "
            f"modal ₹{modal}/q (min {item.get('min_price_rs_quintal')}, "
            f"max {item.get('max_price_rs_quintal')}). Trend: {item.get('trend')}."
        )
        add(f"What is the mandi price for {crop} at {mandi}?", ans, crop=crop, sub="market")

    elif list_key == "fertilizer_recommendations":
        crop = item.get("crop_en") or item.get("crop")
        npk = item.get("recommended_npk_kg_per_acre") or {}
        ans = (
            f"NPK per acre for {crop}: N={npk.get('N')}, P={npk.get('P')}, K={npk.get('K')}. "
            f"Basal: {item.get('basal_dose')}. Top dressing: {item.get('top_dressing')}. "
            f"Micronutrients: {item.get('micronutrients')}."
        )
        add(f"What fertilizer should I apply for {crop}?", ans, crop=crop, sub="fertilizer")

    elif list_key == "soil_types":
        stype = item.get("type") or item.get("name_en") or "soil"
        ans = (
            f"{stype}. {item.get('characteristics_en', '')} "
            f"Suitable crops: {', '.join(item.get('suitable_crops') or [])}."
        )
        add(f"Describe {stype} and suitable crops.", ans, sub="soil")

    elif list_key == "advisories":
        title = item.get("title_en") or item.get("title") or "Advisory"
        content = item.get("content_en") or item.get("content") or item.get("body") or ""
        add(f"Advisory: {title}?", f"{title}. {content}", sub="advisory")
        if item.get("title_mr") or item.get("content_mr"):
            add(
                item.get("title_mr") or f"{title} (MR)",
                item.get("content_mr") or content,
                sub="advisory",
            )

    elif list_key == "varieties":
        crop = item.get("crop_en") or item.get("crop")
        var = item.get("variety") or item.get("name")
        ans = (
            f"Variety {var} for {crop}. Fit: {item.get('agro_climatic_fit')}. "
            f"Notes: {item.get('notes_mr') or item.get('notes', '')}."
        )
        add(f"Which seed variety for {crop}: {var}?", ans, crop=crop, sub="seed")

    elif list_key == "practices":
        title = item.get("title") or "Irrigation practice"
        content = item.get("content") or ""
        crops = item.get("crops") or []
        crop = crops[0] if crops else None
        add(f"What is the guidance for {title}?", f"{title}. {content}", crop=crop, sub="irrigation")

    elif list_key == "zones":
        zone = item.get("zone") or "zone"
        ans = (
            f"Zone {zone}. Rainfall {item.get('rainfall_mm')}. "
            f"Crops: {', '.join(item.get('recommended_crops') or [])}. "
            f"Notes: {item.get('notes', '')}."
        )
        add(f"What crops suit {zone}?", ans, sub="weather")

    elif list_key == "sources":
        name = item.get("name") or "source"
        ans = f"{name} ({item.get('type')}): {item.get('use')}. URL: {item.get('url')}"
        add(f"What open data source is {name}?", ans, sub="catalog", conf=0.6)

    elif list_key == "nodes":
        # skip pure graph nodes — low QA value unless Crop label
        if item.get("label") == "Crop":
            name = item.get("id")
            props = item.get("properties") or {}
            ans = f"Crop entity {name}. {props}"
            add(f"Graph entity: {name}?", ans, crop=name, sub="graph", conf=0.5)

    else:
        # generic fallback
        blob = json.dumps(item, ensure_ascii=False)[:500]
        add(f"What is this {list_key} record about?", blob, sub=list_key, conf=0.4)

    return records


def extract_standard_records_from_processed(
    *,
    max_per_file: int | None = None,
) -> list[StandardRecord]:
    """Walk processed lake and emit StandardRecords."""
    ensure_lake_layout()
    files = iter_processed_json_files()
    all_recs: list[StandardRecord] = []
    seen_ids: set[str] = set()

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            rel = relative_to_repo(path)
            domain = path.resolve().relative_to(LAKE_PROCESSED.resolve()).parts[0]
        except Exception:
            rel = str(path)
            domain = "general"

        lists = extract_record_lists(data)
        if not lists and isinstance(data, dict):
            # whole object as one weak record
            lists = [("$object", [data])]

        count = 0
        for list_key, items in lists:
            if list_key == "edges":
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                for rec in _qa_from_item(item, list_key, domain, rel):
                    if rec.id in seen_ids:
                        continue
                    seen_ids.add(rec.id)
                    all_recs.append(rec)
                    count += 1
                    if max_per_file and count >= max_per_file:
                        break
                if max_per_file and count >= max_per_file:
                    break
            if max_per_file and count >= max_per_file:
                break

    return all_recs


def coverage_stats(records: list[StandardRecord]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "with_language": 0,
            "with_category": 0,
            "language_pct": 0.0,
            "category_pct": 0.0,
            "by_language": {},
            "by_category": {},
            "by_split": {},
        }
    with_lang = sum(
        1 for r in records if r.language and r.language != LanguageCode.UNKNOWN
    )
    with_cat = sum(1 for r in records if r.category)
    by_lang: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for r in records:
        by_lang[r.language.value] = by_lang.get(r.language.value, 0) + 1
        by_cat[r.category.value] = by_cat.get(r.category.value, 0) + 1
        by_split[r.split.value] = by_split.get(r.split.value, 0) + 1
    return {
        "total": total,
        "with_language": with_lang,
        "with_category": with_cat,
        "language_pct": round(100.0 * with_lang / total, 2),
        "category_pct": round(100.0 * with_cat / total, 2),
        "by_language": by_lang,
        "by_category": by_cat,
        "by_split": by_split,
    }


def _write_jsonl(path: Path, records: list[StandardRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_training_dict(), ensure_ascii=False) + "\n")


def _write_parquet(path: Path, records: list[StandardRecord]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_training_dict() for r in records] if records else []
        path.write_bytes(json.dumps(rows, ensure_ascii=False).encode("utf-8"))
        return True
    except Exception:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"[]")
        return True


def export_standard_dataset(
    records: list[StandardRecord],
    *,
    dry_run: bool = False,
    version_tag: str | None = None,
) -> dict[str, Any]:
    """Write train/val/test JSONL (+ parquet) to lake splits and versioned datasets/."""
    ensure_lake_layout()
    stats = coverage_stats(records)
    train = [r for r in records if r.split == DataSplit.TRAIN]
    val = [r for r in records if r.split == DataSplit.VAL]
    test = [r for r in records if r.split == DataSplit.TEST]

    version = version_tag or datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    version_dir = DATASETS_DIR / "versions" / version

    artifacts: list[str] = []
    parquet_ok = False

    if not dry_run:
        # lake split roots
        mapping = {
            LAKE_TRAINING / "standard_records.jsonl": train,
            LAKE_VALIDATION / "standard_records.jsonl": val,
            LAKE_TEST / "standard_records.jsonl": test,
            version_dir / "train.jsonl": train,
            version_dir / "val.jsonl": val,
            version_dir / "test.jsonl": test,
            version_dir / "all.jsonl": records,
        }
        for path, recs in mapping.items():
            _write_jsonl(path, recs)
            artifacts.append(relative_to_repo(path))

        # parquet exports
        for name, recs in (
            ("train.parquet", train),
            ("val.parquet", val),
            ("test.parquet", test),
        ):
            p = version_dir / name
            if _write_parquet(p, recs):
                parquet_ok = True
                artifacts.append(relative_to_repo(p))
            lp = {
                "train.parquet": LAKE_TRAINING / "standard_records.parquet",
                "val.parquet": LAKE_VALIDATION / "standard_records.parquet",
                "test.parquet": LAKE_TEST / "standard_records.parquet",
            }[name]
            if _write_parquet(lp, recs):
                artifacts.append(relative_to_repo(lp))

        manifest = {
            "version": version,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sprint": "S4",
            "counts": {
                "total": len(records),
                "train": len(train),
                "val": len(val),
                "test": len(test),
            },
            "coverage": stats,
            "parquet": parquet_ok,
            "paths": artifacts,
        }
        man_path = version_dir / "manifest.json"
        man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts.append(relative_to_repo(man_path))

        latest = DATASETS_DIR / "LATEST_VERSION.json"
        latest.write_text(
            json.dumps(
                {"version": version, "manifest": relative_to_repo(man_path), **manifest["counts"]},
                indent=2,
            ),
            encoding="utf-8",
        )
        artifacts.append(relative_to_repo(latest))

        lake_report = LAKE_ROOT / "STANDARD_LATEST.json"
        lake_report.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts.append(relative_to_repo(lake_report))

    return {
        "ok": stats["total"] > 0 and stats["language_pct"] >= 90 and stats["category_pct"] >= 90,
        "version": version,
        "counts": {
            "total": len(records),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "coverage": stats,
        "artifacts": artifacts,
        "parquet": parquet_ok,
        "dry_run": dry_run,
    }


def run_standardize_pipeline(*, dry_run: bool = False) -> dict[str, Any]:
    records = extract_standard_records_from_processed()
    export = export_standard_dataset(records, dry_run=dry_run)
    return export
