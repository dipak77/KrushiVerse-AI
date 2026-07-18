"""Taxonomy integrity checks + platform KB entity coverage (Sprint 1)."""

from __future__ import annotations

from typing import Any

from mini.taxonomy.aliases import CROP_ALIASES, resolve_crop_name
from mini.taxonomy.domains import (
    TAXONOMY,
    TAXONOMY_STATUS,
    TAXONOMY_VERSION,
    list_categories,
    list_crop_names_en,
    list_crops,
)
from mini.taxonomy.regions import REGIONS, list_mh_districts
from mini.taxonomy.units import UNITS, list_dimensions


def validate_taxonomy_integrity() -> dict[str, Any]:
    """Validate internal consistency of the frozen taxonomy."""
    errors: list[str] = []
    warnings: list[str] = []

    if TAXONOMY_STATUS != "frozen":
        errors.append(f"Taxonomy status must be 'frozen' for Sprint 1, got {TAXONOMY_STATUS}")

    cats = list_categories()
    if len(cats) < 10:
        errors.append(f"Expected ≥10 categories, got {len(cats)}")
    if len(set(cats)) != len(cats):
        errors.append("Duplicate category ids")

    crops = list_crops()
    if len(crops) < 20:
        errors.append(f"Expected ≥20 crops, got {len(crops)}")
    ids = [c["id"] for c in crops]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate crop ids")

    # Every crop must have EN/MR/HI + group
    for c in crops:
        for field in ("id", "name_en", "name_mr", "name_hi", "group"):
            if not c.get(field):
                errors.append(f"Crop {c.get('id')} missing {field}")

    # Alias coverage: every taxonomy crop name_en must have aliases
    for name in list_crop_names_en():
        if name not in CROP_ALIASES:
            errors.append(f"Missing CROP_ALIASES for taxonomy crop: {name}")

    # Alias reverse: every alias canonical should resolve
    for canonical in CROP_ALIASES:
        resolved = resolve_crop_name(canonical)
        if resolved != canonical:
            warnings.append(f"Canonical '{canonical}' resolves to '{resolved}'")

    # Regions
    if not REGIONS.get("states"):
        errors.append("No states in region hierarchy")
    mh_districts = list_mh_districts()
    if len(mh_districts) < 20:
        warnings.append(f"Only {len(mh_districts)} MH districts listed")

    # Units
    dims = list_dimensions()
    for required in ("mass", "area", "temperature", "rainfall", "currency"):
        if required not in dims:
            errors.append(f"Missing unit dimension: {required}")

    # Stages
    stages = TAXONOMY.get("crop_stages") or []
    if len(stages) < 8:
        errors.append(f"Expected ≥8 crop stages, got {len(stages)}")

    return {
        "ok": len(errors) == 0,
        "version": TAXONOMY_VERSION,
        "status": TAXONOMY_STATUS,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "categories": len(cats),
            "crops": len(crops),
            "crop_aliases": len(CROP_ALIASES),
            "mh_districts": len(mh_districts),
            "unit_dimensions": len(dims),
            "stages": len(stages),
        },
    }


def _map_label_to_category(label: str | None) -> str | None:
    if not label:
        return None
    l = label.lower()
    mapping = {
        "crop": "crop",
        "pest": "pest",
        "disease": "disease",
        "soil": "soil",
        "fertilizer": "fertilizer",
        "scheme": "scheme",
        "weathercondition": "weather",
        "weather": "weather",
    }
    return mapping.get(l)


def validate_platform_kb_coverage() -> dict[str, Any]:
    """Ensure every entity in existing platform data maps to a taxonomy category/crop."""
    from app.knowledge.dataset_loader import kb_loader

    errors: list[str] = []
    warnings: list[str] = []
    mapped: dict[str, int] = {}
    unmapped: list[str] = []

    # Crops
    for crop in kb_loader.crops_and_diseases.get("crops", []):
        name = crop.get("name_en")
        resolved = resolve_crop_name(name or "")
        if not resolved:
            errors.append(f"KB crop not in taxonomy aliases: {name}")
            unmapped.append(f"crop:{name}")
        else:
            mapped["crop"] = mapped.get("crop", 0) + 1

    # Diseases / pests → disease or pest category
    for dis in kb_loader.crops_and_diseases.get("diseases_and_pests", []):
        crop = dis.get("crop_en")
        if crop and not resolve_crop_name(crop):
            warnings.append(f"Disease crop not resolved: {crop} ({dis.get('name_en')})")
        mapped["disease_pest"] = mapped.get("disease_pest", 0) + 1

    # Schemes
    for sch in kb_loader.government_schemes.get("schemes", []):
        mapped["scheme"] = mapped.get("scheme", 0) + 1

    # Soils
    for soil in kb_loader.soil_and_fertilizers.get("soil_types", []):
        mapped["soil"] = mapped.get("soil", 0) + 1

    # Fertilizer recs
    for fert in kb_loader.soil_and_fertilizers.get("fertilizer_recommendations", []):
        crop = fert.get("crop_en")
        if crop and not resolve_crop_name(crop):
            errors.append(f"Fertilizer crop not in taxonomy: {crop}")
            unmapped.append(f"fert:{crop}")
        else:
            mapped["fertilizer"] = mapped.get("fertilizer", 0) + 1

    # Markets
    for m in kb_loader.market_prices.get("markets", []):
        crop = m.get("crop")
        if crop and not resolve_crop_name(crop):
            warnings.append(f"Market crop not resolved: {crop}")
        mapped["market"] = mapped.get("market", 0) + 1

    # Graph nodes
    for node in kb_loader.graph_data.get("nodes", []):
        label = node.get("label")
        cat = _map_label_to_category(label)
        if not cat:
            warnings.append(f"Graph node label unmapped: {label} ({node.get('id')})")
        else:
            mapped[f"graph_{cat}"] = mapped.get(f"graph_{cat}", 0) + 1
        if label == "Crop":
            if not resolve_crop_name(str(node.get("id", ""))):
                errors.append(f"Graph crop node not in taxonomy: {node.get('id')}")

    # Advisories / seeds / irrigation docs — category covered by taxonomy categories list
    for cat_id in ("advisory", "seed", "irrigation"):
        if cat_id not in list_categories():
            errors.append(f"Missing taxonomy category for platform docs: {cat_id}")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "mapped_counts": mapped,
        "unmapped": unmapped,
    }


def full_validation_report() -> dict[str, Any]:
    integrity = validate_taxonomy_integrity()
    coverage = validate_platform_kb_coverage()
    return {
        "ok": integrity["ok"] and coverage["ok"],
        "integrity": integrity,
        "platform_coverage": coverage,
        "taxonomy_version": TAXONOMY_VERSION,
    }
