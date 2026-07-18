"""Public taxonomy service used by workers and platform code."""

from __future__ import annotations

from typing import Any

from mini.taxonomy.aliases import (
    CATEGORY_ALIASES,
    CROP_ALIASES,
    resolve_crop_name,
    resolve_crops_in_text,
)
from mini.taxonomy.domains import (
    TAXONOMY,
    TAXONOMY_VERSION,
    get_crop_by_en,
    list_categories,
    list_category_details,
    list_crop_names_en,
    list_crop_stages,
    list_crops,
    taxonomy_summary,
)
from mini.taxonomy.regions import list_mh_districts, list_states, region_to_standard, resolve_district
from mini.taxonomy.units import convert_area, convert_mass, list_dimensions, normalize_unit_token
from mini.taxonomy.validator import full_validation_report, validate_taxonomy_integrity


class TaxonomyService:
    """Single entry-point for taxonomy operations (Sprint 1 freeze)."""

    version = TAXONOMY_VERSION

    def summary(self) -> dict[str, Any]:
        s = taxonomy_summary()
        s["crop_aliases"] = len(CROP_ALIASES)
        s["mh_districts"] = len(list_mh_districts())
        s["unit_dimensions"] = len(list_dimensions())
        return s

    def categories(self) -> list[str]:
        return list_categories()

    def category_details(self) -> list[dict]:
        return list_category_details()

    def crops(self) -> list[dict]:
        return list_crops()

    def crop_names(self) -> list[str]:
        return list_crop_names_en()

    def stages(self) -> list[dict]:
        return list_crop_stages()

    def resolve_crop(self, text: str) -> str | None:
        return resolve_crop_name(text)

    def extract_crops(self, text: str) -> list[str]:
        return resolve_crops_in_text(text)

    def crop_aliases(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in CROP_ALIASES.items()}

    def resolve_region(self, district: str | None = None, state: str | None = None) -> dict:
        return region_to_standard(district=district, state=state)

    def resolve_district(self, text: str) -> dict | None:
        return resolve_district(text)

    def states(self) -> list[dict]:
        return list_states()

    def mh_districts(self) -> list[str]:
        return list_mh_districts()

    def normalize_unit(self, token: str) -> str | None:
        return normalize_unit_token(token)

    def convert_mass(self, value: float, from_unit: str, to_unit: str = "kg") -> float | None:
        return convert_mass(value, from_unit, to_unit)

    def convert_area(self, value: float, from_unit: str, to_unit: str = "ha") -> float | None:
        return convert_area(value, from_unit, to_unit)

    def detect_category(self, text: str) -> list[str]:
        lower = (text or "").lower()
        hits = []
        for cat, kws in CATEGORY_ALIASES.items():
            if any(k.lower() in lower for k in kws):
                hits.append(cat)
        return hits or ["general"]

    def validate(self) -> dict[str, Any]:
        return full_validation_report()

    def integrity(self) -> dict[str, Any]:
        return validate_taxonomy_integrity()

    def export_snapshot(self) -> dict[str, Any]:
        """Full snapshot for UI / manifests."""
        return {
            "version": TAXONOMY_VERSION,
            "taxonomy": TAXONOMY,
            "crop_aliases": self.crop_aliases(),
            "summary": self.summary(),
        }

    def get_crop_record(self, name_en: str) -> dict | None:
        return get_crop_by_en(name_en)


taxonomy_service = TaxonomyService()
