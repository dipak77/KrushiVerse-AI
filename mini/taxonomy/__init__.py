"""Agriculture domain taxonomy — frozen Sprint 1 (v1.0.0)."""

from mini.taxonomy.domains import (
    TAXONOMY,
    TAXONOMY_VERSION,
    list_categories,
    list_crops,
    list_crop_names_en,
    list_crop_stages,
    taxonomy_summary,
)
from mini.taxonomy.aliases import CROP_ALIASES, resolve_crop_name, resolve_crops_in_text
from mini.taxonomy.service import TaxonomyService, taxonomy_service
from mini.taxonomy.validator import full_validation_report, validate_taxonomy_integrity

__all__ = [
    "TAXONOMY",
    "TAXONOMY_VERSION",
    "CROP_ALIASES",
    "list_categories",
    "list_crops",
    "list_crop_names_en",
    "list_crop_stages",
    "taxonomy_summary",
    "resolve_crop_name",
    "resolve_crops_in_text",
    "TaxonomyService",
    "taxonomy_service",
    "full_validation_report",
    "validate_taxonomy_integrity",
]
