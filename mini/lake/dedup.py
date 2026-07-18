"""Exact and near-duplicate detection for cleaned records (Sprint 3)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from mini.lake.quality import char_ngrams, jaccard, record_fingerprint_text


def stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def extract_record_lists(payload: Any) -> list[tuple[str, list[Any]]]:
    """Return (key, list) pairs of extractable record arrays from a JSON payload."""
    if isinstance(payload, list):
        return [("$root", payload)]
    if not isinstance(payload, dict):
        return []
    keys = [
        "crops",
        "diseases_and_pests",
        "markets",
        "schemes",
        "advisories",
        "varieties",
        "practices",
        "zones",
        "soil_types",
        "fertilizer_recommendations",
        "sources",
        "nodes",
        "edges",
        "records",
    ]
    out: list[tuple[str, list[Any]]] = []
    for k in keys:
        if isinstance(payload.get(k), list):
            out.append((k, payload[k]))
    return out


def dedupe_list(
    items: list[Any],
    *,
    near_threshold: float = 0.92,
) -> tuple[list[Any], dict[str, Any]]:
    """Remove exact and near-duplicate items. Returns (kept, stats)."""
    kept: list[Any] = []
    seen_exact: set[str] = set()
    near_sigs: list[set[str]] = []
    exact_removed = 0
    near_removed = 0

    for item in items:
        h = stable_hash(item)
        if h in seen_exact:
            exact_removed += 1
            continue
        text = record_fingerprint_text(item)
        grams = char_ngrams(text, 3)
        is_near = False
        for prev in near_sigs:
            if jaccard(grams, prev) >= near_threshold:
                is_near = True
                break
        if is_near:
            near_removed += 1
            continue
        seen_exact.add(h)
        near_sigs.append(grams)
        kept.append(item)

    return kept, {
        "input": len(items),
        "kept": len(kept),
        "exact_removed": exact_removed,
        "near_removed": near_removed,
    }


def dedupe_payload(
    payload: Any,
    *,
    near_threshold: float = 0.92,
) -> tuple[Any, dict[str, Any]]:
    """Deduplicate all known list fields in a JSON payload."""
    if isinstance(payload, list):
        kept, stats = dedupe_list(payload, near_threshold=near_threshold)
        return kept, {"$root": stats}

    if not isinstance(payload, dict):
        return payload, {}

    out = dict(payload)
    all_stats: dict[str, Any] = {}
    for key, items in extract_record_lists(payload):
        kept, stats = dedupe_list(items, near_threshold=near_threshold)
        out[key] = kept
        all_stats[key] = stats
    return out, all_stats
