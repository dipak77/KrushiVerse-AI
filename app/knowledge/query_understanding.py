"""Query understanding: crop/entity extraction, intent tags, multi-query expansion.

Crop aliases are sourced from the frozen Mini taxonomy (Sprint 1) so platform
routing and the ML factory stay aligned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mini.taxonomy.aliases import (
    CATEGORY_ALIASES,
    CROP_ALIASES,
    resolve_crops_in_text,
)

# Re-export for callers that imported CROP_ALIASES from this module
__all__ = [
    "CROP_ALIASES",
    "INTENT_KEYWORDS",
    "QueryPlan",
    "QueryUnderstanding",
    "query_understanding",
]

# Map taxonomy category aliases → planner intents (backward compatible keys)
INTENT_KEYWORDS: dict[str, list[str]] = {
    "weather": CATEGORY_ALIASES.get("weather", []),
    "disease": list(
        dict.fromkeys(
            (CATEGORY_ALIASES.get("disease") or []) + (CATEGORY_ALIASES.get("pest") or [])
        )
    ),
    "market": CATEGORY_ALIASES.get("market", []),
    "fertilizer": list(
        dict.fromkeys(
            (CATEGORY_ALIASES.get("fertilizer") or []) + (CATEGORY_ALIASES.get("soil") or [])
        )
    ),
    "scheme": CATEGORY_ALIASES.get("scheme", []),
    "irrigation": CATEGORY_ALIASES.get("irrigation", []),
    "seed": CATEGORY_ALIASES.get("seed", []),
    "finance": CATEGORY_ALIASES.get("finance", []),
    "machinery": CATEGORY_ALIASES.get("machinery", []),
}


@dataclass
class QueryPlan:
    original: str
    normalized: str
    crops: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    needs_web: bool = False
    needs_live_tools: bool = False
    language_hint: str = "en"
    categories: list[str] = field(default_factory=list)
    location: str | None = None


class QueryUnderstanding:
    def understand(self, query: str, default_crop: str | None = None) -> QueryPlan:
        normalized = " ".join(query.strip().split())
        lower = normalized.lower()
        crops = self.extract_crops(lower)
        if not crops and default_crop:
            from mini.taxonomy.aliases import resolve_crop_name

            crops = [resolve_crop_name(default_crop) or default_crop]

        from mini.taxonomy.regions import resolve_district
        dist_info = resolve_district(normalized)
        extracted_location = dist_info.get("district") if dist_info else None

        intents = [
            name for name, kws in INTENT_KEYWORDS.items() if any(k.lower() in lower for k in kws)
        ]
        categories = [
            cat
            for cat, kws in CATEGORY_ALIASES.items()
            if any(k.lower() in lower for k in kws)
        ]
        if not intents:
            intents = ["general"]
        if not categories:
            categories = ["general"]

        expanded = self._expand(normalized, crops, intents)
        needs_web = self._needs_web(lower, intents)
        needs_live = any(i in intents for i in ("weather", "market")) or needs_web

        if re.search(r"[\u0900-\u097F]", normalized):
            lang = "mr"
        else:
            lang = "en"

        return QueryPlan(
            original=query,
            normalized=normalized,
            crops=crops,
            intents=intents,
            expanded_queries=expanded,
            needs_web=needs_web,
            needs_live_tools=needs_live,
            language_hint=lang,
            categories=categories,
            location=extracted_location,
        )

    def extract_crops(self, text_lower: str) -> list[str]:
        # resolve_crops_in_text expects any case; use original-ish lower for match
        return resolve_crops_in_text(text_lower)

    def _expand(self, query: str, crops: list[str], intents: list[str]) -> list[str]:
        qs = [query]
        crop = crops[0] if crops else ""
        if crop:
            qs.append(f"{crop} {query}")
            for intent in intents:
                if intent == "disease":
                    qs.append(f"{crop} pest disease IPM organic chemical control ICAR")
                elif intent == "fertilizer":
                    qs.append(f"{crop} fertilizer NPK recommendation kg per acre soil")
                elif intent == "market":
                    qs.append(f"{crop} mandi price Maharashtra Agmarknet modal rate")
                elif intent == "scheme":
                    qs.append(f"{crop} government scheme subsidy insurance Maharashtra PMFBY")
                elif intent == "irrigation":
                    qs.append(f"{crop} drip irrigation water requirement schedule")
                elif intent == "weather":
                    qs.append(f"{crop} weather advisory humidity rainfall pest risk")
                elif intent == "seed":
                    qs.append(f"{crop} recommended seed varieties India ICAR")
                elif intent == "machinery":
                    qs.append(f"{crop} farm machinery sprayer seeder recommendation")
        seen = set()
        out = []
        for q in qs:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                out.append(q)
        return out[:6]

    def _needs_web(self, lower: str, intents: list[str]) -> bool:
        web_triggers = [
            "latest",
            "today",
            "current",
            "news",
            "2024",
            "2025",
            "2026",
            "web",
            "online",
            "update",
            "recent",
            "forecast",
            "live",
            "आता",
            "आज",
            "ताजे",
            "सद्य",
        ]
        if any(t in lower for t in web_triggers):
            return True
        if "market" in intents or "weather" in intents or "scheme" in intents:
            return True
        return False


query_understanding = QueryUnderstanding()
