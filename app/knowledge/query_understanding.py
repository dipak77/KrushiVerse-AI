"""Query understanding: crop/entity extraction, intent tags, multi-query expansion."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


CROP_ALIASES: dict[str, list[str]] = {
    "Cotton": ["cotton", "कापूस", "kapus", "kapas"],
    "Soybean": ["soybean", "soya", "सोयाबीन", "soy"],
    "Sugarcane": ["sugarcane", "cane", "ऊस", "oos", "us"],
    "Pomegranate": ["pomegranate", "डाळिंब", "dalimb", "anar"],
    "Onion": ["onion", "कांदा", "kanda"],
    "Rice": ["rice", "paddy", "भात", "तांदूळ", "dhan"],
    "Wheat": ["wheat", "गहू", "gahu", "gehun"],
    "Maize": ["maize", "corn", "मका", "maka", "makka"],
    "Tur (Pigeon Pea)": ["tur", "toor", "arhar", "pigeon pea", "तूर", "red gram"],
    "Gram (Chickpea)": ["gram", "chickpea", "chana", "हरभरा", "harbhara"],
    "Groundnut": ["groundnut", "peanut", "भुईमूग", "mungfali", "bhuimug"],
    "Turmeric": ["turmeric", "हळद", "halad", "haldi"],
    "Grapes": ["grape", "grapes", "द्राक्ष", "draksh"],
    "Banana": ["banana", "केळी", "keli"],
    "Mango": ["mango", "आंबा", "amba"],
    "Tomato": ["tomato", "टोमॅटो", "tamatar"],
    "Chilli": ["chilli", "chili", "pepper", "मिरची", "mirchi"],
    "Sorghum (Jowar)": ["sorghum", "jowar", "ज्वारी", "jwari"],
    "Bajra (Pearl Millet)": ["bajra", "pearl millet", "बाजरी", "bajri"],
    "Mustard": ["mustard", "मोहरी", "sarson", "mohri"],
    "Potato": ["potato", "बटाटा", "batata", "aloo"],
    "Orange (Nagpur Santra)": ["orange", "santra", "संत्रा", "citrus", "nagpur santra"],
}

INTENT_KEYWORDS: dict[str, list[str]] = {
    "weather": ["weather", "rain", "rainfall", "monsoon", "humidity", "पाऊस", "हवामान", "temperature"],
    "disease": ["disease", "pest", "blight", "virus", "worm", "thrips", "रोग", "कीड", "spot", "infection"],
    "market": ["market", "price", "mandi", "rate", "msp", "भाव", "बाजार", "मंडी", "rate"],
    "fertilizer": ["fertilizer", "manure", "npk", "urea", "dap", "खत", "माती", "soil", "nutrient"],
    "scheme": ["scheme", "subsidy", "yojana", "insurance", "pm-kisan", "योजना", "अनुदान", "विमा"],
    "irrigation": ["irrigation", "drip", "water", "सिंचन", "ठिबक", "पाणी", "sprinkler"],
    "seed": ["seed", "variety", "hybrid", "बियाणे", "जात", "variety"],
    "finance": ["loan", "kcc", "credit", "roi", "cost", "कर्ज", "वित्त"],
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


class QueryUnderstanding:
    def understand(self, query: str, default_crop: str | None = None) -> QueryPlan:
        normalized = " ".join(query.strip().split())
        lower = normalized.lower()
        crops = self.extract_crops(lower)
        if not crops and default_crop:
            crops = [default_crop]

        intents = [name for name, kws in INTENT_KEYWORDS.items() if any(k in lower for k in kws)]
        if not intents:
            intents = ["general"]

        expanded = self._expand(normalized, crops, intents)
        needs_web = self._needs_web(lower, intents)
        needs_live = any(i in intents for i in ("weather", "market")) or needs_web

        lang = "mr" if re.search(r"[\u0900-\u097F]", normalized) else "en"

        return QueryPlan(
            original=query,
            normalized=normalized,
            crops=crops,
            intents=intents,
            expanded_queries=expanded,
            needs_web=needs_web,
            needs_live_tools=needs_live,
            language_hint=lang,
        )

    def extract_crops(self, text_lower: str) -> list[str]:
        found: list[str] = []
        for crop, aliases in CROP_ALIASES.items():
            for alias in aliases:
                if alias.lower() in text_lower:
                    if crop not in found:
                        found.append(crop)
                    break
        return found

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
        # Deduplicate preserving order
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
            "latest", "today", "current", "news", "2024", "2025", "2026",
            "web", "online", "update", "recent", "forecast", "live",
            "आता", "आज", "ताजे", "सद्य",
        ]
        if any(t in lower for t in web_triggers):
            return True
        if "market" in intents or "weather" in intents or "scheme" in intents:
            return True
        return False


query_understanding = QueryUnderstanding()
