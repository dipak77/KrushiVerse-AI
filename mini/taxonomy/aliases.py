"""Canonical crop / entity aliases (EN / MR / HI) — frozen Sprint 1.

Canonical English names match platform KB crop `name_en` values where possible.
"""

from __future__ import annotations

# Map: canonical English name -> list of aliases (lowercase match keys)
CROP_ALIASES: dict[str, list[str]] = {
    "Cotton": [
        "cotton", "kapas", "kapus", "कापूस", "कपास", "कपासू", "gossypium",
    ],
    "Soybean": [
        "soybean", "soya", "soy", "सोयाबीन", "सोयाबीन", "सोया", "glycine max",
    ],
    "Sugarcane": [
        "sugarcane", "cane", "ऊस", "गन्ना", "oos", "us", "saccharum",
    ],
    "Pomegranate": [
        "pomegranate", "anar", "dalimb", "डाळिंब", "अनार", "punica",
    ],
    "Onion": [
        "onion", "kanda", "कांदा", "प्याज", "pyaz", "allium cepa",
    ],
    "Rice": [
        "rice", "paddy", "dhan", "भात", "तांदूळ", "चावल", "धान", "oryza",
    ],
    "Wheat": [
        "wheat", "gahu", "gehun", "गहू", "गेहूं", "गेहू", "triticum",
    ],
    "Maize": [
        "maize", "corn", "maka", "makka", "मका", "मक्का", "zea mays",
    ],
    "Tur (Pigeon Pea)": [
        "tur", "toor", "arhar", "pigeon pea", "red gram", "तूर", "अरहर",
        "तूर डाळ", "cajanus",
    ],
    "Gram (Chickpea)": [
        "gram", "chickpea", "chana", "harbhara", "हरभरा", "चना", "cicer",
    ],
    "Groundnut": [
        "groundnut", "peanut", "mungfali", "bhuimug", "भुईमूग", "मूंगफली",
        "arachis",
    ],
    "Turmeric": [
        "turmeric", "halad", "haldi", "हळद", "हल्दी", "curcuma",
    ],
    "Grapes": [
        "grape", "grapes", "draksh", "द्राक्ष", "अंगूर", "vitis",
    ],
    "Banana": [
        "banana", "keli", "केळी", "केला", "musa",
    ],
    "Mango": [
        "mango", "amba", "आंबा", "आम", "mangifera",
    ],
    "Tomato": [
        "tomato", "tamatar", "टोमॅटो", "टमाटर", "solanum lycopersicum",
    ],
    "Chilli": [
        "chilli", "chili", "pepper", "mirchi", "मिरची", "मिर्च", "capsicum",
    ],
    "Sorghum (Jowar)": [
        "sorghum", "jowar", "jwari", "ज्वारी", "ज्वार", "sorghum bicolor",
    ],
    "Bajra (Pearl Millet)": [
        "bajra", "bajri", "pearl millet", "बाजरी", "बाजरा", "pennisetum",
    ],
    "Mustard": [
        "mustard", "sarson", "mohri", "मोहरी", "सरसों", "brassica",
    ],
    "Potato": [
        "potato", "batata", "aloo", "बटाटा", "आलू", "solanum tuberosum",
    ],
    "Orange (Nagpur Santra)": [
        "orange", "santra", "citrus", "nagpur santra", "संत्रा", "नारंगी",
        "नागपूर संत्रा",
    ],
}

# Intent / category keyword aliases used by routing & normalize worker
CATEGORY_ALIASES: dict[str, list[str]] = {
    "soil": ["soil", "माती", "मिट्टी", "ph", "npk", "soil health", "माती आरोग्य"],
    "weather": ["weather", "rain", "rainfall", "monsoon", "humidity", "हवामान", "पाऊस", "मौसम", "temperature"],
    "crop": ["crop", "पीक", "फसल", "sowing", "harvest", "variety", "जात"],
    "disease": ["disease", "blight", "virus", "fungal", "रोग", "व्याधि", "infection", "spot"],
    "pest": ["pest", "insect", "worm", "thrips", "bollworm", "कीड", "कीड़ा", "अळी"],
    "fertilizer": [
        "fertilizer", "manure", "urea", "dap", "mop", "npk", "nutrient",
        "खत", "उर्वरक", "यूरिया", "युरिया", "डीएपी", "पालाश", "नत्र", "स्फुरद",
    ],
    "irrigation": ["irrigation", "drip", "sprinkler", "सिंचन", "ठिबक", "सिंचाई", "water"],
    "scheme": ["scheme", "subsidy", "yojana", "pm-kisan", "pmfby", "योजना", "अनुदान", "सब्सिडी"],
    "market": ["market", "mandi", "price", "msp", "भाव", "बाजार", "मंडी", "दर"],
    "finance": ["loan", "kcc", "credit", "insurance", "roi", "कर्ज", "वित्त", "बीमा"],
    "machinery": ["tractor", "drone", "sprayer", "harvester", "यंत्र", "ट्रॅक्टर", "ट्रैक्टर"],
    "seed": ["seed", "hybrid", "variety", "बियाणे", "बीज"],
    "advisory": ["advisory", "recommendation", "salla", "सल्ला", "सलाह"],
}


def build_alias_lookup(aliases: dict[str, list[str]] | None = None) -> dict[str, str]:
    """Return lowercase-alias -> canonical name map (longest aliases preferred)."""
    source = aliases or CROP_ALIASES
    lookup: dict[str, str] = {}
    for canonical, alist in source.items():
        for a in alist:
            key = a.lower().strip()
            if key:
                lookup[key] = canonical
        # also map canonical itself
        lookup[canonical.lower()] = canonical
    return lookup


ALIAS_LOOKUP = build_alias_lookup()


def resolve_crop_name(text: str) -> str | None:
    """Resolve free text to a canonical crop name, or None."""
    if not text:
        return None
    t = text.strip()
    # exact / contained alias match (prefer longer aliases)
    lower = t.lower()
    if lower in ALIAS_LOOKUP:
        return ALIAS_LOOKUP[lower]
    # substring scan sorted by alias length desc
    candidates = sorted(ALIAS_LOOKUP.keys(), key=len, reverse=True)
    for alias in candidates:
        if len(alias) >= 3 and alias in lower:
            return ALIAS_LOOKUP[alias]
    return None


def resolve_crops_in_text(text: str) -> list[str]:
    """Find all crops mentioned in text (order of first appearance)."""
    if not text:
        return []
    lower = text.lower()
    found: list[tuple[int, str]] = []
    seen: set[str] = set()
    for alias, canonical in sorted(ALIAS_LOOKUP.items(), key=lambda x: -len(x[0])):
        if len(alias) < 2:
            continue
        idx = lower.find(alias)
        if idx >= 0 and canonical not in seen:
            seen.add(canonical)
            found.append((idx, canonical))
    found.sort(key=lambda x: x[0])
    return [c for _, c in found]
