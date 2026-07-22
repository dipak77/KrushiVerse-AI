"""Canonical crop / entity aliases (EN / MR / HI) with smart Marathi stemmer + multi-word.

Handles: सोयाबीनला -> सोयाबीन + ला, कापसाला -> कापूस, pigeon pea, nagpur santra
Canonical English names match platform KB crop name_en (simple, no brackets).
"""

from __future__ import annotations

import re

# Only base forms — no inflections! Stemmer handles ला, ने, वरील etc.
BASE_ALIASES: dict[str, list[str]] = {
    "Cotton": ["cotton", "कापूस", "कपास", "kapas", "gossypium"],
    "Soybean": ["soybean", "सोयाबीन", "सोया", "glycine max"],
    "Sugarcane": ["sugarcane", "ऊस", "गन्ना", "saccharum"],
    "Pomegranate": ["pomegranate", "anar", "dalimb", "डाळिंब", "अनार", "punica"],
    "Onion": ["onion", "kanda", "कांदा", "प्याज", "pyaz"],
    "Rice": ["rice", "paddy", "dhan", "भात", "तांदूळ", "चावल", "धान", "oryza"],
    "Wheat": ["wheat", "gahu", "गहू", "गेहूं", "triticum"],
    "Maize": ["maize", "corn", "maka", "मका", "मक्का", "zea mays"],
    "Tur": ["tur", "toor", "arhar", "pigeon pea", "red gram", "तूर", "अरहर", "तूर डाळ", "cajanus", "pigeonpea"],
    "Gram": ["gram", "chickpea", "chana", "harbhara", "हरभरा", "चना", "cicer"],
    "Groundnut": ["groundnut", "peanut", "bhuimug", "भुईमूग", "मूंगफली", "arachis"],
    "Turmeric": ["turmeric", "halad", "हळद", "हल्दी", "curcuma"],
    "Grapes": ["grape", "grapes", "draksh", "द्राक्ष", "अंगूर", "vitis"],
    "Banana": ["banana", "keli", "केळी", "केला", "musa"],
    "Mango": ["mango", "amba", "आंबा", "mangifera"],
    "Tomato": ["tomato", "tamatar", "टोमॅटो", "टमाटर"],
    "Chilli": ["chilli", "chili", "mirchi", "मिरची", "मिर्च", "capsicum"],
    "Sorghum": ["sorghum", "jowar", "ज्वारी", "ज्वार"],
    "Bajra": ["bajra", "bajri", "pearl millet", "बाजरी", "बाजरा", "pennisetum"],
    "Mustard": ["mustard", "sarson", "मोहरी", "सरसों", "brassica"],
    "Potato": ["potato", "batata", "aloo", "बटाटा", "आलू"],
    "Orange": ["orange", "santra", "citrus", "nagpur santra", "संत्रा", "मोसंबी", "नारंगी", "नागपूर संत्रा"],
    "Brinjal": ["brinjal", "eggplant", "baingan", "वांगी", "वांगे", "बैंगन", "solanum melongena"],
}

CROP_ALIASES = BASE_ALIASES # backwards compat

# Longest suffix first — for smart split
MR_SUFFIXES = sorted([
    "ावरील", "ांवरील", "ाच्या", "ांच्या", "ासाठी", "ापेक्षा", "ामध्ये", "ातून",
    "ाची", "ाचा", "ाचे", "ाला", "ाने", "ात", "ास",
    "वरील", "साठी", "पेक्षा", "मध्ये", "तील", "मधून",
    "ची", "चा", "चे", "च्या", "ला", "ने", "त", "वर", "मधील", "ना"
], key=len, reverse=True)

def stem_mr_token(token: str) -> str:
    """Smart stem: सोयाबीनला -> सोयाबीन, कापसाला -> कापूस"""
    t = token.lower().strip()
    if not t:
        return t

    # Hard oblique rules — Marathi stem changes
    if t.startswith("कापसा") or t.startswith("कपाशी") or t.startswith("कापस"): return "कापूस"
    if t.startswith("वांग्या") or t.startswith("वांग्य") or t.startswith("वांगे"): return "वांगी"
    if t.startswith("डाळिंबा"): return "डाळिंब"
    if t.startswith("द्राक्षा"): return "द्राक्ष"
    if t.startswith("कांद्या"): return "कांदा"
    if t.startswith("गव्हा"): return "गहू"
    if t.startswith("मक्या"): return "मका"
    if t.startswith("हरभऱ्या") or t.startswith("हरभर्य"): return "हरभरा"
    if t.startswith("भुईमुगा"): return "भुईमूग"
    if t.startswith("हळदी"): return "हळद"
    if t.startswith("आंब्या"): return "आंबा"
    if t.startswith("बटाट्या"): return "बटाटा"
    if t.startswith("सोयाबीन"): return "सोयाबीन"
    if t.startswith("केळी"): return "केळी"
    if t.startswith("ऊसा"): return "ऊस"

    for suf in MR_SUFFIXES:
        if t.endswith(suf) and len(t) > len(suf) + 2:
            return t[:-len(suf)]
    return t

def tokenize(text: str) -> list[str]:
    return [w for w in re.split(r'[\s,।.|!?()\[\]"\'/\-]+', text.lower()) if w]

# Lookup maps
BASE_LOOKUP: dict[str, str] = {}
for canon, aliases in BASE_ALIASES.items():
    for a in aliases:
        k = a.lower().strip()
        if k: BASE_LOOKUP[k] = canon
    BASE_LOOKUP[canon.lower()] = canon

# Multi-word phrases sorted by word-count then length (longest first)
PHRASES_SORTED = sorted(
    [k for k in BASE_LOOKUP.keys() if " " in k],
    key=lambda x: (-len(x.split()), -len(x))
)

MAX_PHRASE_WORDS = max((len(p.split()) for p in PHRASES_SORTED), default=1)

def resolve_crops_smart(text: str) -> list[str]:
    """
    Smarter multi-word + Marathi inflection handler.
    1. Phrase match (nagpur santra, pigeon pea) longest first
    2. Token + stem + split postposition: सोयाबीनला -> [सोयाबीन, ला] -> सोयाबीन
    """
    if not text:
        return []

    t_low = f" {text.lower()} " # pad for boundary checks
    found: list[str] = []
    seen: set[str] = set()

    # 1. Multi-word phrase match with word boundaries (handles English + Marathi)
    for phrase in PHRASES_SORTED:
        if f" {phrase} " in t_low or t_low.startswith(phrase+" ") or t_low.endswith(" "+phrase) or phrase == t_low.strip():
            canon = BASE_LOOKUP[phrase]
            if canon not in seen:
                seen.add(canon)
                found.append(canon)

    if found:
        return found

    # 2. N-gram sliding window over tokens for inflected multi-word: "नागपूर संत्र्याला"
    tokens = tokenize(text)
    stemmed_tokens = [stem_mr_token(t) for t in tokens]

    for n in range(min(MAX_PHRASE_WORDS, len(tokens)), 1, -1):
        for i in range(len(tokens) - n + 1):
            orig_ngram = " ".join(tokens[i:i+n])
            stem_ngram = " ".join(stemmed_tokens[i:i+n])

            for ng in (orig_ngram, stem_ngram):
                if ng in BASE_LOOKUP:
                    canon = BASE_LOOKUP[ng]
                    if canon not in seen:
                        seen.add(canon)
                        found.append(canon)
                    break

    if found:
        return found

    # 3. Single token + split postposition logic (सोयाबीनला)
    for tok, stem_tok in zip(tokens, stemmed_tokens):
        candidates = {tok, stem_tok}

        # Split postposition: try removing suffix and check remainder
        for suf in MR_SUFFIXES:
            if tok.endswith(suf) and len(tok) > len(suf)+2:
                rem = tok[:-len(suf)]
                if rem:
                    candidates.add(rem)
                    candidates.add(stem_mr_token(rem))

        for cand in candidates:
            if cand in BASE_LOOKUP:
                canon = BASE_LOOKUP[cand]
                if canon not in seen:
                    seen.add(canon)
                    found.append(canon)
                break

    return found

# Backwards compat wrappers — single definition only
def resolve_crop_name(text: str) -> str | None:
    crops = resolve_crops_smart(text)
    return crops[0] if crops else None

def resolve_crops_in_text(text: str) -> list[str]:
    return resolve_crops_smart(text)

def build_alias_lookup(aliases: dict[str, list[str]] | None = None) -> dict[str, str]:
    src = aliases or CROP_ALIASES
    lookup = {}
    for canon, alist in src.items():
        for a in alist:
            k = a.lower().strip()
            if k: lookup[k] = canon
        lookup[canon.lower()] = canon
    return lookup

ALIAS_LOOKUP = build_alias_lookup()

# Category aliases (unchanged)
CATEGORY_ALIASES: dict[str, list[str]] = {
    "soil": ["soil", "माती", "मिट्टी", "ph", "soil health"],
    "weather": ["weather", "rain", "humidity", "हवामान", "पाऊस", "temperature"],
    "crop": ["crop", "पीक", "फसल", "sowing", "harvest", "variety"],
    "disease": ["disease", "blight", "virus", "fungal", "रोग", "करपा", "तेल्या", "बुरशी", "कीड", "अळी", "डाग", "कुज", "मर"],
    "pest": ["pest", "insect", "worm", "thrips", "bollworm", "कीड", "अळी"],
    "fertilizer": ["fertilizer", "manure", "urea", "dap", "mop", "खत", "युरिया", "डीएपी"],
    "irrigation": ["irrigation", "drip", "सिंचन", "ठिबक", "सिंचाई"],
    "scheme": ["scheme", "subsidy", "yojana", "योजना", "अनुदान"],
    "market": ["market", "mandi", "price", "msp", "भाव", "बाजार", "दर"],
}