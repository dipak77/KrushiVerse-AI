"""Curated gold QA sets for Mini evaluation (Sprint 13).

Domains: disease, fertilizer, schemes, market, regional (Maharashtra crops).
These are compact, verified-style items for gate scorecards — not full lake dumps.
"""

from __future__ import annotations

from typing import Any


def disease_gold() -> list[dict[str, Any]]:
    return [
        {
            "id": "gold-dis-001",
            "category": "disease",
            "crop": "Cotton",
            "region": {"state": "Maharashtra", "zone": "Vidarbha"},
            "language": "en",
            "question": "What is a practical IPM approach for pink bollworm in cotton?",
            "answer": (
                "Scout regularly, use pheromone traps, follow ETL thresholds, "
                "prefer resistant hybrids where available, and apply labeled insecticides only when needed."
            ),
            "must_keywords": ["scout", "trap", "etl", "label"],
            "source": "curated_gold",
        },
        {
            "id": "gold-dis-002",
            "category": "disease",
            "crop": "Pomegranate",
            "region": {"state": "Maharashtra", "district": "Solapur"},
            "language": "en",
            "question": "How should bacterial blight on pomegranate be managed?",
            "answer": (
                "Remove infected plant parts, avoid overhead irrigation during wet weather, "
                "apply copper-based protectants as per label, and maintain orchard sanitation."
            ),
            "must_keywords": ["sanitation", "copper", "irrigation"],
            "source": "curated_gold",
        },
        {
            "id": "gold-dis-003",
            "category": "disease",
            "crop": "Grape",
            "region": {"state": "Maharashtra", "district": "Nashik"},
            "language": "en",
            "question": "What reduces downy mildew risk in grapes?",
            "answer": (
                "Improve canopy air flow, avoid prolonged leaf wetness, scout early, "
                "and use labeled fungicides in rotation when weather favors infection."
            ),
            "must_keywords": ["canopy", "wetness", "fungicide"],
            "source": "curated_gold",
        },
        {
            "id": "gold-dis-004",
            "category": "disease",
            "crop": "Wheat",
            "language": "hi",
            "question": "गेहूँ में रस्ट रोग के लक्षण आने पर क्या करें?",
            "answer": (
                "Resistant varieties prefer करें, field scouting बढ़ाएँ, and use labeled fungicide "
                "as recommended by local agri officer; avoid random tank mixes."
            ),
            "must_keywords": ["resistant", "scout", "fungicide"],
            "source": "curated_gold",
        },
    ]


def fertilizer_gold() -> list[dict[str, Any]]:
    return [
        {
            "id": "gold-fert-001",
            "category": "fertilizer",
            "crop": "Cotton",
            "language": "en",
            "question": "Should fertilizer rates ignore soil test results?",
            "answer": (
                "No. Base NPK and micronutrient plans on recent soil test reports, crop stage, "
                "and previous yield; avoid blanket high doses that waste money and pollute."
            ),
            "must_keywords": ["soil", "test", "npk"],
            "source": "curated_gold",
        },
        {
            "id": "gold-fert-002",
            "category": "fertilizer",
            "crop": "Onion",
            "region": {"state": "Maharashtra", "district": "Nashik"},
            "language": "en",
            "question": "How to plan nitrogen for onion?",
            "answer": (
                "Split nitrogen across stages, prefer soil-test based rates, maintain balanced "
                "P and K, and avoid excess N that delays bulb maturity."
            ),
            "must_keywords": ["split", "soil", "nitrogen"],
            "source": "curated_gold",
        },
        {
            "id": "gold-fert-003",
            "category": "fertilizer",
            "language": "mr",
            "question": "माती तपासणी न करता खत टाकावे का?",
            "answer": (
                "नाही. Soil test नंतर NPK व micronutrient ठरवा. जास्त खत खर्च व residual risk वाढवते."
            ),
            "must_keywords": ["soil", "test"],
            "source": "curated_gold",
        },
    ]


def schemes_gold() -> list[dict[str, Any]]:
    return [
        {
            "id": "gold-sch-001",
            "category": "scheme",
            "language": "en",
            "question": "What should farmers verify before claiming a government agri scheme benefit?",
            "answer": (
                "Check official eligibility, required documents (Aadhaar, land records, bank), "
                "application window, and confirm details on official portals or agri department offices."
            ),
            "must_keywords": ["eligibility", "document", "official"],
            "source": "curated_gold",
        },
        {
            "id": "gold-sch-002",
            "category": "scheme",
            "language": "en",
            "question": "Is PM-KISAN related to income support for landholding farmers?",
            "answer": (
                "Yes. PM-KISAN provides income support installments to eligible landholding farmer "
                "families as per central guidelines; verify enrollment status officially."
            ),
            "must_keywords": ["pm-kisan", "income", "eligible"],
            "source": "curated_gold",
        },
        {
            "id": "gold-sch-003",
            "category": "scheme",
            "language": "hi",
            "question": "स्कीम के लिए फर्जी दस्तावेज़ दें?",
            "answer": (
                "नहीं। केवल सही दस्तावेज़ और official portal/कार्यालय से आवेदन करें। फर्जी जानकारी कानूनी समस्या ला सकती है।"
            ),
            "must_keywords": ["official", "document"],
            "source": "curated_gold",
        },
    ]


def market_gold() -> list[dict[str, Any]]:
    return [
        {
            "id": "gold-mkt-001",
            "category": "market",
            "crop": "Soybean",
            "region": {"state": "Maharashtra"},
            "language": "en",
            "question": "How should a farmer use market prices for selling soybean?",
            "answer": (
                "Compare nearby mandi prices, quality grades, transport cost, and MSP context; "
                "avoid distress sale when storage and cashflow allow waiting for better rates."
            ),
            "must_keywords": ["mandi", "price", "quality"],
            "source": "curated_gold",
        },
        {
            "id": "gold-mkt-002",
            "category": "market",
            "crop": "Onion",
            "region": {"state": "Maharashtra", "district": "Lasalgaon"},
            "language": "en",
            "question": "What factors move onion prices in Maharashtra markets?",
            "answer": (
                "Arrivals, storage stocks, weather, festival demand, transport, and quality "
                "grade strongly influence onion mandi prices."
            ),
            "must_keywords": ["arrival", "demand", "quality"],
            "source": "curated_gold",
        },
        {
            "id": "gold-mkt-003",
            "category": "market",
            "language": "mr",
            "question": "मंडी भाव कसे तपासावे?",
            "answer": (
                "Official eNAM/agmarknet किंवा जिल्हा कृषी/मंडी माहिती वापरा; grade आणि transport खर्च ध्यानात घ्या."
            ),
            "must_keywords": ["mandi", "price"],
            "source": "curated_gold",
        },
    ]


def regional_mh_gold() -> list[dict[str, Any]]:
    """Maharashtra regional crop correctness probes."""
    return [
        {
            "id": "gold-reg-001",
            "category": "crop",
            "crop": "Cotton",
            "region": {"state": "Maharashtra", "zone": "Vidarbha"},
            "language": "en",
            "question": "Which major cash crop is widely grown in Vidarbha, Maharashtra?",
            "answer": "Cotton is a major cash crop across much of Vidarbha in Maharashtra.",
            "must_keywords": ["cotton", "vidarbha"],
            "source": "curated_gold",
        },
        {
            "id": "gold-reg-002",
            "category": "crop",
            "crop": "Orange",
            "region": {"state": "Maharashtra", "district": "Nagpur"},
            "language": "en",
            "question": "Nagpur is famous for which fruit crop?",
            "answer": "Nagpur is famous for oranges (Nagpur santra).",
            "must_keywords": ["orange", "nagpur"],
            "source": "curated_gold",
        },
        {
            "id": "gold-reg-003",
            "category": "crop",
            "crop": "Grape",
            "region": {"state": "Maharashtra", "district": "Nashik"},
            "language": "en",
            "question": "Nashik district is well known for which horticulture crop?",
            "answer": "Nashik is a major grape-producing district in Maharashtra.",
            "must_keywords": ["grape", "nashik"],
            "source": "curated_gold",
        },
        {
            "id": "gold-reg-004",
            "category": "crop",
            "crop": "Pomegranate",
            "region": {"state": "Maharashtra", "district": "Solapur"},
            "language": "en",
            "question": "Solapur region is associated with which fruit crop in Maharashtra?",
            "answer": "Pomegranate is widely grown in and around Solapur, Maharashtra.",
            "must_keywords": ["pomegranate", "solapur"],
            "source": "curated_gold",
        },
        {
            "id": "gold-reg-005",
            "category": "weather",
            "region": {"state": "Maharashtra"},
            "language": "en",
            "question": "Should farmers spray pesticides just before heavy rain in Maharashtra monsoon?",
            "answer": (
                "Usually no. Rain soon after spray can wash off chemical, waste money, and reduce control. "
                "Prefer a dry window."
            ),
            "must_keywords": ["rain", "spray", "wash"],
            "source": "curated_gold",
        },
    ]


def load_all_gold(*, categories: list[str] | None = None) -> list[dict[str, Any]]:
    packs = {
        "disease": disease_gold(),
        "fertilizer": fertilizer_gold(),
        "scheme": schemes_gold(),
        "market": market_gold(),
        "regional": regional_mh_gold(),
    }
    if categories:
        keys = [c.lower().strip() for c in categories]
        rows: list[dict[str, Any]] = []
        for k in keys:
            rows.extend(packs.get(k, []))
        return rows
    out: list[dict[str, Any]] = []
    for rows in packs.values():
        out.extend(rows)
    return out


def gold_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    for r in rows:
        c = str(r.get("category") or "other")
        by_cat[c] = by_cat.get(c, 0) + 1
        lang = str(r.get("language") or "en")
        by_lang[lang] = by_lang.get(lang, 0) + 1
    return {"n": len(rows), "by_category": by_cat, "by_language": by_lang}
