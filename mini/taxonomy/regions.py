"""Region hierarchy — Maharashtra focus + India stubs (Sprint 1 freeze)."""

from __future__ import annotations

from typing import Any

# Agro-climatic / administrative hierarchy used by normalize + RAG filters
REGIONS: dict[str, Any] = {
    "version": "1.0.0",
    "country": {
        "id": "IN",
        "name_en": "India",
        "name_mr": "भारत",
        "name_hi": "भारत",
    },
    "states": [
        {
            "id": "MH",
            "name_en": "Maharashtra",
            "name_mr": "महाराष्ट्र",
            "name_hi": "महाराष्ट्र",
            "zones": [
                {
                    "id": "mh_scarcity",
                    "name_en": "Western Maharashtra Scarcity",
                    "districts": ["Solapur", "Ahmednagar", "Pune", "Satara"],
                },
                {
                    "id": "mh_marathwada",
                    "name_en": "Marathwada",
                    "districts": [
                        "Chhatrapati Sambhajinagar",
                        "Aurangabad",
                        "Beed",
                        "Latur",
                        "Nanded",
                        "Parbhani",
                        "Hingoli",
                        "Jalna",
                        "Dharashiv",
                        "Osmanabad",
                    ],
                },
                {
                    "id": "mh_vidarbha",
                    "name_en": "Vidarbha",
                    "districts": [
                        "Nagpur",
                        "Amravati",
                        "Akola",
                        "Yavatmal",
                        "Wardha",
                        "Buldhana",
                        "Washim",
                        "Chandrapur",
                        "Gadchiroli",
                        "Gondia",
                        "Bhandara",
                    ],
                },
                {
                    "id": "mh_konkan",
                    "name_en": "Konkan",
                    "districts": ["Thane", "Raigad", "Ratnagiri", "Sindhudurg", "Palghar", "Mumbai"],
                },
                {
                    "id": "mh_western_ghat",
                    "name_en": "Western Ghat / Transition",
                    "districts": ["Kolhapur", "Sangli", "Satara", "Nashik"],
                },
                {
                    "id": "mh_khandesh",
                    "name_en": "Khandesh / North MH",
                    "districts": ["Jalgaon", "Dhule", "Nandurbar"],
                },
            ],
        },
        {
            "id": "MP",
            "name_en": "Madhya Pradesh",
            "name_mr": "मध्य प्रदेश",
            "name_hi": "मध्य प्रदेश",
            "zones": [
                {
                    "id": "mp_malwa",
                    "name_en": "Malwa / Central India plains",
                    "districts": ["Indore", "Ujjain", "Bhopal"],
                }
            ],
        },
        {
            "id": "KA",
            "name_en": "Karnataka",
            "name_mr": "कर्नाटक",
            "name_hi": "कर्नाटक",
            "zones": [{"id": "ka_north", "name_en": "North Karnataka", "districts": ["Belagavi", "Kalaburagi"]}],
        },
        {
            "id": "GJ",
            "name_en": "Gujarat",
            "name_mr": "गुजरात",
            "name_hi": "गुजरात",
            "zones": [{"id": "gj_saurashtra", "name_en": "Saurashtra", "districts": ["Rajkot", "Junagadh"]}],
        },
    ],
}

# District aliases (lowercase) -> (state_id, canonical district)
DISTRICT_ALIASES: dict[str, tuple[str, str]] = {
    "pune": ("MH", "Pune"),
    "पुणे": ("MH", "Pune"),
    "solapur": ("MH", "Solapur"),
    "सोलापूर": ("MH", "Solapur"),
    "latur": ("MH", "Latur"),
    "लातूर": ("MH", "Latur"),
    "akola": ("MH", "Akola"),
    "अकोला": ("MH", "Akola"),
    "nagpur": ("MH", "Nagpur"),
    "नागपूर": ("MH", "Nagpur"),
    "nashik": ("MH", "Nashik"),
    "नाशिक": ("MH", "Nashik"),
    "nasik": ("MH", "Nashik"),
    "kolhapur": ("MH", "Kolhapur"),
    "कोल्हापूर": ("MH", "Kolhapur"),
    "sangli": ("MH", "Sangli"),
    "सांगली": ("MH", "Sangli"),
    "jalgaon": ("MH", "Jalgaon"),
    "जळगाव": ("MH", "Jalgaon"),
    "aurangabad": ("MH", "Chhatrapati Sambhajinagar"),
    "chhatrapati sambhajinagar": ("MH", "Chhatrapati Sambhajinagar"),
    "sambhajinagar": ("MH", "Chhatrapati Sambhajinagar"),
    "beed": ("MH", "Beed"),
    "बीड": ("MH", "Beed"),
    "nanded": ("MH", "Nanded"),
    "नांदेड": ("MH", "Nanded"),
    "yavatmal": ("MH", "Yavatmal"),
    "यवतमाळ": ("MH", "Yavatmal"),
    "wardha": ("MH", "Wardha"),
    "वर्धा": ("MH", "Wardha"),
    "mumbai": ("MH", "Mumbai"),
    "मुंबई": ("MH", "Mumbai"),
    "ahmednagar": ("MH", "Ahmednagar"),
    "अहमदनगर": ("MH", "Ahmednagar"),
    "ahilyanagar": ("MH", "Ahmednagar"),
    "dhule": ("MH", "Dhule"),
    "धुळे": ("MH", "Dhule"),
    "parbhani": ("MH", "Parbhani"),
    "परभणी": ("MH", "Parbhani"),
    "indore": ("MP", "Indore"),
    "इंदौर": ("MP", "Indore"),
}


def list_states() -> list[dict]:
    return list(REGIONS["states"])


def list_mh_districts() -> list[str]:
    mh = next(s for s in REGIONS["states"] if s["id"] == "MH")
    districts: list[str] = []
    for z in mh["zones"]:
        districts.extend(z["districts"])
    # unique preserve order
    seen = set()
    out = []
    for d in districts:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def resolve_district(text: str) -> dict | None:
    if not text:
        return None
    key = text.strip().lower()
    if key in DISTRICT_ALIASES:
        state_id, district = DISTRICT_ALIASES[key]
        state = next((s for s in REGIONS["states"] if s["id"] == state_id), None)
        zone = None
        if state:
            for z in state["zones"]:
                if district in z["districts"] or any(
                    district.lower() == d.lower() for d in z["districts"]
                ):
                    zone = z["name_en"]
                    break
        return {
            "state_id": state_id,
            "state": state["name_en"] if state else state_id,
            "district": district,
            "zone": zone,
            "country": "India",
        }
    # fuzzy: substring
    for alias, (state_id, district) in DISTRICT_ALIASES.items():
        if alias in key or key in alias:
            return resolve_district(alias)
    return None


def region_to_standard(district: str | None = None, state: str | None = "Maharashtra") -> dict:
    resolved = resolve_district(district or "") if district else None
    if resolved:
        return resolved
    return {
        "state_id": "MH" if (state or "").lower().startswith("maha") else None,
        "state": state or "Maharashtra",
        "district": district,
        "zone": None,
        "country": "India",
    }
