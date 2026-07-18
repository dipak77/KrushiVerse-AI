"""Sprint 0 draft taxonomy — extended and frozen in Sprint 1."""

from __future__ import annotations

from typing import Any

TAXONOMY: dict[str, Any] = {
    "version": "0.1.0-draft",
    "sprint": "S0",
    "categories": [
        "soil",
        "weather",
        "crop",
        "disease",
        "pest",
        "fertilizer",
        "irrigation",
        "scheme",
        "market",
        "finance",
        "machinery",
        "seed",
        "advisory",
        "general",
    ],
    "crop_stages": [
        "seed_selection",
        "seed_treatment",
        "land_preparation",
        "sowing",
        "germination",
        "vegetative",
        "flowering",
        "fruiting",
        "harvest",
        "storage",
    ],
    "crops": [
        {"id": "cotton", "name_en": "Cotton", "name_mr": "कापूस", "group": "fiber"},
        {"id": "soybean", "name_en": "Soybean", "name_mr": "सोयाबीन", "group": "oilseed"},
        {"id": "sugarcane", "name_en": "Sugarcane", "name_mr": "ऊस", "group": "cash"},
        {"id": "pomegranate", "name_en": "Pomegranate", "name_mr": "डाळिंब", "group": "fruit"},
        {"id": "onion", "name_en": "Onion", "name_mr": "कांदा", "group": "vegetable"},
        {"id": "rice", "name_en": "Rice", "name_mr": "भात", "group": "cereal"},
        {"id": "wheat", "name_en": "Wheat", "name_mr": "गहू", "group": "cereal"},
        {"id": "maize", "name_en": "Maize", "name_mr": "मका", "group": "cereal"},
        {"id": "tur", "name_en": "Tur (Pigeon Pea)", "name_mr": "तूर", "group": "pulse"},
        {"id": "gram", "name_en": "Gram (Chickpea)", "name_mr": "हरभरा", "group": "pulse"},
        {"id": "groundnut", "name_en": "Groundnut", "name_mr": "भुईमूग", "group": "oilseed"},
        {"id": "turmeric", "name_en": "Turmeric", "name_mr": "हळद", "group": "spice"},
        {"id": "grapes", "name_en": "Grapes", "name_mr": "द्राक्ष", "group": "fruit"},
        {"id": "banana", "name_en": "Banana", "name_mr": "केळी", "group": "fruit"},
        {"id": "mango", "name_en": "Mango", "name_mr": "आंबा", "group": "fruit"},
        {"id": "tomato", "name_en": "Tomato", "name_mr": "टोमॅटो", "group": "vegetable"},
        {"id": "chilli", "name_en": "Chilli", "name_mr": "मिरची", "group": "spice"},
        {"id": "sorghum", "name_en": "Sorghum (Jowar)", "name_mr": "ज्वारी", "group": "cereal"},
        {"id": "bajra", "name_en": "Bajra (Pearl Millet)", "name_mr": "बाजरी", "group": "cereal"},
        {"id": "mustard", "name_en": "Mustard", "name_mr": "मोहरी", "group": "oilseed"},
        {"id": "potato", "name_en": "Potato", "name_mr": "बटाटा", "group": "vegetable"},
        {"id": "orange", "name_en": "Orange (Nagpur Santra)", "name_mr": "संत्रा", "group": "fruit"},
    ],
    "soil_aspects": [
        "soil_type",
        "texture",
        "pH",
        "EC",
        "organic_carbon",
        "NPK",
        "micronutrients",
        "water_holding_capacity",
        "soil_health_card",
    ],
    "weather_aspects": [
        "rainfall",
        "temperature",
        "humidity",
        "wind",
        "heatwave",
        "frost",
        "monsoon",
        "forecast",
        "seasonal_outlook",
    ],
    "irrigation_types": ["drip", "flood", "sprinkler", "canal", "rainfed"],
    "scheme_examples": [
        "PM-KISAN",
        "PMFBY",
        "Soil Health Card",
        "PM-KUSUM",
        "PMKSY",
        "PoCRA",
    ],
    "languages": ["en", "mr", "hi"],
}


def list_categories() -> list[str]:
    return list(TAXONOMY["categories"])


def list_crops() -> list[dict[str, str]]:
    return list(TAXONOMY["crops"])
