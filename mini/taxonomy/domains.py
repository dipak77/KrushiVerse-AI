"""Frozen agriculture domain taxonomy — Sprint 1 (v1.0.0).

This is the single source of truth for categories, crops, stages, and domain facets.
Later workers (normalize, QA synth, KG) must import from here — not hardcode lists.
"""

from __future__ import annotations

from typing import Any

TAXONOMY_VERSION = "1.0.0"
TAXONOMY_STATUS = "frozen"  # draft | frozen
TAXONOMY_SPRINT = "S1"

TAXONOMY: dict[str, Any] = {
    "version": TAXONOMY_VERSION,
    "status": TAXONOMY_STATUS,
    "sprint": TAXONOMY_SPRINT,
    "languages": ["en", "mr", "hi"],
    "categories": [
        {
            "id": "soil",
            "name_en": "Soil",
            "name_mr": "माती",
            "name_hi": "मिट्टी",
            "aspects": [
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
        },
        {
            "id": "weather",
            "name_en": "Weather",
            "name_mr": "हवामान",
            "name_hi": "मौसम",
            "aspects": [
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
        },
        {
            "id": "crop",
            "name_en": "Crop",
            "name_mr": "पीक",
            "name_hi": "फसल",
            "aspects": ["calendar", "variety", "growth_stage", "package_of_practice"],
        },
        {
            "id": "disease",
            "name_en": "Disease",
            "name_mr": "रोग",
            "name_hi": "रोग",
            "aspects": ["name", "symptoms", "causes", "prevention", "treatment", "severity", "images"],
        },
        {
            "id": "pest",
            "name_en": "Pest",
            "name_mr": "कीड",
            "name_hi": "कीट",
            "aspects": ["name", "lifecycle", "symptoms", "biological", "chemical", "organic"],
        },
        {
            "id": "fertilizer",
            "name_en": "Fertilizer",
            "name_mr": "खत",
            "name_hi": "उर्वरक",
            "aspects": ["organic", "chemical", "micronutrients", "schedule", "quantity", "timing"],
        },
        {
            "id": "irrigation",
            "name_en": "Irrigation",
            "name_mr": "सिंचन",
            "name_hi": "सिंचाई",
            "aspects": ["drip", "flood", "sprinkler", "canal", "rainfed", "scheduling"],
        },
        {
            "id": "scheme",
            "name_en": "Government Scheme",
            "name_mr": "शासकीय योजना",
            "name_hi": "सरकारी योजना",
            "aspects": ["eligibility", "benefits", "documents", "portal"],
        },
        {
            "id": "market",
            "name_en": "Market",
            "name_mr": "बाजार",
            "name_hi": "बाजार",
            "aspects": ["msp", "mandi_price", "export", "demand", "storage"],
        },
        {
            "id": "finance",
            "name_en": "Finance",
            "name_mr": "वित्त",
            "name_hi": "वित्त",
            "aspects": ["loan", "insurance", "subsidy", "roi", "cost_of_cultivation"],
        },
        {
            "id": "machinery",
            "name_en": "Machinery",
            "name_mr": "यंत्रसामग्री",
            "name_hi": "मशीनरी",
            "aspects": ["tractor", "drone", "sprayer", "seeder", "harvester"],
        },
        {
            "id": "seed",
            "name_en": "Seed Variety",
            "name_mr": "बियाणे",
            "name_hi": "बीज",
            "aspects": ["variety", "hybrid", "treatment", "source"],
        },
        {
            "id": "advisory",
            "name_en": "Advisory",
            "name_mr": "सल्ला",
            "name_hi": "सलाह",
            "aspects": ["stage_checklist", "ipm", "weather_linked"],
        },
        {
            "id": "general",
            "name_en": "General",
            "name_mr": "सामान्य",
            "name_hi": "सामान्य",
            "aspects": [],
        },
    ],
    "crop_stages": [
        {"id": "seed_selection", "name_en": "Seed Selection", "name_mr": "बियाणे निवड", "name_hi": "बीज चयन"},
        {"id": "seed_treatment", "name_en": "Seed Treatment", "name_mr": "बीजप्रक्रिया", "name_hi": "बीज उपचार"},
        {"id": "land_preparation", "name_en": "Land Preparation", "name_mr": "जमीन तयारी", "name_hi": "भूमि तैयारी"},
        {"id": "sowing", "name_en": "Sowing", "name_mr": "पेरणी", "name_hi": "बुवाई"},
        {"id": "germination", "name_en": "Germination", "name_mr": "उगवण", "name_hi": "अंकुरण"},
        {"id": "vegetative", "name_en": "Vegetative Growth", "name_mr": "वाढीची अवस्था", "name_hi": "वनस्पति वृद्धि"},
        {"id": "flowering", "name_en": "Flowering", "name_mr": "फुलोरा", "name_hi": "फूलन"},
        {"id": "fruiting", "name_en": "Fruiting / Boll / Pod", "name_mr": "फळधारणा", "name_hi": "फलन"},
        {"id": "harvest", "name_en": "Harvest", "name_mr": "कापणी", "name_hi": "कटाई"},
        {"id": "storage", "name_en": "Storage", "name_mr": "साठवण", "name_hi": "भंडारण"},
    ],
    "crops": [
        {"id": "cotton", "name_en": "Cotton", "name_mr": "कापूस", "name_hi": "कपास", "group": "fiber", "scientific": "Gossypium hirsutum"},
        {"id": "soybean", "name_en": "Soybean", "name_mr": "सोयाबीन", "name_hi": "सोयाबीन", "group": "oilseed", "scientific": "Glycine max"},
        {"id": "sugarcane", "name_en": "Sugarcane", "name_mr": "ऊस", "name_hi": "गन्ना", "group": "cash", "scientific": "Saccharum officinarum"},
        {"id": "pomegranate", "name_en": "Pomegranate", "name_mr": "डाळिंब", "name_hi": "अनार", "group": "fruit", "scientific": "Punica granatum"},
        {"id": "onion", "name_en": "Onion", "name_mr": "कांदा", "name_hi": "प्याज", "group": "vegetable", "scientific": "Allium cepa"},
        {"id": "rice", "name_en": "Rice", "name_mr": "भात", "name_hi": "चावल", "group": "cereal", "scientific": "Oryza sativa"},
        {"id": "wheat", "name_en": "Wheat", "name_mr": "गहू", "name_hi": "गेहूं", "group": "cereal", "scientific": "Triticum aestivum"},
        {"id": "maize", "name_en": "Maize", "name_mr": "मका", "name_hi": "मक्का", "group": "cereal", "scientific": "Zea mays"},
        {"id": "tur", "name_en": "Tur (Pigeon Pea)", "name_mr": "तूर", "name_hi": "अरहर", "group": "pulse", "scientific": "Cajanus cajan"},
        {"id": "gram", "name_en": "Gram (Chickpea)", "name_mr": "हरभरा", "name_hi": "चना", "group": "pulse", "scientific": "Cicer arietinum"},
        {"id": "groundnut", "name_en": "Groundnut", "name_mr": "भुईमूग", "name_hi": "मूंगफली", "group": "oilseed", "scientific": "Arachis hypogaea"},
        {"id": "turmeric", "name_en": "Turmeric", "name_mr": "हळद", "name_hi": "हल्दी", "group": "spice", "scientific": "Curcuma longa"},
        {"id": "grapes", "name_en": "Grapes", "name_mr": "द्राक्ष", "name_hi": "अंगूर", "group": "fruit", "scientific": "Vitis vinifera"},
        {"id": "banana", "name_en": "Banana", "name_mr": "केळी", "name_hi": "केला", "group": "fruit", "scientific": "Musa spp."},
        {"id": "mango", "name_en": "Mango", "name_mr": "आंबा", "name_hi": "आम", "group": "fruit", "scientific": "Mangifera indica"},
        {"id": "tomato", "name_en": "Tomato", "name_mr": "टोमॅटो", "name_hi": "टमाटर", "group": "vegetable", "scientific": "Solanum lycopersicum"},
        {"id": "chilli", "name_en": "Chilli", "name_mr": "मिरची", "name_hi": "मिर्च", "group": "spice", "scientific": "Capsicum annuum"},
        {"id": "sorghum", "name_en": "Sorghum (Jowar)", "name_mr": "ज्वारी", "name_hi": "ज्वार", "group": "cereal", "scientific": "Sorghum bicolor"},
        {"id": "bajra", "name_en": "Bajra (Pearl Millet)", "name_mr": "बाजरी", "name_hi": "बाजरा", "group": "cereal", "scientific": "Pennisetum glaucum"},
        {"id": "mustard", "name_en": "Mustard", "name_mr": "मोहरी", "name_hi": "सरसों", "group": "oilseed", "scientific": "Brassica juncea"},
        {"id": "potato", "name_en": "Potato", "name_mr": "बटाटा", "name_hi": "आलू", "group": "vegetable", "scientific": "Solanum tuberosum"},
        {"id": "orange", "name_en": "Orange (Nagpur Santra)", "name_mr": "संत्रा", "name_hi": "संतरा", "group": "fruit", "scientific": "Citrus reticulata"},
    ],
    "irrigation_types": [
        {"id": "drip", "name_en": "Drip", "name_mr": "ठिबक", "name_hi": "ड्रिप"},
        {"id": "flood", "name_en": "Flood / Furrow", "name_mr": "पाटपाणी", "name_hi": "बाढ़ सिंचाई"},
        {"id": "sprinkler", "name_en": "Sprinkler", "name_mr": "तुषार", "name_hi": "स्प्रिंकलर"},
        {"id": "canal", "name_en": "Canal", "name_mr": "कालवा", "name_hi": "नहर"},
        {"id": "rainfed", "name_en": "Rainfed", "name_mr": "कोरडवाहू", "name_hi": "वर्षा आधारित"},
    ],
    "fertilizer_types": [
        {"id": "urea", "name_en": "Urea", "nutrient": "N"},
        {"id": "dap", "name_en": "DAP", "nutrient": "N+P"},
        {"id": "mop", "name_en": "MOP", "nutrient": "K"},
        {"id": "ssp", "name_en": "SSP", "nutrient": "P"},
        {"id": "fym", "name_en": "FYM", "nutrient": "organic"},
        {"id": "vermicompost", "name_en": "Vermicompost", "nutrient": "organic"},
        {"id": "npk_water_soluble", "name_en": "19:19:19", "nutrient": "NPK"},
        {"id": "gypsum", "name_en": "Gypsum", "nutrient": "Ca+S"},
    ],
    "machinery_types": [
        {"id": "tractor", "name_en": "Tractor", "name_mr": "ट्रॅक्टर", "name_hi": "ट्रैक्टर"},
        {"id": "drone", "name_en": "Drone", "name_mr": "ड्रोन", "name_hi": "ड्रोन"},
        {"id": "sprayer", "name_en": "Sprayer", "name_mr": "फवारणी यंत्र", "name_hi": "स्प्रेयर"},
        {"id": "seeder", "name_en": "Seeder / Planter", "name_mr": "पेरणी यंत्र", "name_hi": "सीडर"},
        {"id": "harvester", "name_en": "Harvester", "name_mr": "कापणी यंत्र", "name_hi": "हार्वेस्टर"},
    ],
    "scheme_catalog": [
        {"id": "pm_kisan", "name_en": "PM-KISAN"},
        {"id": "pmfby", "name_en": "PMFBY"},
        {"id": "soil_health_card", "name_en": "Soil Health Card"},
        {"id": "pm_kusum", "name_en": "PM-KUSUM"},
        {"id": "pmksy", "name_en": "PMKSY Micro-Irrigation"},
        {"id": "pocra", "name_en": "PoCRA"},
        {"id": "kcc", "name_en": "Kisan Credit Card"},
        {"id": "midh", "name_en": "MIDH / NHM"},
        {"id": "enam", "name_en": "eNAM"},
        {"id": "nfsm", "name_en": "NFSM"},
    ],
    "soil_types": [
        {"id": "black_cotton", "name_en": "Black Cotton Soil (Vertisol)", "name_mr": "काळी कसदार माती"},
        {"id": "medium_black", "name_en": "Medium Black Soil", "name_mr": "मध्यम काळी जमीन"},
        {"id": "red_laterite", "name_en": "Red & Laterite Soil", "name_mr": "तांबडी व जांभी जमीन"},
        {"id": "alluvial", "name_en": "Alluvial Soil", "name_mr": "गाळाची जमीन"},
        {"id": "sandy_loam", "name_en": "Sandy Loam", "name_mr": "वालुकामय पोयटा"},
        {"id": "clay_loam", "name_en": "Clay Loam", "name_mr": "चिकण पोयटा"},
    ],
}


def list_categories() -> list[str]:
    return [c["id"] for c in TAXONOMY["categories"]]


def list_category_details() -> list[dict]:
    return list(TAXONOMY["categories"])


def list_crops() -> list[dict]:
    return list(TAXONOMY["crops"])


def list_crop_names_en() -> list[str]:
    return [c["name_en"] for c in TAXONOMY["crops"]]


def list_crop_stages() -> list[dict]:
    return list(TAXONOMY["crop_stages"])


def get_crop_by_en(name_en: str) -> dict | None:
    for c in TAXONOMY["crops"]:
        if c["name_en"].lower() == (name_en or "").lower():
            return c
    return None


def taxonomy_summary() -> dict[str, Any]:
    return {
        "version": TAXONOMY_VERSION,
        "status": TAXONOMY_STATUS,
        "sprint": TAXONOMY_SPRINT,
        "categories": len(TAXONOMY["categories"]),
        "crops": len(TAXONOMY["crops"]),
        "crop_stages": len(TAXONOMY["crop_stages"]),
        "irrigation_types": len(TAXONOMY["irrigation_types"]),
        "fertilizer_types": len(TAXONOMY["fertilizer_types"]),
        "machinery_types": len(TAXONOMY["machinery_types"]),
        "schemes": len(TAXONOMY["scheme_catalog"]),
        "soil_types": len(TAXONOMY["soil_types"]),
        "languages": TAXONOMY["languages"],
    }
