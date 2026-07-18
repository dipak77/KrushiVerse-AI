"""Unit vocabulary for agriculture records (Sprint 1 freeze)."""

from __future__ import annotations

from typing import Any

UNITS: dict[str, Any] = {
    "version": "1.0.0",
    "dimensions": {
        "mass": {
            "canonical": "kg",
            "aliases": {
                "kg": 1.0,
                "kilogram": 1.0,
                "किलो": 1.0,
                "g": 0.001,
                "gram": 0.001,
                "quintal": 100.0,
                "q": 100.0,
                "क्विंटल": 100.0,
                "tonne": 1000.0,
                "t": 1000.0,
                "टन": 1000.0,
                "lb": 0.453592,
            },
        },
        "area": {
            "canonical": "ha",
            "aliases": {
                "ha": 1.0,
                "hectare": 1.0,
                "हेक्टर": 1.0,
                "acre": 0.404686,
                "ac": 0.404686,
                "एकर": 0.404686,
                "guntha": 0.010117,
                "गुंठा": 0.010117,
            },
        },
        "rate_mass_per_area": {
            "canonical": "kg/ha",
            "aliases": {
                "kg/ha": 1.0,
                "kg per ha": 1.0,
                "kg/acre": 2.47105,  # convert to kg/ha multiplier when reading acre rates
                "किलो/हेक्टर": 1.0,
                "किलो/एकर": 2.47105,
            },
        },
        "temperature": {
            "canonical": "C",
            "aliases": {
                "c": "C",
                "°c": "C",
                "celsius": "C",
                "सेल्सिअस": "C",
                "f": "F",
                "°f": "F",
                "fahrenheit": "F",
            },
        },
        "rainfall": {
            "canonical": "mm",
            "aliases": {
                "mm": 1.0,
                "millimeter": 1.0,
                "मिमी": 1.0,
                "cm": 10.0,
                "inch": 25.4,
                "इंच": 25.4,
            },
        },
        "currency": {
            "canonical": "INR",
            "aliases": {
                "inr": "INR",
                "rs": "INR",
                "₹": "INR",
                "rupee": "INR",
                "रुपये": "INR",
            },
        },
        "price": {
            "canonical": "INR/quintal",
            "aliases": {
                "rs/q": "INR/quintal",
                "₹/q": "INR/quintal",
                "rs/quintal": "INR/quintal",
                "inr/quintal": "INR/quintal",
                "रु/क्विंटल": "INR/quintal",
            },
        },
        "volume_water": {
            "canonical": "L",
            "aliases": {
                "l": 1.0,
                "liter": 1.0,
                "litre": 1.0,
                "लिटर": 1.0,
                "ml": 0.001,
                "m3": 1000.0,
            },
        },
        "concentration": {
            "canonical": "g/L",
            "aliases": {
                "g/l": "g/L",
                "g/L": "g/L",
                "ml/l": "ml/L",
                "ppm": "ppm",
                "ग्रॅम/लिटर": "g/L",
            },
        },
        "percentage": {
            "canonical": "%",
            "aliases": {"%": 1.0, "pct": 1.0, "percent": 1.0, "टक्के": 1.0},
        },
        "ec": {
            "canonical": "dS/m",
            "aliases": {"ds/m": "dS/m", "mmhos/cm": "dS/m", "ms/cm": "dS/m"},
        },
        "ph": {
            "canonical": "pH",
            "aliases": {"ph": "pH", "पीएच": "pH"},
        },
    },
    "preferred_display": {
        "fertilizer_rate": "kg/acre",  # farmer-facing MH convention often per acre
        "soil_nutrient": "kg/ha",
        "mandi_price": "INR/quintal",
        "rainfall": "mm",
        "temperature": "°C",
    },
}


def list_dimensions() -> list[str]:
    return list(UNITS["dimensions"].keys())


def normalize_unit_token(token: str) -> str | None:
    t = (token or "").strip().lower().replace("°", "")
    for dim, spec in UNITS["dimensions"].items():
        aliases = spec.get("aliases") or {}
        for alias, val in aliases.items():
            if alias.lower().replace("°", "") == t:
                if isinstance(val, str):
                    return val
                return spec["canonical"]
    return None


def convert_mass(value: float, from_unit: str, to_unit: str = "kg") -> float | None:
    mass = UNITS["dimensions"]["mass"]["aliases"]
    fu = from_unit.lower()
    tu = to_unit.lower()
    if fu not in mass or tu not in mass:
        return None
    kg = value * float(mass[fu])
    return kg / float(mass[tu])


def convert_area(value: float, from_unit: str, to_unit: str = "ha") -> float | None:
    area = UNITS["dimensions"]["area"]["aliases"]
    fu = from_unit.lower()
    tu = to_unit.lower()
    if fu not in area or tu not in area:
        return None
    ha = value * float(area[fu])
    return ha / float(area[tu])
