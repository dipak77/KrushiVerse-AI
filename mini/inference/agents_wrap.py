"""Light W-AGENT wrap: intent-selected specialist notes for Mini context (S15)."""

from __future__ import annotations

from typing import Any


def collect_agent_notes(
    query: str,
    *,
    intents: list[str] | None = None,
    crop: str | None = None,
    location: str = "Pune",
    enable: bool = True,
) -> dict[str, Any]:
    """Run a subset of platform agents and return short notes (not full planner)."""
    if not enable:
        return {"notes": [], "agent_outputs": {}, "enabled": False}

    intents = intents or ["general"]
    crop = crop or "Cotton"
    notes: list[str] = []
    outputs: dict[str, Any] = {}
    ctx = {
        "query": query,
        "crop": crop,
        "location": location,
        "district": location,
        "acreage": 2.0,
        "temperature_c": 29.0,
        "humidity_pct": 70.0,
        "rainfall_mm": 5.0,
        "image_filename": "leaf_sample.jpg",
        "soil_card_text": "pH: 7.2, Nitrogen: 180 kg/ha, Phosphorus: 22 kg/ha, Potassium: 280 kg/ha",
    }

    try:
        if any(i in intents for i in ("disease", "pest", "general")) or "bollworm" in query.lower():
            from app.agents.disease_agent import DiseaseAgent

            d = DiseaseAgent().execute(query, ctx)
            outputs["disease"] = d
            if isinstance(d, dict):
                diag = d.get("diagnosis") or {}
                if isinstance(diag, dict):
                    name = (
                        diag.get("disease_identified_en")
                        or diag.get("disease_name")
                        or diag.get("label")
                        or diag.get("predicted_class")
                    )
                    if name:
                        notes.append(f"Disease agent: {name}")
                risk = d.get("outbreak_risk_analysis") or {}
                if isinstance(risk, dict) and risk.get("risk_level"):
                    notes.append(f"Outbreak risk: {risk.get('risk_level')}")
    except Exception as e:
        outputs["disease_error"] = str(e)

    try:
        if "weather" in intents:
            from app.agents.weather_agent import WeatherAgent

            w = WeatherAgent().execute(query, ctx)
            outputs["weather"] = w
            if isinstance(w, dict):
                notes.append(
                    f"Weather: {w.get('location', location)} temp={w.get('temperature_c', '?')}C"
                )
    except Exception as e:
        outputs["weather_error"] = str(e)

    try:
        if "fertilizer" in intents or "soil" in intents:
            from app.agents.fertilizer_agent import FertilizerAgent

            f = FertilizerAgent().execute(query, ctx)
            outputs["fertilizer"] = f
            notes.append("Fertilizer agent consulted (soil-test based plan preferred).")
    except Exception as e:
        outputs["fertilizer_error"] = str(e)

    return {
        "notes": notes[:6],
        "agent_outputs": {k: v for k, v in outputs.items() if not k.endswith("_error")},
        "errors": {k: v for k, v in outputs.items() if k.endswith("_error")},
        "enabled": True,
        "intents": intents,
        "crop": crop,
    }
