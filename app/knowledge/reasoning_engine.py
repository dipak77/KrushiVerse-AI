"""Chain-of-Thought (CoT) Agronomic Reasoning & Mathematical Engine.

Provides:
1. Land Acreage & NPK Fertilizer Dosage Math
2. Irrigation Physics & Evapotranspiration Drip Calculator
3. Predictive Weather Risk & Disease Outbreak Assessment
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from mini.taxonomy.aliases import resolve_crops_smart


@dataclass
class FertilizerMathResult:
    crop: str
    acres: float
    urea_kg: float
    ssp_kg: float
    mop_kg: float
    fym_tonnes: float
    explanation_mr: str
    explanation_en: str


@dataclass
class IrrigationMathResult:
    crop: str
    drip_hours_per_day: float
    liters_per_plant_day: float
    status: str
    recommendation_mr: str
    recommendation_en: str


@dataclass
class DiseaseRiskResult:
    risk_level: str  # HIGH, MEDIUM, LOW
    primary_threat: str
    probability_pct: int
    advisory_mr: str
    advisory_en: str


class AgronomicReasoningEngine:
    """CoT Engine for Agricultural Physics, Dosage Math, and Risk Prediction."""

    # Recommended NPK per acre (N, P, K in kg/acre) & FYM (tonnes/acre)
    CROP_NPK_STANDARD: dict[str, dict[str, float]] = {
        "Cotton": {"N": 50, "P": 25, "K": 25, "Urea": 100, "SSP": 150, "MOP": 40, "FYM": 5},
        "Soybean": {"N": 12, "P": 24, "K": 16, "Urea": 30, "SSP": 150, "MOP": 30, "FYM": 4},
        "Sugarcane": {"N": 100, "P": 45, "K": 45, "Urea": 220, "SSP": 280, "MOP": 75, "FYM": 10},
        "Pomegranate": {"N": 35, "P": 25, "K": 25, "Urea": 75, "SSP": 150, "MOP": 40, "FYM": 8},
        "Onion": {"N": 40, "P": 20, "K": 20, "Urea": 90, "SSP": 125, "MOP": 35, "FYM": 6},
        "Rice": {"N": 40, "P": 20, "K": 20, "Urea": 90, "SSP": 125, "MOP": 35, "FYM": 5},
        "Wheat": {"N": 48, "P": 24, "K": 24, "Urea": 100, "SSP": 150, "MOP": 40, "FYM": 5},
        "Maize": {"N": 48, "P": 24, "K": 24, "Urea": 100, "SSP": 150, "MOP": 40, "FYM": 5},
        "Tur": {"N": 10, "P": 20, "K": 10, "Urea": 25, "SSP": 125, "MOP": 20, "FYM": 4},
        "Gram": {"N": 10, "P": 20, "K": 10, "Urea": 25, "SSP": 125, "MOP": 20, "FYM": 4},
        "Green Gram": {"N": 8, "P": 16, "K": 8, "Urea": 20, "SSP": 100, "MOP": 15, "FYM": 3},
        "Groundnut": {"N": 10, "P": 20, "K": 15, "Urea": 25, "SSP": 125, "MOP": 25, "FYM": 4},
        "Turmeric": {"N": 60, "P": 25, "K": 50, "Urea": 130, "SSP": 150, "MOP": 80, "FYM": 8},
        "Grapes": {"N": 40, "P": 30, "K": 50, "Urea": 90, "SSP": 180, "MOP": 80, "FYM": 10},
        "Banana": {"N": 80, "P": 30, "K": 100, "Urea": 175, "SSP": 180, "MOP": 160, "FYM": 10},
        "Mango": {"N": 50, "P": 25, "K": 50, "Urea": 110, "SSP": 150, "MOP": 80, "FYM": 12},
        "Tomato": {"N": 50, "P": 25, "K": 25, "Urea": 110, "SSP": 150, "MOP": 40, "FYM": 6},
        "Chilli": {"N": 40, "P": 20, "K": 20, "Urea": 90, "SSP": 125, "MOP": 35, "FYM": 5},
        "Brinjal": {"N": 40, "P": 20, "K": 20, "Urea": 90, "SSP": 125, "MOP": 35, "FYM": 5},
        "Ginger": {"N": 30, "P": 20, "K": 25, "Urea": 65, "SSP": 125, "MOP": 40, "FYM": 22.5},
    }

    def parse_acreage(self, text: str) -> float | None:
        """Extract farm acreage from queries like '2.5 एकर कापूस साठी'."""
        match = re.search(r"(\d+(?:\.\d+)?)\n?\s*(?:एकर|एकरा|acre|acres)", text.lower())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None

    def calculate_fertilizer_dosage(self, query: str, default_acres: float = 1.0) -> FertilizerMathResult:
        """CoT dosage calculator: computes exact Urea / SSP / MOP math for given acreage."""
        crops = resolve_crops_smart(query)
        crop = crops[0] if crops else "Cotton"
        acres = self.parse_acreage(query) or default_acres

        std = self.CROP_NPK_STANDARD.get(crop, self.CROP_NPK_STANDARD["Cotton"])
        urea = round(std["Urea"] * acres, 1)
        ssp = round(std["SSP"] * acres, 1)
        mop = round(std["MOP"] * acres, 1)
        fym = round(std["FYM"] * acres, 1)

        expl_mr = (
            f"{acres} एकर {crop} पिकासाठी:\n"
            f"• युरिया (Urea): {urea} किलो ({std['Urea']} किलो/एकरप्रमाणे)\n"
            f"• सिंगल सुपर फॉस्फेट (SSP): {ssp} किलो ({std['SSP']} किलो/एकरप्रमाणे)\n"
            f"• म्युरिएट ऑफ पोटॅश (MOP): {mop} किलो ({std['MOP']} किलो/एकरप्रमाणे)\n"
            f"• शेणखत (FYM): {fym} टन"
        )
        expl_en = (
            f"For {acres} acre(s) of {crop}:\n"
            f"• Urea: {urea} kg (@ {std['Urea']} kg/acre)\n"
            f"• Single Super Phosphate (SSP): {ssp} kg (@ {std['SSP']} kg/acre)\n"
            f"• Muriate of Potash (MOP): {mop} kg (@ {std['MOP']} kg/acre)\n"
            f"• Organic FYM: {fym} tonnes"
        )

        return FertilizerMathResult(
            crop=crop,
            acres=acres,
            urea_kg=urea,
            ssp_kg=ssp,
            mop_kg=mop,
            fym_tonnes=fym,
            explanation_mr=expl_mr,
            explanation_en=expl_en,
        )

    DEFAULT_DRIP: dict[str, dict[str, float]] = {
        "Cotton": {"lpd": 5.0, "lph": 4.0, "drippers": 1},
        "Soybean": {"lpd": 4.0, "lph": 4.0, "drippers": 1},
        "Sugarcane": {"lpd": 8.0, "lph": 4.0, "drippers": 1},
        "Pomegranate": {"lpd": 20.0, "lph": 8.0, "drippers": 2},
        "Tur": {"lpd": 4.0, "lph": 4.0, "drippers": 1},
        "Onion": {"lpd": 3.5, "lph": 4.0, "drippers": 1},
        "Grapes": {"lpd": 16.0, "lph": 8.0, "drippers": 2},
        "Banana": {"lpd": 15.0, "lph": 8.0, "drippers": 1},
        "Turmeric": {"lpd": 5.0, "lph": 4.0, "drippers": 1},
        "Ginger": {"lpd": 5.0, "lph": 4.0, "drippers": 1},
        "Mango": {"lpd": 25.0, "lph": 8.0, "drippers": 2},
        "Tomato": {"lpd": 4.0, "lph": 4.0, "drippers": 1},
        "Chilli": {"lpd": 4.0, "lph": 4.0, "drippers": 1},
        "Brinjal": {"lpd": 4.0, "lph": 4.0, "drippers": 1},
        "Wheat": {"lpd": 4.0, "lph": 4.0, "drippers": 1},
        "Rice": {"lpd": 5.0, "lph": 4.0, "drippers": 1},
        "Maize": {"lpd": 4.0, "lph": 4.0, "drippers": 1},
        "Gram": {"lpd": 3.5, "lph": 4.0, "drippers": 1},
        "Green Gram": {"lpd": 3.5, "lph": 4.0, "drippers": 1},
        "Groundnut": {"lpd": 4.0, "lph": 4.0, "drippers": 1},
        "Sorghum": {"lpd": 4.0, "lph": 4.0, "drippers": 1},
        "Bajra": {"lpd": 3.5, "lph": 4.0, "drippers": 1},
        "Orange": {"lpd": 20.0, "lph": 8.0, "drippers": 2},
        "Sunflower": {"lpd": 4.0, "lph": 4.0, "drippers": 1},
        "Sesame": {"lpd": 3.5, "lph": 4.0, "drippers": 1},
    }

    def calculate_irrigation_schedule(
        self, crop: str, temp_c: float = 28.0, rainfall_mm: float = 0.0
    ) -> IrrigationMathResult:
        """CoT drip physics & ET water calculator."""
        crops = resolve_crops_smart(crop)
        crop_key = crops[0] if crops else "Cotton"

        if rainfall_mm >= 15.0:
            return IrrigationMathResult(
                crop=crop_key,
                drip_hours_per_day=0.0,
                liters_per_plant_day=0.0,
                status="OFF_RAINFALL",
                recommendation_mr=f"पाऊस {rainfall_mm:.1f} मिमी असल्यामुळे पुढील २ दिवस ठिबक बंद ठेवावे.",
                recommendation_en=f"Rainfall is {rainfall_mm:.1f}mm. Keep drip irrigation OFF for next 2 days.",
            )

        cfg = self.DEFAULT_DRIP.get(crop_key, {"lpd": 5.0, "lph": 4.0, "drippers": 1})
        lpd = cfg["lpd"]
        if temp_c >= 35.0:
            lpd = round(lpd * 1.2, 1)

        hours = lpd / (cfg["lph"] * cfg["drippers"])
        hours = max(1.5, round(hours, 1))

        return IrrigationMathResult(
            crop=crop_key,
            drip_hours_per_day=hours,
            liters_per_plant_day=lpd,
            status="ACTIVE_DRIP",
            recommendation_mr=f"दर आड दिवशी {hours:.1f} तास (प्रति झाड {lpd:.1f} लिटर) ठिबक सिंचन द्यावे.",
            recommendation_en=f"Run drip for {hours:.1f} hours alternate day ({lpd:.1f} L/plant/day).",
        )

    def evaluate_disease_risk(
        self, crop: str, temp_c: float = 28.0, humidity_pct: float = 81.0, rainfall_mm: float = 12.0
    ) -> DiseaseRiskResult:
        """CoT early outbreak predictor for fungal blights & sucking pests."""
        threats = {
            "Grapes": ("भुरी रोग (Powdery Mildew)", 85),
            "Cotton": ("गुलाबी बोंड अळी व ठिपक्यांची अळी", 80),
            "Tomato": ("लवकर करपा रोग (Early Blight)", 85),
            "Soybean": ("तांबेरा व खोड माशी", 75),
            "Banana": ("सिगाटोका रोग (Sigatoka)", 80),
        }
        threat, prob = threats.get(crop, ("बुरशीजन्य करपा रोग", 70))

        if humidity_pct > 80.0 or rainfall_mm > 5.0:
            risk = "HIGH"
            adv_mr = f"जास्त आर्द्रता ({humidity_pct:.0f}%) व पावसामुळे {threat} वाढीचा उच्च धोका (संभाव्यता {prob}%). पाऊस थांबल्यावर २ दिवसांनी फवारणी करा."
            adv_en = f"High humidity ({humidity_pct:.0f}%) & rain indicates High Risk ({prob}%) for {threat}. Spray 2 days after rain stops."
        else:
            risk = "MEDIUM"
            adv_mr = f"हवामान सामान्य असून {threat} चा मध्यम धोका (संभाव्यता ५०%). नियमित पाहणी ठेवावी."
            adv_en = f"Weather conditions normal; Medium Risk (50%) for {threat}. Monitor field regularly."

        return DiseaseRiskResult(
            risk_level=risk,
            primary_threat=threat,
            probability_pct=prob,
            advisory_mr=adv_mr,
            advisory_en=adv_en,
        )


agri_reasoning_engine = AgronomicReasoningEngine()
