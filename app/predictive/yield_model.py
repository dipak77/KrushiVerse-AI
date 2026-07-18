class CropYieldPredictionModel:
    """Predictive yield estimation model combining soil, climate, and crop parameters."""

    BASE_YIELDS_PER_ACRE = {
        "cotton": {"base_quintals": 12.0, "unit": "Quintals"},
        "soybean": {"base_quintals": 10.0, "unit": "Quintals"},
        "sugarcane": {"base_quintals": 450.0, "unit": "Quintals (45 Tonnes)"},
        "pomegranate": {"base_quintals": 60.0, "unit": "Quintals (6 Tonnes)"},
        "onion": {"base_quintals": 120.0, "unit": "Quintals (12 Tonnes)"}
    }

    def predict_yield(self, crop: str, acreage: float, N_status: str = "Medium", P_status: str = "Medium", K_status: str = "Medium", irrigation_quality: str = "Good") -> dict:
        crop_key = crop.lower()
        info = self.BASE_YIELDS_PER_ACRE.get(crop_key, {"base_quintals": 15.0, "unit": "Quintals"})

        base_val = info["base_quintals"]

        # Calculate multi-factor yield multiplier
        multiplier = 1.0

        # Nutrient impact
        if N_status == "Deficient":
            multiplier -= 0.12
        elif N_status == "High":
            multiplier += 0.05

        if P_status == "Deficient":
            multiplier -= 0.10

        if K_status == "High":
            multiplier += 0.08

        # Irrigation impact
        if irrigation_quality == "Excellent Drip":
            multiplier += 0.15
        elif irrigation_quality == "Rainfed":
            multiplier -= 0.25

        yield_per_acre = round(base_val * multiplier, 2)
        total_predicted_yield = round(yield_per_acre * acreage, 2)

        return {
            "crop": crop.capitalize(),
            "acreage": acreage,
            "predicted_yield_per_acre": yield_per_acre,
            "total_predicted_yield": total_predicted_yield,
            "unit": info["unit"],
            "yield_confidence_score": 0.88,
            "key_limiting_factors": [
                f"Nitrogen deficiency impact: {'Negative' if N_status=='Deficient' else 'Optimal'}",
                f"Water regime: {irrigation_quality}"
            ]
        }

yield_model = CropYieldPredictionModel()
