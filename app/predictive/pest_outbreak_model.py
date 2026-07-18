class PestOutbreakRiskModel:
    """Predictive epidemiological model calculating pest/disease outbreak risk probabilities."""

    def calculate_outbreak_risk(self, crop: str, temperature_c: float, humidity_pct: float, rainfall_mm: float) -> dict:
        crop_lower = crop.lower()
        risks = []

        # Pomegranate Bacterial Blight Risk
        if "pomegranate" in crop_lower:
            risk_score = 0
            if 25 <= temperature_c <= 35:
                risk_score += 35
            if humidity_pct > 75:
                risk_score += 45
            if rainfall_mm > 5.0:
                risk_score += 20

            level = "CRITICAL / HIGH RISK" if risk_score >= 70 else ("MODERATE" if risk_score >= 40 else "LOW")
            risks.append({
                "disease_pest": "Bacterial Blight (Telyat) / तेल्या रोग",
                "risk_percentage": min(98, risk_score),
                "risk_level": level,
                "advisory": "Perform preventative spray of Copper Oxychloride (2g/L) + Streptocycline immediately if humidity stays >80%."
            })

        # Cotton Pink Bollworm Risk
        if "cotton" in crop_lower:
            risk_score = 0
            if 22 <= temperature_c <= 32:
                risk_score += 30
            if humidity_pct > 70:
                risk_score += 40

            level = "HIGH RISK" if risk_score >= 60 else ("MODERATE" if risk_score >= 35 else "LOW")
            risks.append({
                "disease_pest": "Pink Bollworm / गुलाबी बोंड अळी",
                "risk_percentage": min(95, risk_score),
                "risk_level": level,
                "advisory": "Install Pheromone Traps (5/acre) to monitor moth counts."
            })

        # Default general fungal risk
        general_fungal_risk = min(95, int((humidity_pct / 100.0) * 60 + (rainfall_mm / 30.0) * 35))
        risks.append({
            "disease_pest": "General Fungal / Foliar Blight Risk",
            "risk_percentage": general_fungal_risk,
            "risk_level": "HIGH" if general_fungal_risk > 65 else ("MODERATE" if general_fungal_risk > 35 else "LOW"),
            "advisory": "Ensure clear field drainage and avoid overhead foliage wetting."
        })

        return {
            "crop": crop.capitalize(),
            "environmental_conditions": {
                "temperature_c": temperature_c,
                "humidity_pct": humidity_pct,
                "rainfall_mm": rainfall_mm
            },
            "assessed_risks": risks
        }

pest_outbreak_model = PestOutbreakRiskModel()
