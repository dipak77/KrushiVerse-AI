class SmartIrrigationModel:
    """Evapotranspiration and drip irrigation runtime calculator."""

    CROP_COEFFICIENTS = {
        "cotton": 0.85,
        "soybean": 0.90,
        "sugarcane": 1.15,
        "pomegranate": 0.75,
        "onion": 0.80
    }

    def calculate_water_requirement(self, crop: str, acreage: float, temperature_c: float, humidity_pct: float, drip_discharge_lph_per_plant: float = 4.0, plants_per_acre: int = 400) -> dict:
        crop_key = crop.lower()
        kc = self.CROP_COEFFICIENTS.get(crop_key, 0.80)

        # Approximate reference evapotranspiration (ETo in mm/day) based on Hargreaves formula
        eto = (0.0023 * (temperature_c + 17.8) * (35.0 - temperature_c) ** 0.5 * 10) / (humidity_pct / 50.0)
        eto = max(3.0, min(8.5, eto))

        # Crop evapotranspiration ETc (mm/day)
        etc_mm_day = eto * kc

        # Convert to Liters per Acre per Day (1 mm rain over 1 acre = 4046.86 Liters)
        water_liters_per_acre = etc_mm_day * 4046.86
        total_water_liters = water_liters_per_acre * acreage

        # Drip irrigation hours computation
        total_drip_discharge_lph = plants_per_acre * drip_discharge_lph_per_plant
        recommended_drip_hours = round(water_liters_per_acre / total_drip_discharge_lph, 2)

        return {
            "crop": crop.capitalize(),
            "acreage": acreage,
            "ref_evapotranspiration_mm_day": round(eto, 2),
            "crop_evapotranspiration_mm_day": round(etc_mm_day, 2),
            "water_required_liters_per_acre_day": round(water_liters_per_acre, 1),
            "total_farm_water_required_liters_day": round(total_water_liters, 1),
            "drip_irrigation_schedule": {
                "drip_runtime_hours_per_day": recommended_drip_hours,
                "recommended_splits": "Run 2 slots: Early morning (6:00 AM - 8:30 AM) and Evening (5:00 PM - 6:30 PM)."
            }
        }

irrigation_model = SmartIrrigationModel()
