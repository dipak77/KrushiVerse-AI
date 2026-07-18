from datetime import datetime

class FarmMemory:
    """In-Memory & Storage system for personalized farm profiles and operational history."""

    def __init__(self):
        self.farms = {}
        self._initialize_default_demo_farm()

    def _initialize_default_demo_farm(self):
        """Seed a default sample farm memory profile for Pimpri / Pune region."""
        demo_id = "FARM_101"
        self.farms[demo_id] = {
            "farm_id": demo_id,
            "farmer_name": "Ramesh Patil",
            "location": {"district": "Pune", "taluka": "Haveli", "village": "Pimpri-Chinchwad"},
            "land_area_acres": 4.5,
            "soil_profile": {
                "soil_type": "Medium Black Soil",
                "pH": 7.2,
                "EC_dS_m": 0.45,
                "organic_carbon_pct": 0.52,
                "nitrogen_kg_ha": 180,  # Low
                "phosphorus_kg_ha": 22, # Medium
                "potassium_kg_ha": 280, # High
                "last_tested_date": "2025-11-15"
            },
            "current_crop": {
                "crop_name": "Pomegranate",
                "crop_name_mr": "डाळिंब",
                "variety": "Bhagwa",
                "sowing_or_bahar_date": "2026-05-10",
                "growth_stage": "Fruit Development",
                "water_source": "Drip Irrigation + Borewell"
            },
            "crop_history": [
                {"year": "2024-2025", "crop": "Soybean (Kharif)", "yield_quintals": 12},
                {"year": "2024-2025", "crop": "Onion (Rabi)", "yield_quintals": 85}
            ],
            "action_logs": [
                {
                    "date": "2026-06-20",
                    "type": "Fertigation",
                    "details": "Applied 0:52:34 @ 3kg/acre via drip."
                },
                {
                    "date": "2026-07-02",
                    "type": "Pest Control",
                    "details": "Sprayed Neem oil (5000 ppm) for Thrips control."
                }
            ]
        }

    def get_farm(self, farm_id: str) -> dict | None:
        return self.farms.get(farm_id)

    def create_or_update_farm(self, farm_id: str, farm_data: dict) -> dict:
        if farm_id not in self.farms:
            self.farms[farm_id] = {
                "farm_id": farm_id,
                "farmer_name": farm_data.get("farmer_name", "Unknown Farmer"),
                "location": farm_data.get("location", {"district": "Maharashtra"}),
                "land_area_acres": farm_data.get("land_area_acres", 1.0),
                "soil_profile": farm_data.get("soil_profile", {}),
                "current_crop": farm_data.get("current_crop", {}),
                "crop_history": farm_data.get("crop_history", []),
                "action_logs": []
            }
        else:
            self.farms[farm_id].update(farm_data)

        return self.farms[farm_id]

    def log_action(self, farm_id: str, action_type: str, details: str):
        if farm_id in self.farms:
            log_entry = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "type": action_type,
                "details": details
            }
            self.farms[farm_id]["action_logs"].append(log_entry)
            return log_entry
        return None

farm_memory_store = FarmMemory()
