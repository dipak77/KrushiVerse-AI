import os
import json
from app.config import settings

class KnowledgeBaseLoader:
    def __init__(self, data_dir: str = settings.DATA_DIR):
        self.data_dir = data_dir
        self.crops_and_diseases = self._load_json("crops_and_diseases.json")
        self.soil_and_fertilizers = self._load_json("soil_and_fertilizers.json")
        self.government_schemes = self._load_json("government_schemes.json")
        self.market_prices = self._load_json("market_prices.json")
        self.graph_data = self._load_json("knowledge_graph.json")

    def _load_json(self, filename: str) -> dict:
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_documents(self) -> list[dict]:
        """Flatten all knowledge entries into document objects for vector and hybrid search indexing."""
        docs = []

        # Crop & Disease documents
        for crop in self.crops_and_diseases.get("crops", []):
            docs.append({
                "id": f"crop_{crop['crop_id']}",
                "category": "Crop",
                "title": f"Crop Guide: {crop['name_en']} ({crop['name_mr']})",
                "content": f"Crop: {crop['name_en']} ({crop['name_mr']}). Season: {crop['season']}. "
                           f"Ideal soil: {', '.join(crop['ideal_soil'])}. "
                           f"Ideal temp: {crop['ideal_temp_c']['min']}-{crop['ideal_temp_c']['max']}°C. "
                           f"Ideal rainfall: {crop['ideal_rainfall_mm']['min']}-{crop['ideal_rainfall_mm']['max']} mm. "
                           f"Stages: {', '.join(crop['growth_stages'])}. "
                           f"Major pests: {', '.join(crop['major_pests'])}. "
                           f"Major diseases: {', '.join(crop['major_diseases'])}.",
                "metadata": crop
            })

        for dis in self.crops_and_diseases.get("diseases_and_pests", []):
            docs.append({
                "id": f"disease_{dis['id']}",
                "category": "Disease",
                "title": f"Disease & Pest: {dis['name_en']} ({dis['name_mr']}) in {dis['crop_en']}",
                "content": f"Disease/Pest: {dis['name_en']} ({dis['name_mr']}) affecting {dis['crop_en']}. "
                           f"Symptoms (English): {dis['symptoms_en']}. "
                           f"Symptoms (Marathi): {dis['symptoms_mr']}. "
                           f"Organic Control: {dis['organic_control_en']} / {dis['organic_control_mr']}. "
                           f"Chemical Control: {dis['chemical_control_en']} / {dis['chemical_control_mr']}.",
                "metadata": dis
            })

        # Soil & Fertilizer documents
        for fert in self.soil_and_fertilizers.get("fertilizer_recommendations", []):
            docs.append({
                "id": f"fert_{fert['crop_en'].lower()}",
                "category": "Fertilizer",
                "title": f"Fertilizer Guide: {fert['crop_en']} ({fert['crop_mr']})",
                "content": f"Fertilizer guide for {fert['crop_en']} ({fert['crop_mr']}). "
                           f"Recommended NPK per acre: N={fert['recommended_npk_kg_per_acre']['N']}kg, "
                           f"P={fert['recommended_npk_kg_per_acre']['P']}kg, K={fert['recommended_npk_kg_per_acre']['K']}kg. "
                           f"Basal Dose: {fert['basal_dose']} ({fert['basal_dose_mr']}). "
                           f"Top Dressing: {fert['top_dressing']} ({fert['top_dressing_mr']}). "
                           f"Micronutrients: {fert['micronutrients']}.",
                "metadata": fert
            })

        # Government Scheme documents
        for scheme in self.government_schemes.get("schemes", []):
            docs.append({
                "id": f"scheme_{scheme['scheme_id']}",
                "category": "Government Scheme",
                "title": f"Government Scheme: {scheme['name_en']} ({scheme['name_mr']})",
                "content": f"Government Scheme: {scheme['name_en']} ({scheme['name_mr']}). "
                           f"Benefits: {scheme['benefits_en']} ({scheme['benefits_mr']}). "
                           f"Eligibility: {scheme['eligibility_en']} ({scheme['eligibility_mr']}). "
                           f"Documents required: {', '.join(scheme['documents_required'])}.",
                "metadata": scheme
            })

        return docs

kb_loader = KnowledgeBaseLoader()
