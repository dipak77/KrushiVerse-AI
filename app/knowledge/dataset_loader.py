import os
import json
from app.config import settings


class KnowledgeBaseLoader:
    """Loads and flattens multi-source open agricultural knowledge for RAG indexing."""

    def __init__(self, data_dir: str = settings.DATA_DIR):
        self.data_dir = data_dir
        self.crops_and_diseases = self._load_json("crops_and_diseases.json")
        self.soil_and_fertilizers = self._load_json("soil_and_fertilizers.json")
        self.government_schemes = self._load_json("government_schemes.json")
        self.market_prices = self._load_json("market_prices.json")
        self.graph_data = self._load_json("knowledge_graph.json")
        self.agri_advisories = self._load_json("agri_advisories.json")
        self.seed_varieties = self._load_json("seed_varieties.json")
        self.irrigation_practices = self._load_json("irrigation_practices.json")
        self.climate_zones = self._load_json("climate_zones.json")
        self.open_source_catalog = self._load_json("open_source_catalog.json")

    def _load_json(self, filename: str) -> dict:
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def knowledge_stats(self) -> dict:
        docs = self.get_all_documents()
        by_cat: dict[str, int] = {}
        for d in docs:
            by_cat[d["category"]] = by_cat.get(d["category"], 0) + 1
        return {
            "total_documents": len(docs),
            "by_category": by_cat,
            "crops": len(self.crops_and_diseases.get("crops", [])),
            "diseases_pests": len(self.crops_and_diseases.get("diseases_and_pests", [])),
            "schemes": len(self.government_schemes.get("schemes", [])),
            "markets": len(self.market_prices.get("markets", [])),
            "advisories": len(self.agri_advisories.get("advisories", [])),
            "graph_nodes": len(self.graph_data.get("nodes", [])),
            "graph_edges": len(self.graph_data.get("edges", [])),
            "open_sources_catalogued": len(self.open_source_catalog.get("sources", [])),
        }

    def get_all_documents(self) -> list[dict]:
        """Flatten all knowledge entries into document objects for hybrid/vector search."""
        docs: list[dict] = []

        for crop in self.crops_and_diseases.get("crops", []):
            sci = crop.get("scientific_name", "")
            docs.append({
                "id": f"crop_{crop['crop_id']}",
                "category": "Crop",
                "title": f"Crop Guide: {crop['name_en']} ({crop['name_mr']})",
                "content": (
                    f"Crop: {crop['name_en']} ({crop['name_mr']}). Scientific: {sci}. "
                    f"Season: {crop['season']}. Ideal soil: {', '.join(crop.get('ideal_soil', []))}. "
                    f"Ideal temp: {crop.get('ideal_temp_c', {}).get('min')}-{crop.get('ideal_temp_c', {}).get('max')}°C. "
                    f"Ideal rainfall: {crop.get('ideal_rainfall_mm', {}).get('min')}-{crop.get('ideal_rainfall_mm', {}).get('max')} mm. "
                    f"Duration: {crop.get('duration_days')} days. "
                    f"Stages: {', '.join(crop.get('growth_stages', []))}. "
                    f"Major pests: {', '.join(crop.get('major_pests', []))}. "
                    f"Major diseases: {', '.join(crop.get('major_diseases', []))}. "
                    f"Source: {crop.get('source', 'ICAR/SAU open advisory')}."
                ),
                "metadata": crop,
                "source": crop.get("source", "local_kb"),
            })

        for dis in self.crops_and_diseases.get("diseases_and_pests", []):
            docs.append({
                "id": f"disease_{dis['id']}",
                "category": "Disease",
                "title": f"Disease & Pest: {dis['name_en']} ({dis.get('name_mr', '')}) in {dis['crop_en']}",
                "content": (
                    f"Disease/Pest: {dis['name_en']} ({dis.get('name_mr', '')}) affecting {dis['crop_en']}. "
                    f"Symptoms (English): {dis.get('symptoms_en', '')}. "
                    f"Symptoms (Marathi): {dis.get('symptoms_mr', '')}. "
                    f"Organic Control: {dis.get('organic_control_en', '')} / {dis.get('organic_control_mr', '')}. "
                    f"Chemical Control: {dis.get('chemical_control_en', '')} / {dis.get('chemical_control_mr', '')}. "
                    f"Source: {dis.get('source', 'ICAR IPM')}."
                ),
                "metadata": dis,
                "source": dis.get("source", "local_kb"),
            })

        for soil in self.soil_and_fertilizers.get("soil_types", []):
            docs.append({
                "id": f"soil_{soil['type'].lower().replace(' ', '_')[:40]}",
                "category": "Soil",
                "title": f"Soil Type: {soil['type']} ({soil.get('type_mr', '')})",
                "content": (
                    f"Soil: {soil['type']}. {soil.get('characteristics_en', '')} "
                    f"({soil.get('characteristics_mr', '')}). "
                    f"Suitable crops: {', '.join(soil.get('suitable_crops', []))}."
                ),
                "metadata": soil,
                "source": "soil_kb",
            })

        for fert in self.soil_and_fertilizers.get("fertilizer_recommendations", []):
            npk = fert.get("recommended_npk_kg_per_acre", {})
            docs.append({
                "id": f"fert_{fert['crop_en'].lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                "category": "Fertilizer",
                "title": f"Fertilizer Guide: {fert['crop_en']} ({fert.get('crop_mr', '')})",
                "content": (
                    f"Fertilizer guide for {fert['crop_en']} ({fert.get('crop_mr', '')}). "
                    f"Recommended NPK per acre: N={npk.get('N')}kg, P={npk.get('P')}kg, K={npk.get('K')}kg. "
                    f"Basal Dose: {fert.get('basal_dose', '')} ({fert.get('basal_dose_mr', '')}). "
                    f"Top Dressing: {fert.get('top_dressing', '')} ({fert.get('top_dressing_mr', '')}). "
                    f"Micronutrients: {fert.get('micronutrients', '')}. "
                    f"Source: {fert.get('source', 'SAU PoP')}."
                ),
                "metadata": fert,
                "source": fert.get("source", "local_kb"),
            })

        for scheme in self.government_schemes.get("schemes", []):
            docs.append({
                "id": f"scheme_{scheme['scheme_id']}",
                "category": "Government Scheme",
                "title": f"Government Scheme: {scheme['name_en']} ({scheme.get('name_mr', '')})",
                "content": (
                    f"Government Scheme: {scheme['name_en']} ({scheme.get('name_mr', '')}). "
                    f"Benefits: {scheme.get('benefits_en', '')} ({scheme.get('benefits_mr', '')}). "
                    f"Eligibility: {scheme.get('eligibility_en', '')} ({scheme.get('eligibility_mr', '')}). "
                    f"Documents required: {', '.join(scheme.get('documents_required', []))}. "
                    f"Portal: {scheme.get('portal', 'n/a')}. "
                    f"Source: {scheme.get('source', 'gov open info')}."
                ),
                "metadata": scheme,
                "source": scheme.get("source", "gov_kb"),
            })

        for i, m in enumerate(self.market_prices.get("markets", [])):
            docs.append({
                "id": f"market_{i}_{m.get('mandi', '').replace(' ', '_')}",
                "category": "Market",
                "title": f"Mandi Price: {m.get('crop')} @ {m.get('mandi')}",
                "content": (
                    f"Market price for {m.get('crop')} ({m.get('crop_mr', '')}) variety {m.get('variety')} "
                    f"at {m.get('mandi')}, {m.get('district')}, {m.get('state')}. "
                    f"Min ₹{m.get('min_price_rs_quintal')}, Modal ₹{m.get('modal_price_rs_quintal')}, "
                    f"Max ₹{m.get('max_price_rs_quintal')} per quintal. "
                    f"Arrivals {m.get('arrival_tonnes')} tonnes. Trend: {m.get('trend')}. "
                    f"Date: {m.get('date')}. Source: {m.get('source', 'Agmarknet-style')}."
                ),
                "metadata": m,
                "source": m.get("source", "market_kb"),
            })

        for adv in self.agri_advisories.get("advisories", []):
            docs.append({
                "id": f"adv_{adv['id']}",
                "category": "Advisory",
                "title": adv.get("title_en", "Advisory"),
                "content": (
                    f"{adv.get('title_en', '')} / {adv.get('title_mr', '')}. "
                    f"{adv.get('content_en', '')} {adv.get('content_mr', '')}. "
                    f"Category: {adv.get('category', 'Advisory')}. "
                    f"Source: {adv.get('source', 'open advisory')}."
                ),
                "metadata": adv,
                "source": adv.get("source", "advisory_kb"),
            })

        for i, seed in enumerate(self.seed_varieties.get("varieties", [])):
            docs.append({
                "id": f"seed_{i}_{seed.get('crop_en', '').replace(' ', '_')}",
                "category": "Seed Variety",
                "title": f"Variety: {seed.get('variety')} ({seed.get('crop_en')})",
                "content": (
                    f"Seed variety {seed.get('variety')} for {seed.get('crop_en')}. "
                    f"Notes: {seed.get('notes_mr', '')}. "
                    f"Agro-climatic fit: {seed.get('agro_climatic_fit', '')}. "
                    f"Source: {seed.get('source', 'variety notes')}."
                ),
                "metadata": seed,
                "source": seed.get("source", "seed_kb"),
            })

        for prac in self.irrigation_practices.get("practices", []):
            docs.append({
                "id": f"irr_{prac['id']}",
                "category": "Irrigation",
                "title": prac.get("title", "Irrigation practice"),
                "content": (
                    f"{prac.get('title', '')}. {prac.get('content', '')} "
                    f"Crops: {', '.join(prac.get('crops', []))}. "
                    f"Source: {prac.get('source', 'irrigation open knowledge')}."
                ),
                "metadata": prac,
                "source": prac.get("source", "irrigation_kb"),
            })

        for i, zone in enumerate(self.climate_zones.get("zones", [])):
            docs.append({
                "id": f"zone_{i}",
                "category": "Climate Zone",
                "title": f"Agro-climate: {zone.get('zone')}",
                "content": (
                    f"Climate zone {zone.get('zone')}. Districts: {', '.join(zone.get('districts', []))}. "
                    f"Rainfall: {zone.get('rainfall_mm')} mm. "
                    f"Recommended crops: {', '.join(zone.get('recommended_crops', []))}. "
                    f"Notes: {zone.get('notes', '')}."
                ),
                "metadata": zone,
                "source": "climate_zone_kb",
            })

        for i, src in enumerate(self.open_source_catalog.get("sources", [])):
            docs.append({
                "id": f"catalog_{i}",
                "category": "Open Source Catalog",
                "title": f"Open Source: {src.get('name')}",
                "content": (
                    f"Open data source {src.get('name')} ({src.get('type')}). "
                    f"Use for: {src.get('use')}. URL: {src.get('url')}."
                ),
                "metadata": src,
                "source": "open_catalog",
            })

        return docs


kb_loader = KnowledgeBaseLoader()
