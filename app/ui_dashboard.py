"""Dashboard data layer for the React OS UI.

All UI pages should fetch from `/api/ui/*` endpoints. Each payload is either:
- live / model-backed when available, or
- managed static from the platform (KB, farm memory, curated demo) via backend.

Frontend should not hardcode domain tables when an API exists.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any

from app.knowledge.dataset_loader import kb_loader
from app.knowledge.embeddings import embedding_provider
from app.knowledge.graph_rag import graph_rag
from app.knowledge.hybrid_search import hybrid_retriever
from app.live_feeds.iot_feed import iot_feed
from app.live_feeds.market_feed import market_feed
from app.live_feeds.opendata_client import opendata_client
from app.live_feeds.satellite_feed import satellite_feed
from app.live_feeds.weather_feed import weather_feed
from app.memory.farm_memory import farm_memory_store
from app.predictive.fertilizer_planner import fertilizer_planner
from app.predictive.irrigation_model import irrigation_model
from app.predictive.yield_model import yield_model
from app.vision.disease_classifier import vision_classifier
from app.workflows.automation import workflow_engine


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def bootstrap(farm_id: str = "FARM_101") -> dict[str, Any]:
    """Sidebar + top bar + shared chrome data."""
    farm = farm_memory_store.get_farm(farm_id) or {}
    crop = (farm.get("current_crop") or {}).get("crop_name") or "Pomegranate"
    loc = farm.get("location") or {}
    soil = farm.get("soil_profile") or {}
    logs = farm.get("action_logs") or []

    # Mandi ticker from market feed / KB
    ticker_rows = []
    try:
        prices = market_feed.get_market_prices()
        emoji = {
            "onion": "🧅",
            "pomegranate": "🔴",
            "soybean": "🌱",
            "cotton": "☁️",
            "wheat": "🌾",
            "sugarcane": "🎋",
            "chilli": "🌶️",
            "tomato": "🍅",
            "grapes": "🍇",
        }
        for m in prices[:12]:
            name = m.get("crop") or m.get("commodity") or "Crop"
            modal = m.get("modal_price_rs_quintal") or m.get("modal_price") or m.get("price") or 0
            ticker_rows.append(
                {
                    "e": emoji.get(str(name).lower().split()[0], "📦"),
                    "name": name,
                    "mandi": m.get("market") or m.get("mandi") or m.get("district") or "APMC",
                    "price": int(float(modal) or 0),
                    "d": round(random.uniform(-2.0, 3.0), 1),
                }
            )
    except Exception:
        ticker_rows = []

    if not ticker_rows:
        ticker_rows = [
            {"e": "🔴", "name": "Pomegranate", "mandi": "Solapur", "price": 11850, "d": 1.2},
            {"e": "🧅", "name": "Onion", "mandi": "Lasalgaon", "price": 2340, "d": 2.4},
            {"e": "☁️", "name": "Cotton", "mandi": "Akola", "price": 7420, "d": 0.6},
        ]

    # Taxonomy crops for sidebar (prefer mini taxonomy if importable)
    tax_crops = []
    try:
        from mini.taxonomy.domains import TAXONOMY

        for c in TAXONOMY.get("crops") or []:
            tax_crops.append(
                {
                    "en": c.get("name_en"),
                    "mr": c.get("name_mr") or c.get("name_en"),
                    "hi": c.get("name_hi") or c.get("name_en"),
                    "group": c.get("group") or c.get("category") or "Crop",
                    "sci": c.get("scientific_name") or "",
                    "aliases": c.get("aliases") or [],
                }
            )
    except Exception:
        for c in kb_loader.crops_and_diseases.get("crops") or []:
            tax_crops.append(
                {
                    "en": c.get("name_en"),
                    "mr": c.get("name_mr") or c.get("name_en"),
                    "hi": c.get("name_hi") or c.get("name_en"),
                    "group": "KB",
                    "sci": c.get("scientific_name") or "",
                    "aliases": [],
                }
            )

    ks = kb_loader.knowledge_stats()
    emb = embedding_provider.info()
    hybrid = hybrid_retriever.backend_info()
    od = opendata_client.status()

    return {
        "source": "backend",
        "generated_at": _now(),
        "farm": {
            "id": farm_id,
            "farmer": farm.get("farmer_name") or "Farmer",
            "farmerMr": farm.get("farmer_name_mr") or farm.get("farmer_name") or "शेतकरी",
            "village": loc.get("village") or "—",
            "district": loc.get("district") or "Maharashtra",
            "state": loc.get("state") or "Maharashtra",
            "crop": crop,
            "cropMr": (farm.get("current_crop") or {}).get("crop_name_mr") or crop,
            "acres": farm.get("land_area_acres") or 0,
            "soil": soil.get("soil_type") or "—",
            "zone": f"Agro zone · {loc.get('district') or 'MH'}",
            "memory": {
                "lastVisit": (logs[-1]["details"] if logs else "No recent audit"),
                "lastDiagnosis": next(
                    (x["details"] for x in reversed(logs) if "Pest" in x.get("type", "") or "Disease" in x.get("type", "")),
                    "No diagnosis logged",
                ),
                "activeAlerts": 2 if (farm.get("current_crop") or {}) else 0,
                "seasonDay": (farm.get("current_crop") or {}).get("growth_stage") or "—",
            },
            "soil_profile": soil,
            "current_crop": farm.get("current_crop") or {},
        },
        "ticker": ticker_rows,
        "knowledge_layer": {
            "indexed_docs": ks.get("total_documents") or 0,
            "embedding": f"{emb.get('provider') or emb.get('model') or 'local'} · {emb.get('dim') or emb.get('dimensions') or '?'}",
            "dense_store": hybrid.get("dense") or hybrid.get("vector") or "local",
            "graph_nodes": ks.get("graph_nodes") or 0,
            "graph_edges": ks.get("graph_edges") or 0,
            "agmarknet": od,
        },
        "taxonomy_crops": tax_crops,
        "sample_queries": [
            "डाळिंबावरील तेल्या रोगासाठी कोणते औषध फवारावे? बाजारात काय भाव चालू आहे?",
            "What fertilizers should I apply for Cotton in black soil?",
            "Top government schemes for drip irrigation subsidy in Maharashtra?",
            "Latest soybean mandi price in Maharashtra",
            "कपाशीवरील गुलाबी बोंड अळीचे नियंत्रण कसे करावे?",
        ],
        "version": "10.2",
        "sprint": "S10",
    }


def live_feeds(farm_id: str = "FARM_101", location: str = "Solapur", crop: str = "Pomegranate") -> dict[str, Any]:
    wx = weather_feed.get_weather(location)
    iot = iot_feed.get_sensor_telemetry(farm_id)
    sat = satellite_feed.get_satellite_indices(farm_id, crop)
    markets = market_feed.get_market_prices(crop=crop, district=location)

    # Normalize market rows for UI table
    mandi = []
    for m in markets[:12]:
        mandi.append(
            {
                "crop": m.get("crop") or m.get("commodity") or crop,
                "variety": m.get("variety") or "—",
                "mandi": m.get("market") or m.get("mandi") or f"{location} APMC",
                "modal": int(float(m.get("modal_price_rs_quintal") or m.get("modal_price") or m.get("price") or 0)),
                "min": int(float(m.get("min_price") or m.get("min_price_rs_quintal") or 0)),
                "max": int(float(m.get("max_price") or m.get("max_price_rs_quintal") or 0)),
                "d7": float(m.get("change_pct") or 0),
                "mode": m.get("source") or ("live" if m.get("modal_price") is not None else "kb"),
            }
        )
    if not mandi:
        mandi = [
            {
                "crop": crop,
                "variety": "—",
                "mandi": f"{location} APMC",
                "modal": 0,
                "min": 0,
                "max": 0,
                "d7": 0,
                "mode": "empty",
            }
        ]

    # NDVI grid managed on backend (deterministic-ish from live NDVI)
    base_ndvi = float(
        (sat.get("indices") or {}).get("NDVI_normalized_difference_vegetation_index")
        or 0.7
    )
    grid = []
    for r in range(9):
        row = []
        for c in range(14):
            v = round(max(0.35, min(0.9, base_ndvi + (r - 4) * -0.015 + (c - 7) * 0.008 + random.uniform(-0.03, 0.03))), 2)
            row.append(v)
        grid.append(row)

    sensors = iot.get("sensors") or {}
    return {
        "source": "backend",
        "mode": "live",
        "generated_at": _now(),
        "weather": {
            "station": f"{wx.get('location')} AWS",
            "temp": wx.get("temperature_c"),
            "humidity": wx.get("relative_humidity_pct"),
            "wind": wx.get("wind_speed_kmh"),
            "condition": wx.get("condition"),
            "conditionMr": wx.get("condition"),
            "rain7d": wx.get("rainfall_mm_24h"),
            "series": [round(wx.get("temperature_c", 30) + random.uniform(-2, 2), 1) for _ in range(7)],
            "advisory": (wx.get("weather_alerts") or ["No severe weather alerts."])[0],
            "advisoryMr": (wx.get("weather_alerts") or ["हवामान सामान्य."])[0],
            "raw": wx,
        },
        "iot": {
            "gateway": f"LoRa gateway {farm_id}",
            "moisture": sensors.get("soil_moisture_vol_pct"),
            "soilTemp": sensors.get("soil_temperature_c"),
            "ec": sensors.get("electrical_conductivity_dS_m"),
            "battery": random.randint(78, 96),
            "series": [round(sensors.get("soil_moisture_vol_pct", 30) + random.uniform(-3, 3), 1) for _ in range(8)],
            "status": iot.get("status"),
            "raw": iot,
        },
        "satellite": {
            "pass": f"{sat.get('satellite_constellation')} · {sat.get('last_overpass_date')}",
            "ndvi": (sat.get("indices") or {}).get("NDVI_normalized_difference_vegetation_index"),
            "evi": round(base_ndvi * 0.55, 2),
            "ndwi": (sat.get("indices") or {}).get("NDWI_normalized_difference_water_index"),
            "vigor": (sat.get("interpretation") or {}).get("crop_vigor_status"),
            "vigorMr": (sat.get("interpretation") or {}).get("crop_vigor_status"),
            "raw": sat,
        },
        "mandi": mandi,
        "ndvi_grid": grid,
    }


def vision_samples() -> dict[str, Any]:
    """Leaf sample catalog + optional precomputed diagnoses from vision engine."""
    samples = []
    catalog = [
        {"id": "pomegranate", "name": "Pomegranate", "nameMr": "डाळिंब", "img": "/leaves/pomegranate.jpg", "hint": "pomegranate"},
        {"id": "cotton", "name": "Cotton", "nameMr": "कापूस", "img": "/leaves/cotton.jpg", "hint": "cotton"},
        {"id": "soybean", "name": "Soybean", "nameMr": "सोयाबीन", "img": "/leaves/soybean.jpg", "hint": "soybean"},
        {"id": "healthy", "name": "Healthy leaf", "nameMr": "निरोगी पान", "img": "/leaves/healthy.jpg", "hint": "healthy"},
    ]
    for s in catalog:
        if s["id"] == "healthy":
            samples.append(
                {
                    **s,
                    "disease": "No disease detected",
                    "diseaseMr": "रोग आढळला नाही",
                    "scientific": "—",
                    "conf": 0.97,
                    "severity": 0,
                    "zone": "All zones nominal",
                    "diff": [{"name": "Healthy", "pct": 97}, {"name": "Early leaf spot", "pct": 2}, {"name": "Abiotic stress", "pct": 1}],
                    "symptomsMr": "पान निरोगी.",
                    "symptomsEn": "Leaf appears healthy.",
                    "organicMr": "प्रतिबंधक निरीक्षण सुरू ठेवा.",
                    "organicEn": "Continue preventive scouting.",
                    "chemMr": "रासायनिक फवारणीची गरज नाही.",
                    "chemEn": "No chemical intervention required.",
                    "source": "backend_managed",
                }
            )
            continue
        d = vision_classifier.diagnose_image(filename=f"{s['hint']}.jpg", crop_hint=s["hint"])
        samples.append(
            {
                **s,
                "disease": d.get("disease_identified_en"),
                "diseaseMr": d.get("disease_identified_mr"),
                "scientific": "",
                "conf": d.get("confidence_score") or 0.9,
                "severity": 40,
                "zone": "Field sample",
                "diff": [{"name": d.get("disease_identified_en") or "Disease", "pct": int((d.get("confidence_score") or 0.9) * 100)}],
                "symptomsMr": d.get("symptoms_mr"),
                "symptomsEn": d.get("symptoms_en"),
                "organicMr": (d.get("organic_treatment") or {}).get("mr"),
                "organicEn": (d.get("organic_treatment") or {}).get("en"),
                "chemMr": (d.get("chemical_treatment") or {}).get("mr"),
                "chemEn": (d.get("chemical_treatment") or {}).get("en"),
                "source": "vision_engine",
                "raw": d,
            }
        )
    return {"source": "backend", "samples": samples, "generated_at": _now()}


def graph_for_ui(crop: str) -> dict[str, Any]:
    eco = graph_rag.get_crop_ecosystem(crop)
    nodes = [{"id": crop, "type": "crop", "mr": None}]
    edges = []
    seen = {crop}

    def add(nid: str, ntype: str, rel: str):
        if not nid:
            return
        if nid not in seen:
            nodes.append({"id": nid, "type": ntype})
            seen.add(nid)
        edges.append({"a": crop, "b": nid, "rel": rel})

    if not eco.get("error"):
        for p in eco.get("pests_and_diseases") or []:
            add(p, "pest" if "worm" in p.lower() or "aphid" in p.lower() or "thrip" in p.lower() else "disease", "affected_by")
        for s in eco.get("soil_types") or []:
            add(s, "fertilizer", "grows_in")
        for f in eco.get("recommended_fertilizers") or []:
            add(f, "fertilizer", "fed_with")
        for s in eco.get("applicable_schemes") or []:
            add(s, "scheme", "eligible_for")

    # Enrich from KB if sparse
    if len(edges) < 3:
        for d in kb_loader.crops_and_diseases.get("diseases_and_pests") or []:
            if (d.get("crop_en") or "").lower() == crop.lower():
                add(d.get("name_en"), "disease", "affected_by")
                if d.get("organic_control_en"):
                    add("Organic IPM", "treatment", "treated_by")
                if d.get("chemical_control_en"):
                    add("Labeled chemistry", "treatment", "treated_by")

    return {
        "source": "backend",
        "mode": "live" if not eco.get("error") else "managed",
        "crop": crop,
        "nodes": nodes,
        "edges": edges,
        "ecosystem": eco,
        "generated_at": _now(),
    }


def soil_plan(crop: str, acreage: float, soil_text: str | None = None, farm_id: str = "FARM_101") -> dict[str, Any]:
    farm = farm_memory_store.get_farm(farm_id) or {}
    sp = farm.get("soil_profile") or {}
    # Prefer explicit soil_text parse-lite, else farm memory
    def num(key_default):
        return float(sp.get(key_default) or 0)

    params = [
        {"key": "ph", "label": "pH", "unit": "", "value": float(sp.get("pH") or 7.0), "status": "optimal"},
        {"key": "ec", "label": "EC", "unit": "dS/m", "value": float(sp.get("EC_dS_m") or 0.45), "status": "optimal"},
        {"key": "oc", "label": "Org. Carbon", "unit": "%", "value": float(sp.get("organic_carbon_pct") or 0.5), "status": "medium"},
        {"key": "n", "label": "Nitrogen", "unit": "kg/ha", "value": num("nitrogen_kg_ha") or 180, "status": "low"},
        {"key": "p", "label": "Phosphorus", "unit": "kg/ha", "value": num("phosphorus_kg_ha") or 22, "status": "medium"},
        {"key": "k", "label": "Potassium", "unit": "kg/ha", "value": num("potassium_kg_ha") or 280, "status": "high"},
    ]
    if soil_text:
        import re

        def grab(pat, default=0.0):
            m = re.search(pat, soil_text, re.I)
            return float(m.group(1)) if m else default

        params[0]["value"] = grab(r"ph[:\s]*([0-9.]+)", params[0]["value"])
        params[1]["value"] = grab(r"ec[:\s]*([0-9.]+)", params[1]["value"])
        params[2]["value"] = grab(r"(?:organic carbon|oc)[:\s]*([0-9.]+)", params[2]["value"])
        params[3]["value"] = grab(r"nitrogen[^0-9]*([0-9.]+)", params[3]["value"])
        params[4]["value"] = grab(r"phosphorus[^0-9]*([0-9.]+)", params[4]["value"])
        params[5]["value"] = grab(r"potassium[^0-9]*([0-9.]+)", params[5]["value"])

    fert = fertilizer_planner.calculate_fertilizer_bags(
        crop=crop,
        acreage=acreage,
        N_kg_ha=params[3]["value"],
        P_kg_ha=params[4]["value"],
        K_kg_ha=params[5]["value"],
    )
    return {
        "source": "backend",
        "generated_at": _now(),
        "crop": crop,
        "acreage": acreage,
        "soil_text": soil_text,
        "params": params,
        "fertilizer": fert,
        "schedule_mr": "माती परीक्षणानुसार बेसल + टॉप-ड्रेसिंग; ओलवाणीसोबत मात्रा द्या.",
        "schedule_en": "Follow soil-test based basal + top-dress splits with irrigation.",
    }


def predictive(crop: str, acreage: float, temperature_c: float = 30.0, humidity_pct: float = 75.0, farm_id: str = "FARM_101") -> dict[str, Any]:
    y = yield_model.predict_yield(crop=crop, acreage=acreage, irrigation_quality="Excellent Drip")
    irr = irrigation_model.calculate_water_requirement(
        crop=crop, acreage=acreage, temperature_c=temperature_c, humidity_pct=humidity_pct
    )
    audit = workflow_engine.run_farm_health_checks(farm_id)
    alerts = []
    # Normalize workflow output into UI alerts
    if isinstance(audit, dict):
        for a in audit.get("alerts") or audit.get("checks") or audit.get("actions") or []:
            if isinstance(a, dict):
                alerts.append(
                    {
                        "sev": a.get("severity") or a.get("level") or "info",
                        "trigger": a.get("trigger") or a.get("type") or "CHECK",
                        "msgMr": a.get("message_mr") or a.get("message") or str(a),
                        "msgEn": a.get("message") or str(a),
                        "action": a.get("action") or a.get("recommendation") or "review",
                    }
                )
            else:
                alerts.append({"sev": "info", "trigger": "CHECK", "msgMr": str(a), "msgEn": str(a), "action": "review"})
    if not alerts:
        alerts = [
            {
                "sev": "info",
                "trigger": "SYSTEM",
                "msgMr": "फार्म ऑडिट पूर्ण — मोठ्या अलर्ट नाहीत.",
                "msgEn": "Farm audit complete — no critical alerts.",
                "action": "continue scouting",
            }
        ]
    # simple month distribution for chart (managed)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    dist = [max(1, int(8 + 6 * ((i % 5) - 2))) for i in range(12)]
    return {
        "source": "backend",
        "generated_at": _now(),
        "yield": y,
        "irrigation": irr,
        "workflow": audit,
        "alerts": alerts,
        "chart": {"months": months, "dist": dist, "per_acre": y.get("predicted_yield_per_acre")},
    }


def taxonomy_bundle() -> dict[str, Any]:
    crops = []
    categories = []
    stages = []
    districts = []
    try:
        from mini.taxonomy.domains import TAXONOMY, list_crop_stages
        from mini.taxonomy.regions import list_mh_districts

        for c in TAXONOMY.get("crops") or []:
            crops.append(
                {
                    "en": c.get("name_en"),
                    "mr": c.get("name_mr"),
                    "hi": c.get("name_hi"),
                    "group": c.get("group") or "Crop",
                    "sci": c.get("scientific_name") or "",
                    "aliases": c.get("aliases") or [],
                }
            )
        for cat in TAXONOMY.get("categories") or []:
            categories.append(
                {
                    "code": (cat.get("id") or cat.get("name_en") or "?")[:6].upper(),
                    "name": cat.get("name_en"),
                    "mr": cat.get("name_mr"),
                    "docs": len(cat.get("aspects") or []),
                }
            )
        for st in list_crop_stages():
            stages.append({"code": (st.get("id") or "")[:3].upper(), "en": st.get("name_en"), "mr": st.get("name_mr")})
        districts = list_mh_districts()
        version = TAXONOMY.get("version") or "1.0.0"
    except Exception:
        for c in kb_loader.crops_and_diseases.get("crops") or []:
            crops.append({"en": c.get("name_en"), "mr": c.get("name_mr"), "hi": c.get("name_hi"), "group": "KB", "sci": c.get("scientific_name") or "", "aliases": []})
        categories = [
            {"code": "CROP", "name": "Crop", "mr": "पीक", "docs": len(crops)},
            {"code": "DIS", "name": "Disease", "mr": "रोग", "docs": len(kb_loader.crops_and_diseases.get("diseases_and_pests") or [])},
        ]
        stages = []
        districts = []
        version = "kb"

    return {
        "source": "backend",
        "version": version,
        "crops": crops,
        "categories": categories,
        "stages": stages,
        "districts": districts,
        "generated_at": _now(),
    }


def factory_status() -> dict[str, Any]:
    out: dict[str, Any] = {
        "source": "backend",
        "generated_at": _now(),
        "records": 0,
        "langs": {"mr": 0, "en": 0, "hi": 0},
        "cats": {},
        "dupPct": 0,
        "missingPct": 0,
        "gaps": [],
        "workers": [
            {"id": "W-INGEST", "desc": "Pull sources → lake/raw", "eta": "8s", "status": "ready"},
            {"id": "W-QUALITY", "desc": "Validate → clean → dedup", "eta": "12s", "status": "ready"},
            {"id": "W-STANDARDIZE", "desc": "Schema v1 → train/val/test", "eta": "6s", "status": "ready"},
            {"id": "W-ANALYZE", "desc": "Coverage & gap report", "eta": "9s", "status": "ready"},
            {"id": "W-QASYNTH", "desc": "Expert QA packs", "eta": "74s", "status": "ready"},
            {"id": "W-KGBUILD", "desc": "Knowledge graph builder", "eta": "15s", "status": "ready"},
            {"id": "W-TOKEN", "desc": "Domain tokenizer", "eta": "120s", "status": "ready"},
            {"id": "W-PRETRAIN", "desc": "Mini ~1M harness", "eta": "30s", "status": "ready"},
            {"id": "W-SFT", "desc": "Instruction + agri-QA SFT", "eta": "90s", "status": "ready"},
            {"id": "W-EVAL", "desc": "Gold QA + gates scorecard", "eta": "45s", "status": "ready"},
            {"id": "W-QUANT", "desc": "INT8/INT4 + size budgets", "eta": "40s", "status": "ready"},
            {"id": "W-DEPLOY", "desc": "Package + version registry", "eta": "15s", "status": "ready"},
        ],
        "reports": {},
    }
    try:
        from mini.paths import DATASETS_DIR, EVAL_DIR, LAKE_ROOT, MODELS_DIR, TOKENIZER_DIR
        import json

        for name, path in [
            ("analyze", LAKE_ROOT / "ANALYZE_LATEST.json"),
            ("qasynth", LAKE_ROOT / "QASYNTH_LATEST.json"),
            ("qasynth_mini", DATASETS_DIR / "QASYNTH_LATEST.json"),
            ("kg", DATASETS_DIR / "KG_LATEST.json"),
            ("tokenizer", TOKENIZER_DIR / "TOKENIZER_LATEST.json"),
            ("pretrain", MODELS_DIR / "PRETRAIN_LATEST.json"),
            ("sft", MODELS_DIR / "SFT_LATEST.json"),
            ("eval", EVAL_DIR / "EVAL_LATEST.json"),
            ("quant", MODELS_DIR / "QUANT_LATEST.json"),
            ("deploy", MODELS_DIR / "DEPLOY_LATEST.json"),
            ("registry", MODELS_DIR / "VERSION_REGISTRY.json"),
            ("param_count", MODELS_DIR / "PARAM_COUNT.json"),
        ]:
            if path.exists():
                try:
                    out["reports"][name] = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    out["reports"][name] = {"path": str(path), "ok": True}
        a = out["reports"].get("analyze") or {}
        q = out["reports"].get("qasynth") or out["reports"].get("qasynth_mini") or {}
        summary = a.get("summary") or {}
        counts = q.get("counts") or {}
        out["records"] = summary.get("total_records") or counts.get("total") or out["records"]
        out["dupPct"] = summary.get("duplicate_rate_pct") or out["dupPct"]
        by_lang = q.get("by_language") or summary.get("by_language") or {}
        total_lang = sum(by_lang.values()) or 1
        out["langs"] = {
            "mr": round(100 * by_lang.get("mr", 0) / total_lang, 1),
            "en": round(100 * by_lang.get("en", 0) / total_lang, 1),
            "hi": round(100 * by_lang.get("hi", 0) / total_lang, 1),
        }
        by_cat = q.get("by_category") or summary.get("by_category") or {}
        if by_cat:
            s = sum(by_cat.values()) or 1
            out["cats"] = {k: round(100 * v / s, 1) for k, v in list(by_cat.items())[:12]}
        gaps = (a.get("gaps") or summary.get("gaps") or [])[:8]
        out["gaps"] = [
            {"crop": g.get("crop") or g.get("entity") or "—", "category": g.get("category") or "—", "gap": g.get("message") or g.get("gap") or str(g)}
            if isinstance(g, dict)
            else {"crop": "—", "category": "—", "gap": str(g)}
            for g in gaps
        ]
        if not out["records"]:
            # managed defaults when lake empty
            out["records"] = kb_loader.knowledge_stats().get("total_documents") or 0
            out["note"] = "Lake reports not found — showing KB document count"
    except Exception as e:
        out["error"] = str(e)
        out["note"] = "Factory module partially unavailable"
    return out


def rag_explorer(query: str, crop: str | None = None, top_k: int = 8, enable_web: bool = True, enable_tools: bool = True) -> dict[str, Any]:
    from app.knowledge.advanced_rag import advanced_rag

    r = advanced_rag.retrieve(
        query,
        crop=crop,
        location="Maharashtra",
        top_k=top_k,
        enable_web=enable_web,
        enable_tools=enable_tools,
        force_web=False,
    )
    fused = r.get("fused_documents") or r.get("documents") or r.get("results") or []
    docs = []
    for i, d in enumerate(fused[:top_k]):
        docs.append(
            {
                "title": d.get("title") or f"Document {i + 1}",
                "origin": d.get("origin") or d.get("source_type") or "KB",
                "category": d.get("category") or "GEN",
                "source": d.get("source") or "platform",
                "score": float(d.get("rrf_score") or d.get("score") or 0.5),
                "snippet": (d.get("content") or d.get("snippet") or d.get("text") or "")[:240],
            }
        )
    backends = [
        {"name": "Hybrid (BM25 + dense)", "backend": str(hybrid_retriever.backend_info()), "ms": 200, "docs": r.get("local_hit_count") or len(docs)},
        {"name": "GraphRAG", "backend": "networkx", "ms": 180, "docs": len((r.get("graph") or {}).get("crop_ecosystems") or [])},
        {"name": "Tools", "backend": "registry", "ms": 300, "docs": len(r.get("tools_used") or [])},
        {"name": "Web", "backend": "web", "ms": 800, "docs": len(r.get("web_results") or [])},
    ]
    return {
        "source": "backend",
        "generated_at": _now(),
        "query": query,
        "metrics": {
            "local": r.get("local_hit_count") or 0,
            "fused": len(docs),
            "web": len(r.get("web_results") or []),
            "tools": len(r.get("tools_used") or []),
        },
        "docs": docs,
        "backends": backends,
        "query_plan": r.get("query_plan"),
        "tools_used": r.get("tools_used") or [],
        "raw_keys": list(r.keys()),
    }
