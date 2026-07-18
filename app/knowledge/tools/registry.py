"""External tool registry for tool-augmented RAG (weather, market, wiki, open data)."""

from __future__ import annotations

from typing import Any, Callable

import httpx

from app.live_feeds.weather_feed import weather_feed
from app.live_feeds.market_feed import market_feed
from app.live_feeds.opendata_client import opendata_client
from app.knowledge.dataset_loader import kb_loader
from app.knowledge.web_search import web_search_provider


ToolFn = Callable[[dict], dict]


class ExternalToolRegistry:
    """
    Advanced tool layer used by the multi-source RAG orchestrator.
    Each tool returns a structured payload + RAG-ready documents.
    """

    def __init__(self):
        self.tools: dict[str, dict[str, Any]] = {
            "open_meteo_weather": {
                "description": "Live weather via Open-Meteo (free, no API key) with IMD-style local fallback",
                "fn": self.open_meteo_weather,
            },
            "local_weather": {
                "description": "Regional weather feed (demo IMD-style)",
                "fn": self.local_weather,
            },
            "market_prices": {
                "description": "Mandi market prices from open knowledge + live feed layer",
                "fn": self.market_prices,
            },
            "agmarknet_opendata": {
                "description": "Live Agmarknet commodity prices via data.gov.in (API key) with local fallback",
                "fn": self.agmarknet_opendata,
            },
            "wikipedia": {
                "description": "Wikipedia open encyclopedia search",
                "fn": self.wikipedia_tool,
            },
            "web_search": {
                "description": "DuckDuckGo + open web search for latest public info",
                "fn": self.web_search_tool,
            },
            "government_schemes": {
                "description": "Government scheme lookup from expanded open scheme KB",
                "fn": self.schemes_tool,
            },
            "open_source_catalog": {
                "description": "Catalog of open agri data sources",
                "fn": self.catalog_tool,
            },
            "crop_knowledge": {
                "description": "Structured crop package-of-practice knowledge",
                "fn": self.crop_tool,
            },
        }

    def list_tools(self) -> list[dict]:
        return [{"name": k, "description": v["description"]} for k, v in self.tools.items()]

    def run(self, name: str, params: dict | None = None) -> dict:
        params = params or {}
        if name not in self.tools:
            return {"tool": name, "ok": False, "error": f"Unknown tool: {name}", "documents": []}
        try:
            payload = self.tools[name]["fn"](params)
            return {"tool": name, "ok": True, **payload}
        except Exception as e:
            return {"tool": name, "ok": False, "error": str(e), "documents": []}

    def route_for_intents(self, intents: list[str], crops: list[str], location: str, query: str) -> list[str]:
        selected: list[str] = []
        if "weather" in intents:
            selected += ["open_meteo_weather", "local_weather"]
        if "market" in intents:
            selected.append("market_prices")
            selected.append("agmarknet_opendata")
        if "scheme" in intents:
            selected.append("government_schemes")
        if "disease" in intents or "fertilizer" in intents or "seed" in intents or "irrigation" in intents:
            selected.append("crop_knowledge")
        if any(i in intents for i in ("general", "disease", "scheme", "market", "weather")):
            selected.append("web_search")
            selected.append("wikipedia")
        selected.append("open_source_catalog")
        # unique preserve order
        seen = set()
        out = []
        for s in selected:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    # ---- tools ----

    def open_meteo_weather(self, params: dict) -> dict:
        location = params.get("location", "Pune")
        # Approximate coords for common MH districts; default Pune
        coords = {
            "pune": (18.52, 73.85),
            "solapur": (17.65, 75.90),
            "nagpur": (21.14, 79.08),
            "nashik": (19.99, 73.78),
            "latur": (18.40, 76.58),
            "akola": (20.70, 77.00),
            "mumbai": (19.07, 72.87),
            "kolhapur": (16.70, 74.24),
        }
        lat, lon = coords.get(str(location).lower(), (18.52, 73.85))
        live = None
        try:
            with httpx.Client(timeout=6.0) as client:
                r = client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
                        "timezone": "Asia/Kolkata",
                    },
                )
                if r.status_code == 200:
                    live = r.json().get("current", {})
        except Exception:
            live = None

        if live:
            data = {
                "location": location,
                "provider": "Open-Meteo (open weather API)",
                "temperature_c": live.get("temperature_2m"),
                "relative_humidity_pct": live.get("relative_humidity_2m"),
                "rainfall_mm": live.get("precipitation"),
                "wind_speed_kmh": live.get("wind_speed_10m"),
                "raw": live,
            }
        else:
            fb = weather_feed.get_weather(location)
            data = {**fb, "provider": "Local weather fallback (Open-Meteo unavailable)"}

        doc = {
            "id": f"tool_weather_{location}",
            "category": "Tool:Weather",
            "title": f"Weather tool: {location}",
            "content": (
                f"Weather for {location}: temp {data.get('temperature_c')}°C, "
                f"humidity {data.get('relative_humidity_pct')}%, "
                f"precip {data.get('rainfall_mm')}, wind {data.get('wind_speed_kmh')}. "
                f"Provider: {data.get('provider')}."
            ),
            "metadata": data,
            "source": data.get("provider", "weather_tool"),
        }
        return {"data": data, "documents": [doc]}

    def local_weather(self, params: dict) -> dict:
        location = params.get("location", "Pune")
        data = weather_feed.get_weather(location)
        doc = {
            "id": f"tool_local_weather_{location}",
            "category": "Tool:Weather",
            "title": f"Regional weather: {location}",
            "content": (
                f"{data.get('location')} weather: {data.get('temperature_c')}°C, "
                f"RH {data.get('relative_humidity_pct')}%, rain {data.get('rainfall_mm_24h')} mm, "
                f"{data.get('condition')}. Alerts: {'; '.join(data.get('weather_alerts') or [])}"
            ),
            "metadata": data,
            "source": data.get("source", "local_weather"),
        }
        return {"data": data, "documents": [doc]}

    def market_prices(self, params: dict) -> dict:
        crop = params.get("crop")
        district = params.get("district")
        data = market_feed.get_market_prices(crop=crop, district=district)
        docs = []
        rows = data if isinstance(data, list) else data.get("prices") or data.get("markets") or [data]
        if isinstance(rows, dict):
            rows = [rows]
        for i, row in enumerate(rows[:8]):
            if not isinstance(row, dict):
                continue
            docs.append({
                "id": f"tool_market_{i}",
                "category": "Tool:Market",
                "title": f"Market: {row.get('crop', crop)} @ {row.get('mandi', row.get('district', ''))}",
                "content": str(row),
                "metadata": row,
                "source": "market_tool",
            })
        # Also pull KB markets
        for i, m in enumerate(kb_loader.market_prices.get("markets", [])):
            if crop and crop.lower() not in str(m.get("crop", "")).lower():
                continue
            docs.append({
                "id": f"tool_kb_market_{i}",
                "category": "Tool:Market",
                "title": f"KB Mandi: {m.get('crop')} @ {m.get('mandi')}",
                "content": (
                    f"{m.get('crop')} modal ₹{m.get('modal_price_rs_quintal')}/q at {m.get('mandi')} "
                    f"({m.get('trend')}) on {m.get('date')}"
                ),
                "metadata": m,
                "source": m.get("source", "market_kb"),
            })
        return {"data": data, "documents": docs[:12]}

    def agmarknet_opendata(self, params: dict) -> dict:
        crop = params.get("crop")
        district = params.get("district")
        state = params.get("state") or "Maharashtra"
        result = opendata_client.fetch_commodity_prices(
            state=state,
            district=district,
            commodity=crop,
            limit=int(params.get("limit", 40)),
        )
        docs = opendata_client.to_rag_documents(result)
        return {"data": result, "documents": docs}

    def wikipedia_tool(self, params: dict) -> dict:
        query = params.get("query", "agriculture India")
        hits = web_search_provider._wikipedia_search(query, max_results=params.get("max_results", 3))
        docs = web_search_provider.to_rag_docs(hits)
        for d in docs:
            d["category"] = "Tool:Wikipedia"
        return {"data": hits, "documents": docs}

    def web_search_tool(self, params: dict) -> dict:
        query = params.get("query", "")
        hits = web_search_provider.search(query, max_results=params.get("max_results", 5))
        docs = web_search_provider.to_rag_docs(hits)
        return {"data": hits, "documents": docs}

    def schemes_tool(self, params: dict) -> dict:
        query = (params.get("query") or "").lower()
        schemes = kb_loader.government_schemes.get("schemes", [])
        matched = []
        for s in schemes:
            blob = f"{s.get('name_en','')} {s.get('name_mr','')} {s.get('benefits_en','')}".lower()
            if not query or any(tok in blob for tok in query.split() if len(tok) > 3):
                matched.append(s)
        if not matched:
            matched = schemes[:5]
        docs = []
        for s in matched[:8]:
            docs.append({
                "id": f"tool_scheme_{s.get('scheme_id')}",
                "category": "Tool:Scheme",
                "title": s.get("name_en"),
                "content": (
                    f"{s.get('name_en')}: {s.get('benefits_en')} Eligibility: {s.get('eligibility_en')} "
                    f"Portal: {s.get('portal', '')}"
                ),
                "metadata": s,
                "source": s.get("source", "scheme_tool"),
            })
        return {"data": matched[:8], "documents": docs}

    def catalog_tool(self, params: dict) -> dict:
        sources = kb_loader.open_source_catalog.get("sources", [])
        docs = [{
            "id": f"tool_catalog_{i}",
            "category": "Tool:Catalog",
            "title": s.get("name"),
            "content": f"{s.get('name')} ({s.get('type')}): {s.get('use')} — {s.get('url')}",
            "metadata": s,
            "source": "open_source_catalog",
        } for i, s in enumerate(sources)]
        return {"data": sources, "documents": docs}

    def crop_tool(self, params: dict) -> dict:
        crop = (params.get("crop") or "").lower()
        docs = []
        data = {"crops": [], "diseases": [], "fertilizer": []}
        for c in kb_loader.crops_and_diseases.get("crops", []):
            if not crop or crop in c.get("name_en", "").lower() or crop in c.get("name_mr", ""):
                data["crops"].append(c)
                docs.append({
                    "id": f"tool_crop_{c.get('crop_id')}",
                    "category": "Tool:Crop",
                    "title": f"Crop PoP: {c.get('name_en')}",
                    "content": f"{c.get('name_en')} season {c.get('season')} pests {', '.join(c.get('major_pests', []))} diseases {', '.join(c.get('major_diseases', []))}",
                    "metadata": c,
                    "source": c.get("source", "crop_tool"),
                })
        for d in kb_loader.crops_and_diseases.get("diseases_and_pests", []):
            if not crop or crop in d.get("crop_en", "").lower() or crop in d.get("name_en", "").lower():
                data["diseases"].append(d)
                docs.append({
                    "id": f"tool_dis_{d.get('id')}",
                    "category": "Tool:Disease",
                    "title": f"{d.get('name_en')} on {d.get('crop_en')}",
                    "content": f"{d.get('symptoms_en')} Organic: {d.get('organic_control_en')} Chemical: {d.get('chemical_control_en')}",
                    "metadata": d,
                    "source": d.get("source", "crop_tool"),
                })
        for f in kb_loader.soil_and_fertilizers.get("fertilizer_recommendations", []):
            if not crop or crop in f.get("crop_en", "").lower():
                data["fertilizer"].append(f)
                docs.append({
                    "id": f"tool_fert_{f.get('crop_en')}",
                    "category": "Tool:Fertilizer",
                    "title": f"NPK for {f.get('crop_en')}",
                    "content": str(f.get("recommended_npk_kg_per_acre")),
                    "metadata": f,
                    "source": f.get("source", "crop_tool"),
                })
        return {"data": data, "documents": docs[:20]}


tool_registry = ExternalToolRegistry()
