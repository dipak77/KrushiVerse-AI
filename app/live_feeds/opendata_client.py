"""Open data clients for data.gov.in / Agmarknet-style commodity prices.

Requires free API key from https://data.gov.in (set DATA_GOV_IN_API_KEY).
Without a key, methods return structured demos from the local market KB.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.config import settings



class DataGovInClient:
    def __init__(self):
        self.base_url = settings.DATA_GOV_IN_BASE_URL.rstrip("/")
        self.api_key = settings.DATA_GOV_IN_API_KEY
        self.resource_id = settings.AGMARKNET_RESOURCE_ID
        self.timeout = settings.OPENDATA_TIMEOUT_SEC
        self._cache: dict[str, tuple[float, Any]] = {}
        self.ttl = settings.OPENDATA_CACHE_TTL_SEC

    @property
    def configured(self) -> bool:
        return bool(self.api_key) and self.api_key not in ("YOUR_API_KEY", "demo", "changeme")

    def status(self) -> dict:
        return {
            "provider": "data.gov.in",
            "configured": self.configured,
            "resource_id": self.resource_id,
            "base_url": self.base_url,
            "live_enabled": settings.ENABLE_LIVE_AGMARKNET,
            "note": None
            if self.configured
            else "Set DATA_GOV_IN_API_KEY from https://data.gov.in to enable live Agmarknet pulls.",
        }

    def fetch_commodity_prices(
        self,
        *,
        state: str | None = "Maharashtra",
        district: str | None = None,
        commodity: str | None = None,
        market: str | None = None,
        limit: int = 50,
    ) -> dict:
        """
        Fetch daily mandi prices from data.gov.in Agmarknet resource.
        Falls back to local open KB when key missing or request fails.
        """
        cache_key = f"{state}|{district}|{commodity}|{market}|{limit}"
        now = time.time()
        if cache_key in self._cache and now - self._cache[cache_key][0] < self.ttl:
            return self._cache[cache_key][1]

        if not settings.ENABLE_LIVE_AGMARKNET or not self.configured:
            result = self._local_fallback(state=state, district=district, commodity=commodity, reason="api_key_missing_or_disabled")
            self._cache[cache_key] = (now, result)
            return result

        params: dict[str, Any] = {
            "api-key": self.api_key,
            "format": "json",
            "limit": limit,
            "offset": 0,
        }
        # data.gov.in filter syntax
        if state:
            params["filters[state.keyword]"] = state
            params["filters[state]"] = state
        if district:
            params["filters[district]"] = district
        if commodity:
            params["filters[commodity]"] = commodity
        if market:
            params["filters[market]"] = market

        url = f"{self.base_url}/resource/{self.resource_id}"
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                r = client.get(url, params=params)
                if r.status_code != 200:
                    # try alternate filter keys used by some resources
                    alt = {
                        "api-key": self.api_key,
                        "format": "json",
                        "limit": limit,
                    }
                    if state:
                        alt["filters[State]"] = state
                    if commodity:
                        alt["filters[Commodity]"] = commodity
                    if district:
                        alt["filters[District]"] = district
                    r = client.get(url, params=alt)
                r.raise_for_status()
                payload = r.json()
        except Exception as e:
            result = self._local_fallback(
                state=state,
                district=district,
                commodity=commodity,
                reason=f"live_fetch_failed:{type(e).__name__}:{e}",
            )
            self._cache[cache_key] = (now, result)
            return result

        records = payload.get("records") or payload.get("data") or []
        normalized = [self._normalize_record(rec) for rec in records]
        normalized = [n for n in normalized if n]
        if not normalized:
            result = self._local_fallback(state=state, district=district, commodity=commodity, reason="empty_live_response")
            self._cache[cache_key] = (now, result)
            return result

        result = {
            "ok": True,
            "source": "data.gov.in / Agmarknet live",
            "resource_id": self.resource_id,
            "count": len(normalized),
            "filters": {"state": state, "district": district, "commodity": commodity, "market": market},
            "records": normalized,
            "mode": "live",
        }
        self._cache[cache_key] = (now, result)
        return result

    def _normalize_record(self, rec: dict) -> dict | None:
        if not isinstance(rec, dict):
            return None
        # Handle varying key casings from data.gov.in
        def g(*keys, default=None):
            lower_map = {str(k).lower(): v for k, v in rec.items()}
            for k in keys:
                if k in rec:
                    return rec[k]
                if k.lower() in lower_map:
                    return lower_map[k.lower()]
            return default

        commodity = g("commodity", "Commodity", "crop")
        market = g("market", "Market", "mandi")
        if not commodity and not market:
            return None

        def num(val):
            try:
                if val is None or val == "":
                    return None
                return float(str(val).replace(",", "").strip())
            except Exception:
                return None

        return {
            "state": g("state", "State"),
            "district": g("district", "District"),
            "market": market,
            "mandi": market,
            "commodity": commodity,
            "crop": commodity,
            "variety": g("variety", "Variety"),
            "arrival_date": g("arrival_date", "Arrival_Date", "date"),
            "min_price_rs_quintal": num(g("min_price", "Min_Price", "min_price_rs_quintal")),
            "max_price_rs_quintal": num(g("max_price", "Max_Price", "max_price_rs_quintal")),
            "modal_price_rs_quintal": num(g("modal_price", "Modal_Price", "modal_price_rs_quintal")),
            "raw": rec,
            "source": "data.gov.in",
        }

    def _local_fallback(
        self,
        *,
        state: str | None,
        district: str | None,
        commodity: str | None,
        reason: str,
    ) -> dict:
        from app.knowledge.dataset_loader import kb_loader
        markets = kb_loader.market_prices.get("markets", [])
        rows = []
        for m in markets:
            if state and state.lower() not in str(m.get("state", "")).lower():
                # keep MH defaults if state filter fails lightly
                if state.lower() not in ("maharashtra", "all", ""):
                    continue
            if district and district.lower() not in str(m.get("district", "")).lower():
                continue
            if commodity and commodity.lower() not in str(m.get("crop", "")).lower():
                continue
            rows.append({
                **m,
                "commodity": m.get("crop"),
                "market": m.get("mandi"),
                "source": m.get("source", "local_market_kb"),
            })
        if not rows:
            rows = [{**m, "commodity": m.get("crop"), "market": m.get("mandi")} for m in markets[:10]]
        return {
            "ok": True,
            "source": "local open market KB (Agmarknet-style)",
            "resource_id": self.resource_id,
            "count": len(rows),
            "filters": {"state": state, "district": district, "commodity": commodity},
            "records": rows,
            "mode": "fallback",
            "fallback_reason": reason,
        }

    def to_rag_documents(self, result: dict) -> list[dict]:
        docs = []
        for i, rec in enumerate((result.get("records") or [])[:20]):
            crop = rec.get("crop") or rec.get("commodity")
            mandi = rec.get("mandi") or rec.get("market")
            modal = rec.get("modal_price_rs_quintal")
            docs.append({
                "id": f"agmarknet_{i}_{mandi}_{crop}".replace(" ", "_")[:80],
                "category": "OpenData:Agmarknet",
                "title": f"Agmarknet: {crop} @ {mandi}",
                "content": (
                    f"Mandi price for {crop} at {mandi}, {rec.get('district')}, {rec.get('state')}. "
                    f"Min ₹{rec.get('min_price_rs_quintal')}, Modal ₹{modal}, Max ₹{rec.get('max_price_rs_quintal')} /q. "
                    f"Date: {rec.get('arrival_date') or rec.get('date')}. "
                    f"Source: {result.get('source')} mode={result.get('mode')}."
                ),
                "metadata": rec,
                "source": result.get("source", "agmarknet"),
            })
        return docs


opendata_client = DataGovInClient()
