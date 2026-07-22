from app.live_feeds.opendata_client import opendata_client


class MarketFeedProvider:
    """Agmarknet / data.gov.in market price provider with local open KB fallback."""

    def __init__(self):
        self._prices_db = None

    @property
    def prices_db(self) -> list[dict]:
        if self._prices_db is None:
            from app.knowledge.dataset_loader import kb_loader
            self._prices_db = kb_loader.market_prices.get("markets", [])
        return self._prices_db

    def get_market_prices(self, crop: str | None = None, district: str | None = None) -> list[dict]:
        # Prefer live open-data when available
        live = opendata_client.fetch_commodity_prices(
            state="Maharashtra",
            district=district,
            commodity=crop,
            limit=40,
        )
        if live.get("mode") == "live" and live.get("records"):
            return live["records"]

        results = []
        for m in self.prices_db:
            m_crop = (m.get("crop") or m.get("crop_en") or "").lower()
            m_dist = (m.get("district") or m.get("market_name") or "").lower()

            crop_match = (
                crop is None
                or crop.lower() in m_crop
                or crop in m.get("crop_mr", "")
                or crop in m.get("crop_en", "")
            )
            dist_match = district is None or district.lower() in m_dist

            if crop_match and dist_match:
                results.append(m)

        if not results and (crop or district):
            return self.prices_db
        return results

    def get_market_summary_for_crop(self, crop: str) -> dict:
        matched = self.get_market_prices(crop=crop)
        # normalize field names from live vs local
        cleaned = []
        for m in matched:
            if "modal_price_rs_quintal" in m and m["modal_price_rs_quintal"] is not None:
                cleaned.append(m)
            elif m.get("modal_price") is not None:
                cleaned.append({**m, "modal_price_rs_quintal": m.get("modal_price")})
        if not cleaned:
            return {
                "crop": crop,
                "found": False,
                "message": f"No active market feeds found for {crop}.",
                "opendata": opendata_client.status(),
            }

        avg_modal = sum(float(m["modal_price_rs_quintal"]) for m in cleaned) / len(cleaned)
        max_price = max(float(m.get("max_price_rs_quintal") or m["modal_price_rs_quintal"]) for m in cleaned)
        min_price = min(float(m.get("min_price_rs_quintal") or m["modal_price_rs_quintal"]) for m in cleaned)

        return {
            "crop": crop,
            "found": True,
            "average_modal_price_rs_quintal": round(avg_modal, 2),
            "highest_market_price_rs_quintal": max_price,
            "lowest_market_price_rs_quintal": min_price,
            "reporting_mandis": [m.get("mandi") or m.get("market") for m in cleaned],
            "records": cleaned,
            "opendata": opendata_client.status(),
            "source_mode": "live" if any(m.get("source") == "data.gov.in" for m in cleaned) else "local_or_fallback",
        }


market_feed = MarketFeedProvider()
