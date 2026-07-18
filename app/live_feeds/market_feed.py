from app.knowledge.dataset_loader import kb_loader

class MarketFeedProvider:
    """Agmarknet live APMC Mandi Market price provider."""

    def __init__(self):
        self.prices_db = kb_loader.market_prices.get("markets", [])

    def get_market_prices(self, crop: str | None = None, district: str | None = None) -> list[dict]:
        results = []
        for m in self.prices_db:
            crop_match = crop is None or crop.lower() in m["crop"].lower() or crop in m.get("crop_mr", "")
            dist_match = district is None or district.lower() in m["district"].lower()

            if crop_match and dist_match:
                results.append(m)

        if not results and (crop or district):
            # Fallback to all markets if specific filter yields no records
            return self.prices_db
        return results

    def get_market_summary_for_crop(self, crop: str) -> dict:
        matched = [m for m in self.prices_db if crop.lower() in m["crop"].lower()]
        if not matched:
            return {"crop": crop, "found": False, "message": f"No active market feeds found for {crop}."}

        avg_modal = sum(m["modal_price_rs_quintal"] for m in matched) / len(matched)
        max_price = max(m["max_price_rs_quintal"] for m in matched)
        min_price = min(m["min_price_rs_quintal"] for m in matched)

        return {
            "crop": crop,
            "found": True,
            "average_modal_price_rs_quintal": round(avg_modal, 2),
            "highest_market_price_rs_quintal": max_price,
            "lowest_market_price_rs_quintal": min_price,
            "reporting_mandis": [m["mandi"] for m in matched],
            "records": matched
        }

market_feed = MarketFeedProvider()
