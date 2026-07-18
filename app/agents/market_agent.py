from app.agents.base import BaseAgriAgent
from app.live_feeds.market_feed import market_feed

class MarketAgent(BaseAgriAgent):
    def __init__(self):
        super().__init__(
            name="Market Agent",
            description="Fetches Agmarknet mandi market prices, price trends, and optimal harvest sale timing."
        )

    def execute(self, query: str, context: dict) -> dict:
        crop = context.get("crop", "Pomegranate")
        district = context.get("district", "Solapur")

        summary = market_feed.get_market_summary_for_crop(crop)
        prices = market_feed.get_market_prices(crop=crop, district=district)

        return {
            "agent": self.name,
            "market_summary": summary,
            "mandi_prices": prices
        }
