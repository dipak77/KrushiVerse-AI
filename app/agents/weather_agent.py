from app.agents.base import BaseAgriAgent
from app.live_feeds.weather_feed import weather_feed

class WeatherAgent(BaseAgriAgent):
    def __init__(self):
        super().__init__(
            name="Weather Agent",
            description="Analyzes IMD & OpenWeather live forecasts, rainfall, humidity, and microclimate advisories."
        )

    def execute(self, query: str, context: dict) -> dict:
        location = context.get("location", "Pune")
        weather_data = weather_feed.get_weather(location)
        return {
            "agent": self.name,
            "status": "success",
            "data": weather_data
        }
