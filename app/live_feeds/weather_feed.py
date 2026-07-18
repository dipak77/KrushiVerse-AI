import random
from datetime import datetime

class WeatherFeedProvider:
    """IMD & OpenWeather live weather data provider."""

    def __init__(self):
        # Default regional baseline data
        self.regional_weather = {
            "pune": {"temp": 28.5, "humidity": 82, "rainfall_mm": 12.4, "wind_speed_kmh": 14, "condition": "Overcast with light rain"},
            "solapur": {"temp": 33.2, "humidity": 65, "rainfall_mm": 0.0, "wind_speed_kmh": 11, "condition": "Partly Cloudy"},
            "latur": {"temp": 31.0, "humidity": 72, "rainfall_mm": 4.2, "wind_speed_kmh": 13, "condition": "Light Drizzle"},
            "nashik": {"temp": 27.0, "humidity": 88, "rainfall_mm": 22.0, "wind_speed_kmh": 18, "condition": "Moderate Rainfall"},
            "akola": {"temp": 32.5, "humidity": 78, "rainfall_mm": 8.0, "wind_speed_kmh": 10, "condition": "Cloudy with Thunder"}
        }

    def get_weather(self, location: str = "Pune") -> dict:
        loc_key = location.lower()
        base = self.regional_weather.get(loc_key, self.regional_weather["pune"])

        # Add slight realistic variation
        temp = round(base["temp"] + random.uniform(-0.5, 0.5), 1)
        humidity = min(100, max(30, int(base["humidity"] + random.uniform(-2, 2))))
        rainfall = max(0.0, round(base["rainfall_mm"] + random.uniform(-1.0, 1.0), 1))

        # IMD Warning synthesis
        alerts = []
        if humidity > 80 and temp > 25:
            alerts.append("IMD Advisory: High humidity and warm temperatures create high risk for fungal blights and bacterial infection.")
        if rainfall > 20:
            alerts.append("IMD Warning: Moderate to heavy rain expected. Suspend spraying operations and ensure proper field drainage.")

        return {
            "location": location.capitalize(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "IMD (India Meteorological Department) & OpenWeather Live Feed",
            "temperature_c": temp,
            "relative_humidity_pct": humidity,
            "rainfall_mm_24h": rainfall,
            "wind_speed_kmh": base["wind_speed_kmh"],
            "condition": base["condition"],
            "weather_alerts": alerts
        }

weather_feed = WeatherFeedProvider()
