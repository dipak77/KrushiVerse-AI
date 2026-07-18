from app.live_feeds.weather_feed import weather_feed
from app.live_feeds.market_feed import market_feed
from app.live_feeds.iot_feed import iot_feed
from app.live_feeds.satellite_feed import satellite_feed

def test_weather_feed():
    w = weather_feed.get_weather("Pune")
    assert w["location"] == "Pune"
    assert "temperature_c" in w
    assert "relative_humidity_pct" in w

def test_market_feed():
    prices = market_feed.get_market_prices(crop="Pomegranate")
    assert len(prices) > 0
    summary = market_feed.get_market_summary_for_crop("Pomegranate")
    assert summary["found"] is True

def test_iot_feed():
    iot = iot_feed.get_sensor_telemetry("FARM_101")
    assert "soil_moisture_vol_pct" in iot["sensors"]

def test_satellite_feed():
    sat = satellite_feed.get_satellite_indices("FARM_101")
    assert sat["indices"]["NDVI_normalized_difference_vegetation_index"] > 0
