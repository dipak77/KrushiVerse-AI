import random
from datetime import datetime

class IoTSensorFeedProvider:
    """IoT Soil Moisture, Temperature, and EC sensor telemetry provider."""

    def get_sensor_telemetry(self, farm_id: str = "FARM_101") -> dict:
        moisture_pct = round(random.uniform(22.0, 38.0), 1)
        soil_temp = round(random.uniform(21.0, 27.5), 1)
        ec_val = round(random.uniform(0.35, 0.65), 2)
        ambient_lux = random.randint(35000, 75000)

        irrigation_needed = moisture_pct < 25.0

        return {
            "farm_id": farm_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sensors": {
                "soil_moisture_vol_pct": moisture_pct,
                "soil_temperature_c": soil_temp,
                "electrical_conductivity_dS_m": ec_val,
                "solar_irradiance_lux": ambient_lux
            },
            "status": {
                "soil_moisture_status": "Optimal" if 25.0 <= moisture_pct <= 35.0 else ("Dry (Needs Irrigation)" if moisture_pct < 25.0 else "Saturated"),
                "irrigation_trigger_recommended": irrigation_needed
            }
        }

iot_feed = IoTSensorFeedProvider()
