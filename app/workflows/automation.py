from app.live_feeds.weather_feed import weather_feed
from app.live_feeds.iot_feed import iot_feed
from app.memory.farm_memory import farm_memory_store

class WorkflowAutomationEngine:
    """Automated workflow trigger, alerts, and seasonal calendar execution engine."""

    def run_farm_health_checks(self, farm_id: str = "FARM_101") -> dict:
        farm = farm_memory_store.get_farm(farm_id)
        if not farm:
            return {"status": "error", "message": f"Farm {farm_id} not found."}

        location = farm["location"].get("district", "Pune")
        current_crop = farm["current_crop"].get("crop_name", "Pomegranate")

        weather = weather_feed.get_weather(location)
        telemetry = iot_feed.get_sensor_telemetry(farm_id)

        automated_actions = []

        # Weather humidity check
        if weather["relative_humidity_pct"] > 80:
            automated_actions.append({
                "action": "AUTOMATED_ALERT_DISPATCH",
                "priority": "HIGH",
                "trigger": "High Ambient Humidity (>80%)",
                "message_mr": f"हवामानात आर्द्रता {weather['relative_humidity_pct']}% आहे. {current_crop} पिकावर बुरशीजन्य रोगाचा धोका वाढला आहे. प्रतिबंधात्मक फवारणीचे नियोजन करा.",
                "message_en": f"Humidity reached {weather['relative_humidity_pct']}%. Increased fungal outbreak risk for {current_crop}. Plan preventative fungicide spray."
            })

        # IoT soil moisture check
        moisture = telemetry["sensors"]["soil_moisture_vol_pct"]
        if moisture < 25.0:
            automated_actions.append({
                "action": "IRRIGATION_PUMP_TRIGGER",
                "priority": "MEDIUM",
                "trigger": f"Soil Moisture dropped to {moisture}%",
                "message_mr": f"मातीतील ओलावा {moisture}% वर आला आहे. ठिबक सिंचन २ तासांसाठी चालू करण्याची शिफारस आहे.",
                "message_en": f"Soil moisture dropped to {moisture}%. Turn on drip irrigation for 2 hours."
            })

        # Record event log in memory
        farm_memory_store.log_action(
            farm_id,
            "Automated Telemetry Audit",
            f"Evaluated weather ({weather['temperature_c']}°C, {weather['relative_humidity_pct']}%) and IoT soil moisture ({moisture}%). Triggered {len(automated_actions)} automation actions."
        )

        return {
            "farm_id": farm_id,
            "farmer_name": farm["farmer_name"],
            "crop": current_crop,
            "weather_snapshot": weather,
            "iot_snapshot": telemetry,
            "triggered_workflows": automated_actions
        }

workflow_engine = WorkflowAutomationEngine()
