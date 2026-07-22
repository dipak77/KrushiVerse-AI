from app.agents.base import BaseAgriAgent
from app.vision.disease_classifier import vision_classifier
from app.predictive.pest_outbreak_model import pest_outbreak_model

class DiseaseAgent(BaseAgriAgent):
    def __init__(self):
        super().__init__(
            name="Disease Agent",
            description="Identifies crop pests, pathogens, and symptoms, prescribing organic and chemical treatments."
        )

    def execute(self, query: str, context: dict) -> dict:
        crop = context.get("crop", "Pomegranate")
        temp = context.get("temperature_c", 29.0)
        humidity = context.get("humidity_pct", 82.0)
        rainfall = context.get("rainfall_mm", 10.0)

        diagnosis = vision_classifier.diagnose_image(
            filename=context.get("image_filename", "sample.jpg"),
            crop_hint=crop,
            query=query
        )

        risk = pest_outbreak_model.calculate_outbreak_risk(crop, temp, humidity, rainfall)

        return {
            "agent": self.name,
            "diagnosis": diagnosis,
            "outbreak_risk_analysis": risk
        }
