from app.agents.base import BaseAgriAgent
from app.vision.ocr_processor import ocr_processor

class SoilAgent(BaseAgriAgent):
    def __init__(self):
        super().__init__(
            name="Soil Agent",
            description="Evaluates Soil Health Card parameters (pH, NPK, Organic Carbon, EC) and soil type suitability."
        )

    def execute(self, query: str, context: dict) -> dict:
        soil_card_text = context.get("soil_card_text", "pH: 7.2, Nitrogen: 180 kg/ha, Phosphorus: 22 kg/ha, Potassium: 280 kg/ha")
        extracted = ocr_processor.process_soil_card(soil_card_text)

        return {
            "agent": self.name,
            "soil_analysis": extracted,
            "summary_mr": f"माती चाचणीनुसार: नत्र प्रमाण कमी ({extracted['extracted_parameters']['nitrogen_kg_ha']} kg/ha), स्फुरद मध्यम, व पालाश भरपूर आहे."
        }
