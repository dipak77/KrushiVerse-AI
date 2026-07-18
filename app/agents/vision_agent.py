from app.agents.base import BaseAgriAgent
from app.vision.disease_classifier import vision_classifier

class VisionAgent(BaseAgriAgent):
    def __init__(self):
        super().__init__(
            name="Vision Agent",
            description="Processes leaf image uploads and agricultural documentation using vision models."
        )

    def execute(self, query: str, context: dict) -> dict:
        filename = context.get("image_filename", "uploaded_crop_leaf.jpg")
        crop = context.get("crop", "Pomegranate")
        diag = vision_classifier.diagnose_image(filename=filename, crop_hint=crop)

        return {
            "agent": self.name,
            "vision_analysis": diag
        }
