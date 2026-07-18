from app.agents.base import BaseAgriAgent
from app.knowledge.hybrid_search import hybrid_retriever

class CropAgent(BaseAgriAgent):
    def __init__(self):
        super().__init__(
            name="Crop Agent",
            description="Provides crop calendar guidelines, sowing windows, university advisories, and growth stage management."
        )

    def execute(self, query: str, context: dict) -> dict:
        results = hybrid_retriever.hybrid_search(f"crop guide {query}", top_k=2)
        crop_name = context.get("crop", "Pomegranate")

        return {
            "agent": self.name,
            "crop": crop_name,
            "advisories": [r["doc"] for r in results],
            "guidance_mr": f"{crop_name} पिकासाठी विद्यापीठीय शिफारशींनुसार योग्य वेळेत खत व पाणी व्यवस्थापन करावे."
        }
