from app.agents.base import BaseAgriAgent
from app.knowledge.dataset_loader import kb_loader

class GovernmentAgent(BaseAgriAgent):
    def __init__(self):
        super().__init__(
            name="Government Agent",
            description="Searches government agriculture schemes (PM-KISAN, PMFBY, PoCRA, Magel Tyala Shettale) and subsidies."
        )

    def execute(self, query: str, context: dict) -> dict:
        schemes = kb_loader.government_schemes.get("schemes", [])
        return {
            "agent": self.name,
            "applicable_schemes": schemes
        }
