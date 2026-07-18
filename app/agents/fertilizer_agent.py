from app.agents.base import BaseAgriAgent
from app.predictive.fertilizer_planner import fertilizer_planner

class FertilizerAgent(BaseAgriAgent):
    def __init__(self):
        super().__init__(
            name="Fertilizer Agent",
            description="Calculates exact commercial fertilizer dosages (Urea, DAP, MOP) and fertigation schedules."
        )

    def execute(self, query: str, context: dict) -> dict:
        crop = context.get("crop", "Pomegranate")
        acreage = context.get("acreage", 2.0)
        n_val = context.get("nitrogen_kg_ha", 180.0)
        p_val = context.get("phosphorus_kg_ha", 22.0)
        k_val = context.get("potassium_kg_ha", 280.0)

        plan = fertilizer_planner.calculate_fertilizer_bags(crop, acreage, n_val, p_val, k_val)

        return {
            "agent": self.name,
            "fertilizer_plan": plan
        }
