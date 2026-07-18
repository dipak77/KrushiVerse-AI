from app.agents.base import BaseAgriAgent

class FinanceAgent(BaseAgriAgent):
    def __init__(self):
        super().__init__(
            name="Finance Agent",
            description="Estimates crop production budget, loan requirements, return on investment, and insurance claims."
        )

    def execute(self, query: str, context: dict) -> dict:
        acreage = context.get("acreage", 2.0)
        crop = context.get("crop", "Pomegranate")

        # Basic cost estimation per acre
        cost_per_acre = 35000.0 if "pomegranate" in crop.lower() else (22000.0 if "cotton" in crop.lower() else 18000.0)
        total_cost = cost_per_acre * acreage

        # Estimated revenue
        predicted_yield = context.get("predicted_yield", 60.0 * acreage)
        avg_price = context.get("avg_price", 8000.0)
        total_revenue = (predicted_yield / 10.0) * avg_price  # Convert quintals / price estimate

        net_profit = total_revenue - total_cost

        return {
            "agent": self.name,
            "financial_summary": {
                "crop": crop,
                "acreage": acreage,
                "estimated_input_cost_inr": round(total_cost, 2),
                "estimated_gross_revenue_inr": round(total_revenue, 2),
                "estimated_net_profit_inr": round(net_profit, 2),
                "crop_insurance_coverage_en": "Covered under PMFBY at ₹1 enrollment fee in Maharashtra."
            }
        }
