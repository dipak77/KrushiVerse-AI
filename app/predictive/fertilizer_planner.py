class FertilizerPlannerModel:
    """Calculates exact bag requirements for chemical and organic fertilizers per target crop and soil test level."""

    def calculate_fertilizer_bags(self, crop: str, acreage: float, N_kg_ha: float, P_kg_ha: float, K_kg_ha: float) -> dict:
        crop_lower = crop.lower()

        # Default recommended NPK in kg/acre for standard crops
        defaults = {
            "cotton": {"N": 40, "P": 20, "K": 20},
            "soybean": {"N": 12, "P": 24, "K": 16},
            "sugarcane": {"N": 100, "P": 45, "K": 45},
            "pomegranate": {"N": 60, "P": 25, "K": 50},
            "onion": {"N": 40, "P": 20, "K": 20}
        }

        rec = defaults.get(crop_lower, {"N": 35, "P": 20, "K": 20})

        # Soil credit adjustment (if soil N is low <200 kg/ha, maintain target; if soil N is high >300 kg/ha, reduce target by 20%)
        target_N = rec["N"] * (1.15 if N_kg_ha < 200 else (0.85 if N_kg_ha > 300 else 1.0)) * acreage
        target_P = rec["P"] * (1.15 if P_kg_ha < 20 else (0.85 if P_kg_ha > 35 else 1.0)) * acreage
        target_K = rec["K"] * (1.15 if K_kg_ha < 150 else (0.85 if K_kg_ha > 280 else 1.0)) * acreage

        # Conversion to Commercial Fertilizers:
        # DAP (18% N, 46% P2O5) - handles initial P and part of N
        dap_kg = (target_P / 0.46)
        n_supplied_by_dap = dap_kg * 0.18

        remaining_N = max(0.0, target_N - n_supplied_by_dap)
        urea_kg = (remaining_N / 0.46) # Urea is 46% N

        # MOP (Muriate of Potash - 60% K2O)
        mop_kg = (target_K / 0.60)

        # Standard bag weights in India: Urea (45 kg), DAP (50 kg), MOP (50 kg)
        bags_urea = round(urea_kg / 45.0, 1)
        bags_dap = round(dap_kg / 50.0, 1)
        bags_mop = round(mop_kg / 50.0, 1)

        return {
            "crop": crop.capitalize(),
            "acreage": acreage,
            "net_nutrient_requirement_kg": {
                "Nitrogen_N": round(target_N, 1),
                "Phosphorus_P": round(target_P, 1),
                "Potassium_K": round(target_K, 1)
            },
            "recommended_fertilizer_bags": {
                "Urea_45kg_bags": bags_urea,
                "DAP_50kg_bags": bags_dap,
                "MOP_50kg_bags": bags_mop
            },
            "organic_manure_recommendation": f"FYM (Farmyard Manure): {int(2.5 * acreage)} Tonnes or Vermicompost: {int(1.0 * acreage)} Tonnes as basal dressing.",
            "application_schedule_mr": f"{crop.capitalize()} पिकासाठी पेरणी किंवा ताण सोडताना {bags_dap} पोती डीएपी व {bags_mop} पोती एमओपी द्यावी. उरलेले युरिया {bags_urea} पोती २ ते ३ समान हप्त्यात द्यावे."
        }

fertilizer_planner = FertilizerPlannerModel()
