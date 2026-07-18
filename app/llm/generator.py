class MarathiResponseSynthesizer:
    """Generates localized agricultural responses in Marathi and English."""

    MARATHI_TEMPLATES = {
        "greeting": "नमस्कार शेतकरी मित्र! AI कृषी मित्र प्रणालीमध्ये आपले स्वागत आहे.",
        "weather_prefix": "🌧️ **हवामान अंदाज व सल्ला:**",
        "crop_prefix": "🌱 **पीक नियोजन सल्ला:**",
        "disease_prefix": "🔬 **रोग व कीड निदान सल्ले:**",
        "market_prefix": "📈 **बाजारभाव माहिती:**",
        "soil_prefix": "🧪 **माती आरोग्य व खत व्यवस्थापन:**",
        "gov_prefix": "🏛️ **शासकीय योजना माहिती:**",
        "footer": "\n---\n*कृषी विज्ञान केंद्र आणि ICAR च्या शिफारशींवर आधारित AI कृषी मित्र सल्ला.*"
    }

    def synthesize(self, plan_summary: str, agent_outputs: dict, language: str = "mr") -> str:
        """Synthesize specialized agent outputs into a unified, clear response in Marathi or English."""
        if language.lower() in ["mr", "marathi"]:
            return self._synthesize_marathi(plan_summary, agent_outputs)
        else:
            return self._synthesize_english(plan_summary, agent_outputs)

    def _synthesize_marathi(self, plan_summary: str, agent_outputs: dict) -> str:
        parts = [self.MARATHI_TEMPLATES["greeting"], f"\n**विश्लेषण निष्कर्ष:** {plan_summary}\n"]

        if "weather" in agent_outputs:
            w = agent_outputs["weather"]
            parts.append(f"{self.MARATHI_TEMPLATES['weather_prefix']}")
            parts.append(f"- **ठिकाण:** {w.get('location', 'महाराष्ट्र')}")
            parts.append(f"- **तापमान:** {w.get('temperature_c')}°C | **आर्द्रता:** {w.get('relative_humidity_pct')}% | **पाऊस:** {w.get('rainfall_mm_24h')} मिमी")
            if w.get("weather_alerts"):
                for alert in w["weather_alerts"]:
                    parts.append(f"  ⚠️ *{alert}*")
            parts.append("")

        if "disease" in agent_outputs:
            d = agent_outputs["disease"]
            parts.append(f"{self.MARATHI_TEMPLATES['disease_prefix']}")
            parts.append(f"- **निदान/कीड:** {d.get('disease_identified_mr', d.get('disease_identified_en'))} ({d.get('detected_crop')})")
            parts.append(f"- **लक्षणे:** {d.get('symptoms_mr', d.get('symptoms_en'))}")
            if "organic_treatment" in d:
                parts.append(f"- 🌿 **सेंद्रिय उपाय:** {d['organic_treatment'].get('mr', d['organic_treatment'].get('en'))}")
            if "chemical_treatment" in d:
                parts.append(f"- 🧪 **रासायनिक उपाय:** {d['chemical_treatment'].get('mr', d['chemical_treatment'].get('en'))}")
            parts.append("")

        if "soil" in agent_outputs or "fertilizer" in agent_outputs:
            parts.append(f"{self.MARATHI_TEMPLATES['soil_prefix']}")
            if "soil" in agent_outputs:
                s = agent_outputs["soil"]
                parts.append(f"- {s.get('summary_mr', 'माती चाचणीनुसार खताचे योग्य प्रमाण वापरणे आवश्यक आहे.')}")
            if "fertilizer" in agent_outputs:
                f = agent_outputs["fertilizer"]
                if "application_schedule_mr" in f:
                    parts.append(f"- **मात्रा:** {f['application_schedule_mr']}")
            parts.append("")

        if "market" in agent_outputs:
            m = agent_outputs["market"]
            parts.append(f"{self.MARATHI_TEMPLATES['market_prefix']}")
            if "average_modal_price_rs_quintal" in m:
                parts.append(f"- **सरासरी बाजारभाव:** ₹{m['average_modal_price_rs_quintal']} / क्विंटल (उच्चतम: ₹{m.get('highest_market_price_rs_quintal')})")
            elif isinstance(m, list) and len(m) > 0:
                item = m[0]
                parts.append(f"- **मंडी:** {item.get('mandi')} | **दर:** ₹{item.get('modal_price_rs_quintal')} / क्विंटल ({item.get('trend', 'स्थिर')})")
            parts.append("")

        if "government" in agent_outputs:
            g = agent_outputs["government"]
            parts.append(f"{self.MARATHI_TEMPLATES['gov_prefix']}")
            if isinstance(g, list):
                for sch in g[:2]:
                    parts.append(f"- **{sch.get('name_mr', sch.get('name_en'))}:** {sch.get('benefits_mr', sch.get('benefits_en'))}")
            parts.append("")

        parts.append(self.MARATHI_TEMPLATES["footer"])
        return "\n".join(parts)

    def _synthesize_english(self, plan_summary: str, agent_outputs: dict) -> str:
        parts = ["Greetings! Welcome to AI Krushi Mitra Agriculture Platform.", f"\n**Plan Summary:** {plan_summary}\n"]

        for key, val in agent_outputs.items():
            parts.append(f"### {key.capitalize()} Advisory")
            parts.append(f"```json\n{val}\n```\n")

        return "\n".join(parts)

response_synthesizer = MarathiResponseSynthesizer()
