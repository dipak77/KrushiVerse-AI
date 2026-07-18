from __future__ import annotations

from typing import Any

from app.config import settings


class MarathiResponseSynthesizer:
    """Generates localized agricultural responses in Marathi and English.

    Sprint 16: optional Mini LLM synthesizer behind USE_MINI_LLM (default off).
    """

    MARATHI_TEMPLATES = {
        "greeting": "नमस्कार शेतकरी मित्र! AI कृषी मित्र प्रणालीमध्ये आपले स्वागत आहे.",
        "weather_prefix": "🌧️ **हवामान अंदाज व सल्ला:**",
        "crop_prefix": "🌱 **पीक नियोजन सल्ला:**",
        "disease_prefix": "🔬 **रोग व कीड निदान सल्ले:**",
        "market_prefix": "📈 **बाजारभाव माहिती:**",
        "soil_prefix": "🧪 **माती आरोग्य व खत व्यवस्थापन:**",
        "gov_prefix": "🏛️ **शासकीय योजना माहिती:**",
        "rag_prefix": "📚 **मल्टी-सोर्स ज्ञानाधार (RAG + Web + Tools):**",
        "footer": "\n---\n*कृषी विज्ञान केंद्र, ICAR, खुले शासकीय स्रोत, व वेब/टूल-आधारित RAG वर आधारित AI कृषी मित्र सल्ला. फवारणी/खते स्थानिक तज्ञ तपासून घ्या.*"
    }

    def __init__(self) -> None:
        self.last_meta: dict[str, Any] = {"synthesizer": "template"}

    def synthesize(
        self,
        plan_summary: str,
        agent_outputs: dict,
        language: str = "mr",
        rag_context: str | None = None,
        citations: list | None = None,
        *,
        query: str | None = None,
        crop: str | None = None,
        location: str = "Pune",
        enable_web: bool | None = None,
        use_mini: bool | None = None,
    ) -> str:
        """Synthesize specialized agent outputs into a unified response.

        When ``use_mini`` is True (or settings.USE_MINI_LLM), Mini+RAG is the
        primary synthesizer; agents remain upstream tool specialists.
        Flag off preserves classic template synthesis (backward compatible).
        """
        flag = settings.USE_MINI_LLM if use_mini is None else bool(use_mini)
        if flag and (query or plan_summary):
            try:
                from app.llm.mini_bridge import synthesize_with_mini

                answer, meta = synthesize_with_mini(
                    query or plan_summary,
                    plan_summary=plan_summary,
                    agent_outputs=agent_outputs or {},
                    language=language,
                    crop=crop,
                    location=location,
                    enable_web=enable_web,
                    citations=citations,
                )
                if answer and answer.strip():
                    self.last_meta = meta
                    return answer
            except Exception as e:
                self.last_meta = {"synthesizer": "template", "mini_error": str(e)}
        else:
            self.last_meta = {"synthesizer": "template", "use_mini_llm": False}

        if language.lower() in ["mr", "marathi"]:
            return self._synthesize_marathi(plan_summary, agent_outputs, citations=citations)
        return self._synthesize_english(plan_summary, agent_outputs, citations=citations)

    def _format_citations(self, citations: list | None, language: str = "en") -> str:
        if not citations:
            return ""
        lines = []
        title = "**स्रोत / Sources:**" if language == "mr" else "**Sources:**"
        lines.append(title)
        for i, c in enumerate(citations[:6], 1):
            url = c.get("url") or ""
            origin = c.get("origin") or ""
            src = c.get("source") or ""
            name = c.get("title") or src or origin
            if url:
                lines.append(f"{i}. {name} — {origin}/{src} ({url})")
            else:
                lines.append(f"{i}. {name} — {origin}/{src}")
        return "\n".join(lines)

    def _rag_summary_block(self, agent_outputs: dict, language: str = "en") -> str:
        rag = agent_outputs.get("advanced_rag") or {}
        if not rag:
            return ""
        tools = ", ".join(rag.get("tools_used") or []) or "none"
        top = rag.get("top_documents") or []
        if language == "mr":
            parts = [
                self.MARATHI_TEMPLATES["rag_prefix"],
                f"- साधने: {tools}",
                f"- वेब निकाल: {rag.get('web_result_count', 0)} | स्थानिक hits: {rag.get('local_hit_count', 0)}",
            ]
        else:
            parts = [
                "### Advanced Multi-Source RAG",
                f"- Tools: {tools}",
                f"- Web results: {rag.get('web_result_count', 0)} | Local hits: {rag.get('local_hit_count', 0)}",
            ]
        for d in top[:4]:
            parts.append(f"  • [{d.get('origin')}] {d.get('title')} (score={d.get('score')})")
        return "\n".join(parts)

    def _synthesize_marathi(self, plan_summary: str, agent_outputs: dict, citations: list | None = None) -> str:
        parts = [self.MARATHI_TEMPLATES["greeting"], f"\n**विश्लेषण निष्कर्ष:** {plan_summary}\n"]

        rag_block = self._rag_summary_block(agent_outputs, language="mr")
        if rag_block:
            parts.append(rag_block)
            parts.append("")

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

        cite = self._format_citations(citations or (agent_outputs.get("advanced_rag") or {}).get("citations"), language="mr")
        if cite:
            parts.append(cite)
            parts.append("")

        parts.append(self.MARATHI_TEMPLATES["footer"])
        return "\n".join(parts)

    def _synthesize_english(self, plan_summary: str, agent_outputs: dict, citations: list | None = None) -> str:
        parts = ["Greetings! Welcome to AI Krushi Mitra Agriculture Platform.", f"\n**Plan Summary:** {plan_summary}\n"]

        rag_block = self._rag_summary_block(agent_outputs, language="en")
        if rag_block:
            parts.append(rag_block)
            parts.append("")

        for key, val in agent_outputs.items():
            if key == "advanced_rag":
                continue
            parts.append(f"### {key.capitalize()} Advisory")
            # Prefer human-readable summaries over raw dumps when possible
            if isinstance(val, dict):
                summary_keys = ("summary_mr", "summary_en", "guidance_mr", "guidance_en", "message")
                shown = False
                for sk in summary_keys:
                    if val.get(sk):
                        parts.append(str(val[sk]))
                        shown = True
                        break
                if not shown:
                    # Compact key highlights
                    for k, v in list(val.items())[:8]:
                        if k in ("agent", "raw", "records") or isinstance(v, (list, dict)) and k not in (
                            "recommended_fertilizer_bags",
                            "net_nutrient_requirement_kg",
                            "extracted_parameters",
                            "evaluations",
                            "organic_treatment",
                            "chemical_treatment",
                        ):
                            if isinstance(v, (list, dict)) and k not in (
                                "recommended_fertilizer_bags",
                                "net_nutrient_requirement_kg",
                                "extracted_parameters",
                                "evaluations",
                                "organic_treatment",
                                "chemical_treatment",
                            ):
                                continue
                        parts.append(f"- **{k}:** {v}")
            else:
                parts.append(str(val))
            parts.append("")

        cite = self._format_citations(citations or (agent_outputs.get("advanced_rag") or {}).get("citations"), language="en")
        if cite:
            parts.append(cite)

        return "\n".join(parts)

response_synthesizer = MarathiResponseSynthesizer()
