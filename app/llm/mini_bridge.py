"""Bridge platform synthesizer / API to Mini inference chain (Sprint 16 / FP-9)."""

from __future__ import annotations

from typing import Any

from app.config import settings


def agent_outputs_to_notes(agent_outputs: dict[str, Any] | None, *, max_notes: int = 8) -> list[str]:
    """Flatten specialist agent outputs into clear advisory notes for LLM synthesis."""
    notes: list[str] = []
    if not agent_outputs:
        return notes
    for key, val in agent_outputs.items():
        if key in {"advanced_rag", "vision"}:
            continue
        if isinstance(val, dict):
            if key == "disease":
                d_name = val.get("disease_identified_mr") or val.get("disease_identified_en")
                if d_name:
                    notes.append(f"रोग निदान: {d_name}")
                sym = val.get("symptoms_mr") or val.get("symptoms_en")
                if sym:
                    notes.append(f"लक्षणे: {sym}")
                org_tx = (val.get("organic_treatment") or {}).get("mr") or (val.get("organic_treatment") or {}).get("en")
                if org_tx:
                    notes.append(f"सेंद्रिय उपाय: {org_tx}")
                chem_tx = (val.get("chemical_treatment") or {}).get("mr") or (val.get("chemical_treatment") or {}).get("en")
                if chem_tx:
                    notes.append(f"रासायनिक उपाय: {chem_tx}")
                continue

            if key == "weather":
                temp = val.get("temperature_c")
                rh = val.get("relative_humidity_pct")
                rain = val.get("rainfall_mm_24h")
                if temp is not None:
                    notes.append(f"हवामान: तापमान {temp}°C, आर्द्रता {rh}%, पाऊस {rain} मिमी")
                if val.get("weather_alerts"):
                    for alert in val["weather_alerts"]:
                        notes.append(f"हवामान इशारे: {alert}")
                continue

            if key == "fertilizer":
                sched = val.get("application_schedule_mr") or val.get("organic_manure_recommendation")
                if sched:
                    notes.append(f"खत व्यवस्थापन: {sched}")
                continue

            for sk in ("guidance_mr", "guidance_en", "summary_mr", "summary_en", "message"):
                if val.get(sk) and not str(val[sk]).startswith("agent="):
                    notes.append(f"{key}: {val[sk]}")
                    break
        if len(notes) >= max_notes:
            break
    return notes[:max_notes]


def run_mini_chat(
    query: str,
    *,
    language: str = "en",
    crop: str | None = None,
    location: str = "Pune",
    mode: str | None = None,
    enable_web: bool | None = None,
    enable_agents: bool = True,
    use_platform_rag: bool = True,
    agent_outputs: dict[str, Any] | None = None,
    max_new_tokens: int | None = None,
    seed: int = 42,
    version: str | None = None,
) -> dict[str, Any]:
    """Call Mini inference pipeline and shape a platform-friendly response."""
    from mini.inference.pipeline import run_infer

    mode = mode or settings.MINI_DEFAULT_MODE or "instruct"
    ver = version if (version and version != "auto") else (settings.MINI_MODEL_VERSION or "v0.4-agri-qa")
    max_tokens = int(max_new_tokens or settings.MINI_MAX_NEW_TOKENS or 256)
    if max_tokens < 200:
        max_tokens = 256

    enable_web = settings.ENABLE_WEB_RAG if enable_web is None else enable_web
    report = run_infer(
        query=query,
        dry_run=False,
        mode=mode,
        crop=crop,
        location=location,
        version=ver,
        enable_web=bool(enable_web),
        enable_tools=False,
        enable_agents=enable_agents,
        use_platform_rag=use_platform_rag,
        max_new_tokens=max_tokens,
        seed=seed,
        top_k=settings.RAG_TOP_K,
    )

    agent_notes = agent_outputs_to_notes(agent_outputs)
    answer = report.get("answer") or ""
    if agent_notes and len(answer) < 100:
        note_title = "🔬 **तज्ञ कृषी शिफारशी (Expert Agent Advisory):**" if language in ("mr", "marathi") else "🔬 **Expert Agent Advisory:**"
        answer = f"### 🩺 **{crop or 'कृषी'} सल्ला ({location})**\n\n{note_title}\n" + "\n".join(f"• {n}" for n in agent_notes[:4])

    # Language hint: pipeline may return EN; keep answer as-is (Mini is multi-lang weak)
    return {
        "ok": bool(report.get("ok")),
        "query": query,
        "language": language,
        "mode": mode,
        "answer": answer,
        "synthesized_answer": answer,
        "engine": "local_krushiverse_llm" if settings.USE_MINI_LLM else (report.get("engine") or "mini"),
        "model_variant": "v2-12M-fixed",
        "used_fallback": report.get("used_fallback"),
        "citations": report.get("citations") or [],
        "n_sources": report.get("n_sources") or 0,
        "query_plan": report.get("query_plan") or {},
        "validation": report.get("validation") or {},
        "retrieval": report.get("retrieval") or {},
        "generation": report.get("generation") or {},
        "mini_report": {
            "sprint": report.get("sprint"),
            "feature_phase": report.get("feature_phase"),
            "raw_mini": report.get("raw_mini"),
            "context_preview": report.get("context_preview"),
        },
        "use_mini_llm": True,
        "agent_notes": agent_notes,
    }


def synthesize_with_mini(
    query: str,
    *,
    plan_summary: str,
    agent_outputs: dict[str, Any],
    language: str = "en",
    crop: str | None = None,
    location: str = "Pune",
    enable_web: bool | None = None,
    citations: list | None = None,
) -> tuple[str, dict[str, Any]]:
    """Mini as planner synthesizer brain; agents remain tool specialists upstream."""
    chat = run_mini_chat(
        query,
        language=language,
        crop=crop,
        location=location,
        mode=settings.MINI_DEFAULT_MODE,
        enable_web=enable_web,
        enable_agents=True,
        use_platform_rag=True,
        agent_outputs=agent_outputs,
    )
    # Prefer Mini grounded answer; if it failed hard, fall back to empty to let caller use classic
    answer = chat.get("answer") or ""
    meta = {
        "synthesizer": "mini_llm",
        "use_mini_llm": True,
        "plan_summary": plan_summary,
        "mini": chat,
        "upstream_citations": citations or [],
    }
    return answer, meta
