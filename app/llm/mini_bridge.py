"""Bridge platform synthesizer / API to Mini inference chain (Sprint 16 / FP-9)."""

from __future__ import annotations

from typing import Any

from app.config import settings


def agent_outputs_to_notes(agent_outputs: dict[str, Any] | None, *, max_notes: int = 8) -> list[str]:
    """Flatten specialist agent outputs into short context-channel notes for Mini."""
    notes: list[str] = []
    if not agent_outputs:
        return notes
    for key, val in agent_outputs.items():
        if key in {"advanced_rag"}:
            continue
        if isinstance(val, dict):
            for sk in (
                "summary_en",
                "summary_mr",
                "guidance_en",
                "guidance_mr",
                "message",
                "disease_identified_en",
                "disease_identified_mr",
            ):
                if val.get(sk):
                    notes.append(f"{key}: {val[sk]}")
                    break
            else:
                # compact one-liner
                diag = val.get("diagnosis")
                if isinstance(diag, dict):
                    label = diag.get("disease_identified_en") or diag.get("label") or diag.get("predicted_class")
                    if label:
                        notes.append(f"{key}: {label}")
                elif val.get("agent"):
                    notes.append(f"{key}: agent={val.get('agent')}")
        elif isinstance(val, list) and val:
            notes.append(f"{key}: {str(val[0])[:160]}")
        elif val is not None:
            notes.append(f"{key}: {str(val)[:160]}")
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

    mode = mode or settings.MINI_DEFAULT_MODE or "grounded"
    enable_web = settings.ENABLE_WEB_RAG if enable_web is None else enable_web
    report = run_infer(
        query=query,
        dry_run=False,
        mode=mode,
        crop=crop,
        location=location,
        version=version or settings.MINI_MODEL_VERSION or "auto",
        enable_web=bool(enable_web),
        enable_tools=settings.ENABLE_TOOL_RAG,
        enable_agents=enable_agents,
        use_platform_rag=use_platform_rag,
        max_new_tokens=int(max_new_tokens or settings.MINI_MAX_NEW_TOKENS or 40),
        seed=seed,
    )

    # Optional: prepend agent channel notes into answer footer (not regenerating)
    agent_notes = agent_outputs_to_notes(agent_outputs)
    answer = report.get("answer") or ""
    if agent_notes and "Agent channels" not in answer:
        answer = answer.rstrip() + "\n\n**Agent channels:**\n" + "\n".join(f"- {n}" for n in agent_notes[:5])

    # Language hint: pipeline may return EN; keep answer as-is (Mini is multi-lang weak)
    return {
        "ok": bool(report.get("ok")),
        "query": query,
        "language": language,
        "mode": mode,
        "answer": answer,
        "synthesized_answer": answer,
        "engine": report.get("engine") or "mini",
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
