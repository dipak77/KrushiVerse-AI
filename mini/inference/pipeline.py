"""End-to-end Mini inference: intent → retrieve → generate → validate (Sprint 15 / FP-8)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from mini.inference.agents_wrap import collect_agent_notes
from mini.inference.context import build_context_pack, format_answer_with_citations
from mini.inference.fallback import template_synthesize
from mini.inference.generate_mini import mini_generate
from mini.inference.rag_wrap import retrieve_for_mini
from mini.inference.validate import validate_answer
from mini.paths import INFERENCE_DIR, ensure_lake_layout, relative_to_repo

INFER_LATEST = INFERENCE_DIR / "INFER_LATEST.json"


def run_infer(
    *,
    query: str | None = None,
    dry_run: bool = False,
    mode: str = "instruct",
    crop: str | None = None,
    location: str = "Pune",
    version: str = "v0.4-agri-qa",
    enable_web: bool = False,
    enable_tools: bool = False,
    enable_agents: bool = True,
    use_platform_rag: bool = True,
    max_new_tokens: int = 256,
    min_grounding: float = 0.02,  # fits 0.04 scores
    seed: int = 42,
    top_k: int = 4,
) -> dict[str, Any]:
    ensure_lake_layout()
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    q = (query or "How do I manage pink bollworm in cotton with IPM in Maharashtra?").strip()
    mode = (mode or "instruct").lower()

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "sprint": "S15",
            "feature_phase": "FP-8",
            "query": q,
            "mode": mode,
            "planned": {
                "version": version,
                "enable_web": enable_web,
                "enable_agents": enable_agents,
                "use_platform_rag": use_platform_rag,
            },
        }

    # 1) Retrieve
    rag = retrieve_for_mini(
        q,
        crop=crop,
        location=location,
        top_k=top_k,
        enable_web=enable_web,
        enable_tools=enable_tools,
        use_platform_rag=use_platform_rag,
    )
    plan = rag.get("query_plan") or {}
    crops = list(plan.get("crops") or ([] if not crop else [crop]))
    intents = list(plan.get("intents") or ["general"])
    lang = str(plan.get("language_hint") or "en")

    # 2) Optional agent notes
    agents = collect_agent_notes(
        q,
        intents=intents,
        crop=crops[0] if crops else crop,
        location=location,
        enable=enable_agents,
    )

    # 3) Context pack with citations
    pack = build_context_pack(
        query=q,
        sources=rag.get("sources") or [],
        agent_notes=agents.get("notes") or [],
    )

    # Grounded mode hard rule: no sources → no free-form answer
    if mode == "grounded" and not pack.get("has_sources"):
        refusal = (
            "I cannot answer in grounded mode without retrieved sources. "
            "Please refine the query or check the knowledge base."
        )
        out = {
            "ok": False,
            "dry_run": False,
            "sprint": "S15",
            "feature_phase": "FP-8",
            "query": q,
            "mode": mode,
            "answer": refusal,
            "engine": "refusal",
            "citations": [],
            "query_plan": plan,
            "retrieval": {
                "backends": rag.get("backends"),
                "n_sources": 0,
                "platform": rag.get("platform"),
            },
            "validation": {
                "ok": False,
                "reasons": ["no_sources"],
                "mode": mode,
            },
            "used_fallback": False,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        _write_latest(out)
        return out

    # 4) Mini generate
    gen = mini_generate(
        pack["user_prompt"],
        version=version,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    raw_answer = gen.get("text") or ""

    # 5) Validate
    validation = validate_answer(
        answer=raw_answer,
        context=pack.get("context_text") or "",
        citations=pack.get("citations") or [],
        mode=mode,
        min_grounding=min_grounding,
    )

    used_fallback = False
    engine = "mini_lm"
    final_answer = raw_answer
    if not validation.get("ok"):
        # 6) Template fallback
        fb = template_synthesize(
            query=q,
            intent=intents[0] if intents else "general",
            crops=crops,
            context_text=pack.get("context_text") or "",
            citations=pack.get("citations") or [],
            language=lang,
            reason=",".join(validation.get("reasons") or ["low_confidence"]),
            location=location,
            disease_info=agents.get("disease"),
            weather_info=agents.get("weather"),
        )
        final_answer = fb["answer"]
        engine = "template_fallback"
        used_fallback = True
        validation = validate_answer(
            answer=final_answer,
            context=pack.get("context_text") or "",
            citations=pack.get("citations") or [],
            mode=mode,
            min_grounding=min(min_grounding, 0.05),
        )
        # fallback with sources should pass grounded
        if pack.get("has_sources") and "banned_advice" not in (validation.get("reasons") or []):
            validation["ok"] = True
            validation["reasons"] = [r for r in (validation.get("reasons") or []) if r != "low_grounding"]

    final_answer = format_answer_with_citations(final_answer, pack.get("citations") or [])

    ok = bool(validation.get("ok")) and (mode != "grounded" or bool(pack.get("has_sources")))
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = {
        "ok": ok,
        "dry_run": False,
        "sprint": "S15",
        "feature_phase": "FP-8",
        "query": q,
        "mode": mode,
        "answer": final_answer,
        "raw_mini": raw_answer,
        "engine": engine,
        "used_fallback": used_fallback,
        "citations": pack.get("citations") or [],
        "n_sources": pack.get("n_sources") or 0,
        "query_plan": plan,
        "agents": {"notes": agents.get("notes"), "enabled": agents.get("enabled")},
        "retrieval": {
            "backends": rag.get("backends"),
            "n_sources": rag.get("n_sources"),
            "platform": rag.get("platform"),
        },
        "generation": {
            "latency_ms": gen.get("latency_ms"),
            "model_dir": gen.get("model_dir"),
            "load": gen.get("load"),
        },
        "validation": validation,
        "context_preview": (pack.get("context_text") or "")[:500],
        "created_at": created,
    }
    _write_latest(out)
    return out


def _write_latest(out: dict[str, Any]) -> None:
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    arts = [relative_to_repo(INFER_LATEST)]
    out["artifacts"] = arts
    INFER_LATEST.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
