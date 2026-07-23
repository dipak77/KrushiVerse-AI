"""W-RAG wrap: platform advanced_rag + local Mini context packs (Sprint 15)."""

from __future__ import annotations

from typing import Any

from mini.eval.gold_sets import load_all_gold
from mini.inference.context import normalize_sources


def _local_mini_hits(query: str, *, top_k: int = 2) -> list[dict[str, Any]]:
    """Offline lexical retrieval over curated gold + simple agri facts."""
    q = (query or "").lower()
    q_tokens = set(q.replace("?", " ").split())
    rows = load_all_gold()
    # extra evergreen facts for cotton demo
    extra = [
        {
            "id": "mini-fact-cotton-pbw",
            "category": "disease",
            "crop": "Cotton",
            "question": "pink bollworm cotton IPM",
            "answer": (
                "Pink bollworm on cotton: monitor with pheromone traps, scout weekly, "
                "use ETL thresholds, prefer resistant hybrids, apply labeled insecticide only when needed."
            ),
            "source": "mini_local_kb",
        },
        {
            "id": "mini-fact-cotton-general",
            "category": "crop",
            "crop": "Cotton",
            "question": "cotton cultivation Maharashtra Vidarbha",
            "answer": (
                "Cotton is a major cash crop in Vidarbha, Maharashtra. Manage pests with IPM, "
                "irrigate by soil moisture, and base fertilizer on soil test."
            ),
            "source": "mini_local_kb",
        },
    ]
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in list(rows) + extra:
        blob = " ".join(
            [
                str(r.get("question") or ""),
                str(r.get("answer") or ""),
                str(r.get("crop") or ""),
                str(r.get("category") or ""),
            ]
        ).lower()
        tokens = set(blob.split())
        inter = len(q_tokens & tokens) if q_tokens else 0
        # substring boosts
        boost = 0.0
        for t in ("cotton", "bollworm", "disease", "pink", "ipm", "vidarbha", "pesticide"):
            if t in q and t in blob:
                boost += 0.5
        score = inter + boost
        if score <= 0 and any(t in blob for t in list(q_tokens)[:6]):
            score = 0.2
        if score > 0:
            scored.append(
                (
                    score,
                    {
                        "id": r.get("id") or "local",
                        "title": (r.get("question") or r.get("id") or "local")[:120],
                        "text": r.get("answer") or "",
                        "origin": "mini_local",
                        "source": r.get("source") or "mini_local",
                        "score": score,
                        "category": r.get("category"),
                        "crop": r.get("crop"),
                    },
                )
            )
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def retrieve_for_mini(
    query: str,
    *,
    crop: str | None = None,
    location: str = "Pune",
    top_k: int = 6,
    enable_web: bool = False,
    enable_tools: bool = True,
    use_platform_rag: bool = True,
) -> dict[str, Any]:
    """Retrieve multi-source context; always merge mini-local hits for reliability."""
    plan_dict: dict[str, Any] = {}
    platform: dict[str, Any] = {}
    citations: list[dict[str, Any]] = []
    fused: list[dict[str, Any]] = []
    context_text = ""
    backends = ["mini_local"]

    try:
        from app.knowledge.query_understanding import query_understanding

        plan = query_understanding.understand(query, default_crop=crop)
        plan_dict = {
            "crops": plan.crops,
            "intents": plan.intents,
            "expanded_queries": plan.expanded_queries,
            "needs_web": plan.needs_web,
            "needs_live_tools": plan.needs_live_tools,
            "language_hint": plan.language_hint,
            "categories": getattr(plan, "categories", []),
        }
        crop = crop or (plan.crops[0] if plan.crops else None)
    except Exception as e:
        plan_dict = {"error": str(e), "crops": [crop] if crop else [], "intents": ["general"]}

    if use_platform_rag:
        try:
            from app.knowledge.advanced_rag import advanced_rag

            platform = advanced_rag.retrieve(
                query,
                crop=crop,
                location=location,
                top_k=top_k,
                enable_web=enable_web,
                enable_tools=enable_tools,
            )
            backends.append("advanced_rag")
            context_text = str(platform.get("context_text") or "")
            citations = list(platform.get("citations") or [])
            fused = list(platform.get("fused_documents") or [])
            if not plan_dict.get("crops"):
                plan_dict = platform.get("query_plan") or plan_dict
        except Exception as e:
            platform = {"error": str(e)}

    local = _local_mini_hits(query, top_k=top_k)
    # Prefer structured docs for context builder
    docs: list[dict[str, Any]] = []
    for d in fused:
        if not isinstance(d, dict):
            continue
        docs.append(
            {
                "id": d.get("id") or d.get("doc_id"),
                "title": d.get("title") or d.get("name") or d.get("id"),
                "text": d.get("text") or d.get("snippet") or d.get("content") or "",
                "origin": d.get("origin") or d.get("source") or "fused",
                "url": d.get("url"),
                "score": d.get("score"),
            }
        )
    # citations often lack full text — keep as secondary
    for c in citations:
        if isinstance(c, dict):
            docs.append(
                {
                    "id": c.get("id") or c.get("source"),
                    "title": c.get("title") or c.get("source") or c.get("origin"),
                    "text": c.get("snippet") or c.get("text") or "",
                    "origin": c.get("origin") or c.get("source") or "citation",
                    "url": c.get("url"),
                    "score": c.get("score"),
                }
            )
    docs.extend(local)

    # de-dupe by title+origin
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for d in docs:
        key = f"{d.get('title')}|{d.get('origin')}|{(d.get('text') or '')[:40]}"
        if key in seen:
            continue
        seen.add(key)
        if (d.get("text") or d.get("title")):
            uniq.append(d)

    sources = normalize_sources(uniq[: max(top_k, 4)])
    return {
        "query": query,
        "query_plan": plan_dict,
        "backends": backends,
        "platform": {
            "local_hit_count": platform.get("local_hit_count"),
            "tools_used": platform.get("tools_used"),
            "retrieval_mode": platform.get("retrieval_mode"),
            "error": platform.get("error"),
        },
        "sources": sources,
        "context_text_raw": context_text,
        "n_sources": len(sources),
        "has_sources": len(sources) > 0,
    }
