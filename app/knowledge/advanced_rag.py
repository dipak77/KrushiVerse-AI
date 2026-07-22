"""
Advanced multi-source RAG orchestrator.

Pipeline:
  1. Query understanding (entities, intents, multi-query expansion)
  2. Local hybrid retrieval (BM25 + TF-IDF vector + RRF) across expanded open KB
  3. GraphRAG multi-hop ecosystem context
  4. External tools (weather, market, schemes, crop PoP, catalog)
  5. Web search (DuckDuckGo + Wikipedia + open-source fallbacks)
  6. Cross-source Reciprocal Rank Fusion + source attribution
"""

from __future__ import annotations

from typing import Any

from app.knowledge.hybrid_search import hybrid_retriever
from app.knowledge.graph_rag import graph_rag
from app.knowledge.query_understanding import query_understanding, QueryPlan
from app.knowledge.tools.registry import tool_registry
from app.knowledge.web_search import web_search_provider
from app.knowledge.dataset_loader import kb_loader
from app.knowledge.embeddings import embedding_provider
from app.live_feeds.opendata_client import opendata_client


class AdvancedRAG:
    """Generation-10 style multi-source retrieval for agriculture."""

    SOURCE_WEIGHTS = {
        "local_hybrid": 1.0,
        "graph": 0.9,
        "tool": 0.95,
        "web": 0.85,
    }

    def retrieve(
        self,
        query: str,
        *,
        farm_id: str | None = None,
        crop: str | None = None,
        location: str = "Pune",
        top_k: int = 8,
        enable_web: bool = True,
        enable_tools: bool = True,
        force_web: bool = False,
    ) -> dict[str, Any]:
        plan = query_understanding.understand(query, default_crop=crop)
        if force_web:
            plan.needs_web = True
            plan.needs_live_tools = True

        local_hits = self._local_multi_query(plan, top_k=top_k)
        graph_pack = self._graph_context(plan)
        tool_pack = self._run_tools(plan, location=location, enable=enable_tools)
        web_hits: list[dict] = []
        if enable_web and (plan.needs_web or force_web):
            web_query = plan.expanded_queries[0] if plan.expanded_queries else plan.normalized
            web_hits = web_search_provider.search(
                f"{web_query} agriculture India Maharashtra",
                max_results=5,
            )

        fused = self._fuse_sources(
            plan=plan,
            local_hits=local_hits,
            graph_pack=graph_pack,
            tool_docs=tool_pack.get("documents", []),
            web_hits=web_hits,
            top_k=top_k,
        )

        context_blocks = self._build_context_blocks(fused)
        citations = self._citations(fused, web_hits, tool_pack, query=query)

        return {
            "query": query,
            "query_plan": {
                "crops": plan.crops,
                "intents": plan.intents,
                "expanded_queries": plan.expanded_queries,
                "needs_web": plan.needs_web,
                "needs_live_tools": plan.needs_live_tools,
                "language_hint": plan.language_hint,
            },
            "knowledge_stats": kb_loader.knowledge_stats(),
            "retrieval_backends": {
                "hybrid": hybrid_retriever.backend_info(),
                "embeddings": embedding_provider.info(),
                "opendata": opendata_client.status(),
            },
            "local_hit_count": len(local_hits),
            "graph": graph_pack,
            "tools_used": tool_pack.get("tools_used", []),
            "tool_results": tool_pack.get("results", []),
            "web_results": web_hits,
            "fused_documents": fused,
            "context_text": "\n\n".join(context_blocks),
            "citations": citations,
            "retrieval_mode": "advanced_multi_source_rag_v10_2",
        }

    def _local_multi_query(self, plan: QueryPlan, top_k: int) -> list[dict]:
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}
        rrf_k = 60

        target_crop = plan.crops[0] if (plan and plan.crops) else None
        from mini.taxonomy.aliases import resolve_crops_smart

        for qi, q in enumerate(plan.expanded_queries or [plan.normalized]):
            hits = hybrid_retriever.hybrid_search(q, top_k=top_k * 2)
            for rank, hit in enumerate(hits):
                doc = hit["doc"]
                doc_id = doc["id"]
                doc_map[doc_id] = doc
                # slight boost for earlier expanded queries
                weight = 1.0 / (1.0 + 0.15 * qi)

                if target_crop:
                    d_title = str(doc.get("title") or "")
                    d_content = str(doc.get("content") or "")
                    d_crop = str(doc.get("crop") or "")
                    doc_text = f"{d_title} {d_crop} {d_content[:150]}"
                    doc_crops = resolve_crops_smart(doc_text)

                    if doc_crops and target_crop not in doc_crops:
                        weight *= 0.02  # Heavy penalty for wrong crop documents
                    elif target_crop in doc_crops:
                        weight *= 3.0   # Boost target crop documents

                # Intent-Category matching boost
                doc_cat = (str(doc.get("category") or "")).lower()
                doc_title = (str(doc.get("title") or "")).lower()
                intents = [i.lower() for i in plan.intents] if (plan and plan.intents) else []
                if "disease" in intents or "pest" in intents:
                    if doc_cat in ("disease", "pest") or any(k in doc_title for k in ("disease", "blight", "pest", "करपा", "तेल्या", "रोग", "कीड")):
                        weight *= 3.5
                elif "fertilizer" in intents or "soil" in intents:
                    if doc_cat in ("fertilizer", "soil") or "खत" in doc_title or "fertilizer" in doc_title:
                        weight *= 3.0
                elif "irrigation" in intents:
                    if doc_cat in ("irrigation", "water") or any(k in doc_title for k in ("irrigation", "drip", "ठिबक", "सिंचन", "पाणी")):
                        weight *= 3.5
                elif "market" in intents:
                    if doc_cat in ("market", "price") or "मंडी" in doc_title or "भाव" in doc_title:
                        weight *= 2.5
                elif "scheme" in intents:
                    if doc_cat in ("scheme", "subsidy") or "योजना" in doc_title or "subsidy" in doc_title:
                        weight *= 2.5

                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + weight * (
                    1.0 / (rrf_k + rank + 1)
                )

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        out = []
        for doc_id, score in ranked[: top_k * 2]:
            out.append({
                "rrf_score": score,
                "doc": doc_map[doc_id],
                "origin": "local_hybrid",
            })
        return out

    def _graph_context(self, plan: QueryPlan) -> dict:
        ecosystems = []
        for crop in plan.crops[:3]:
            eco = graph_rag.get_crop_ecosystem(crop)
            if "error" not in eco:
                ecosystems.append(eco)
            entity = graph_rag.query_graph_for_entity(crop)
            if entity.get("found"):
                ecosystems.append({"entity_view": entity})
        return {"crop_ecosystems": ecosystems, "crop_count": len(plan.crops)}

    def _run_tools(self, plan: QueryPlan, location: str, enable: bool) -> dict:
        if not enable:
            return {"tools_used": [], "results": [], "documents": []}

        crop = plan.crops[0] if plan.crops else None
        names = tool_registry.route_for_intents(
            intents=plan.intents,
            crops=plan.crops,
            location=location,
            query=plan.normalized,
        )
        results = []
        documents = []
        for name in names:
            res = tool_registry.run(
                name,
                {
                    "query": plan.normalized,
                    "crop": crop,
                    "location": location,
                    "max_results": 4,
                },
            )
            results.append({"tool": name, "ok": res.get("ok"), "error": res.get("error")})
            for d in res.get("documents") or []:
                d = dict(d)
                d["_origin"] = "tool"
                d["_tool"] = name
                documents.append(d)
        return {"tools_used": names, "results": results, "documents": documents}

    def _fuse_sources(
        self,
        plan: QueryPlan,
        local_hits: list[dict],
        graph_pack: dict,
        tool_docs: list[dict],
        web_hits: list[dict],
        top_k: int,
    ) -> list[dict]:
        rrf_k = 60
        scores: dict[str, float] = {}
        store: dict[str, dict] = {}

        target_crop = plan.crops[0].lower() if (plan and plan.crops) else None

        def add(doc_id: str, rank: int, weight: float, payload: dict):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * (1.0 / (rrf_k + rank + 1))
            if doc_id not in store:
                store[doc_id] = payload

        known_crops = ["rice", "grape", "maize", "millet", "cotton", "soybean", "pomegranate", "wheat", "sugarcane"]

        for rank, hit in enumerate(local_hits):
            doc = hit["doc"]
            w = self.SOURCE_WEIGHTS["local_hybrid"]
            if target_crop:
                text_lower = (str(doc.get("title") or "") + " " + str(doc.get("content") or "")).lower()
                doc_crops = [c for c in known_crops if c in text_lower]
                if doc_crops and target_crop not in doc_crops:
                    w *= 0.05  # Heavy penalty for wrong crop documents
                elif target_crop in text_lower:
                    w *= 2.5   # Boost target crop documents

            add(
                doc["id"],
                rank,
                w,
                {
                    "id": doc["id"],
                    "title": doc.get("title"),
                    "content": doc.get("content"),
                    "category": doc.get("category"),
                    "source": doc.get("source", "local_kb"),
                    "origin": "local_hybrid",
                    "metadata": doc.get("metadata", {}),
                },
            )

        # Graph neighbors as synthetic docs
        g_rank = 0
        for eco in graph_pack.get("crop_ecosystems") or []:
            if "crop" in eco:
                text = (
                    f"GraphRAG ecosystem for {eco.get('crop')}: "
                    f"pests={eco.get('pests_and_diseases')}, soils={eco.get('soil_types')}, "
                    f"fertilizers={eco.get('recommended_fertilizers')}, schemes={eco.get('applicable_schemes')}"
                )
                doc_id = f"graph_{eco.get('crop')}"
                add(
                    doc_id,
                    g_rank,
                    self.SOURCE_WEIGHTS["graph"],
                    {
                        "id": doc_id,
                        "title": f"GraphRAG: {eco.get('crop')}",
                        "content": text,
                        "category": "GraphRAG",
                        "source": "knowledge_graph",
                        "origin": "graph",
                        "metadata": eco,
                    },
                )
                g_rank += 1

        for rank, doc in enumerate(tool_docs):
            doc_id = doc.get("id") or f"tool_doc_{rank}"
            add(
                doc_id,
                rank,
                self.SOURCE_WEIGHTS["tool"],
                {
                    "id": doc_id,
                    "title": doc.get("title"),
                    "content": doc.get("content"),
                    "category": doc.get("category", "Tool"),
                    "source": doc.get("source", "tool"),
                    "origin": "tool",
                    "metadata": doc.get("metadata", {}),
                    "tool": doc.get("_tool"),
                },
            )

        web_docs = web_search_provider.to_rag_docs(web_hits)
        for rank, doc in enumerate(web_docs):
            add(
                doc["id"],
                rank,
                self.SOURCE_WEIGHTS["web"],
                {
                    "id": doc["id"],
                    "title": doc.get("title"),
                    "content": doc.get("content"),
                    "category": "Web",
                    "source": doc.get("source", "web"),
                    "origin": "web",
                    "url": doc.get("url", ""),
                    "metadata": doc.get("metadata", {}),
                },
            )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused = []
        for doc_id, score in ranked[:top_k]:
            item = dict(store[doc_id])
            item["fusion_score"] = round(score, 6)
            fused.append(item)
        return fused

    def _build_context_blocks(self, fused: list[dict]) -> list[str]:
        blocks = []
        for i, d in enumerate(fused, 1):
            blocks.append(
                f"[{i}] ({d.get('origin')}/{d.get('category')}) {d.get('title')}\n"
                f"{d.get('content')}\n"
                f"source={d.get('source')} score={d.get('fusion_score')}"
            )
        return blocks

    def _citations(self, fused: list[dict], web_hits: list[dict], tool_pack: dict, query: str = "") -> list[dict]:
        from mini.taxonomy.aliases import resolve_crops_smart

        query_crops = resolve_crops_smart(query) if query else []
        query_crop_canon = query_crops[0] if query_crops else None

        cites = []
        seen = set()

        for d in fused:
            title = (d.get("title") or "").lower()
            d_id = (d.get("id") or "").lower()
            d_crop = (d.get("crop") or "").lower()

            # Skip all stage checklists & GraphRAG node headers unconditionally
            if "checklist" in title or "checklist" in d_id or title.startswith("graphrag:"):
                continue

            # Smart crop match verification
            if query_crop_canon:
                doc_text = f"{title} {d_crop}"
                doc_crops = resolve_crops_smart(doc_text)
                if doc_crops and query_crop_canon not in doc_crops:
                    continue

            key = title[:40]
            if key in seen:
                continue
            seen.add(key)

            cites.append({
                "title": d.get("title"),
                "origin": d.get("origin"),
                "source": d.get("source"),
                "url": d.get("url") or (d.get("metadata") or {}).get("url") or (d.get("metadata") or {}).get("portal"),
                "score": d.get("fusion_score"),
            })
            if len(cites) >= 2:
                break

        return cites[:2] if cites else [{
            "title": d.get("title"),
            "origin": d.get("origin"),
            "source": d.get("source"),
            "url": d.get("url") or (d.get("metadata") or {}).get("url"),
            "score": d.get("fusion_score"),
        } for d in fused if "checklist" not in (d.get("title") or "").lower()][:1]


advanced_rag = AdvancedRAG()
