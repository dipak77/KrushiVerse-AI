"""W-RAG — platform advanced RAG wrap for Mini context packs (Sprint 15)."""

from __future__ import annotations

from typing import Any

from mini.contracts import WorkerResult
from mini.inference.context import build_context_pack
from mini.inference.rag_wrap import retrieve_for_mini
from mini.workers.base import BaseWorker, register_worker


@register_worker
class RAGWorker(BaseWorker):
    worker_id = "W-RAG"
    name = "RAG Retriever"
    description = "Wrap platform advanced_rag + mini-local for Mini context packs (S15)"
    epic = "E6"
    status = "ready"

    def execute(self, *, dry_run: bool = False, **kwargs: Any) -> WorkerResult:
        query = str(
            kwargs.get("query")
            or "How do I manage pink bollworm in cotton with IPM in Maharashtra?"
        )
        if dry_run:
            return WorkerResult(
                worker_id=self.worker_id,
                ok=True,
                dry_run=True,
                message=f"W-RAG dry-run query={query[:80]}",
                metrics={"sprint": "S15", "dry_run": True, "query": query},
            )

        rag = retrieve_for_mini(
            query,
            crop=kwargs.get("crop"),
            location=str(kwargs.get("location") or "Pune"),
            top_k=int(kwargs.get("top_k") or 6),
            enable_web=bool(kwargs.get("enable_web", False)),
            enable_tools=bool(kwargs.get("enable_tools", True)),
            use_platform_rag=bool(kwargs.get("use_platform_rag", True)),
        )
        pack = build_context_pack(query=query, sources=rag.get("sources") or [])
        report = {
            "sprint": "S15",
            "feature_phase": "FP-8",
            "query": query,
            "query_plan": rag.get("query_plan"),
            "backends": rag.get("backends"),
            "n_sources": pack.get("n_sources"),
            "has_sources": pack.get("has_sources"),
            "citations": pack.get("citations"),
            "context_preview": (pack.get("context_text") or "")[:400],
            "platform": rag.get("platform"),
            "ok": bool(pack.get("has_sources")),
        }
        return WorkerResult(
            worker_id=self.worker_id,
            ok=bool(report["ok"]),
            dry_run=False,
            message=(
                f"RAG n_sources={report['n_sources']} backends={report['backends']} "
                f"intents={((report.get('query_plan') or {}).get('intents'))}"
            ),
            metrics=report,
            errors=[] if report["ok"] else ["No sources retrieved"],
        )
