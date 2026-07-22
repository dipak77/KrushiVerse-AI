"""Knowledge worker for pipeline integration.

Runs knowledge_audit and knowledge_adder_agent to ensure 100% crop disease coverage,
and rebuilds hybrid vector/BM25 search indexes.
"""

import os
import json
from app.knowledge.dataset_loader import kb_loader
from app.knowledge.hybrid_search import hybrid_retriever
from mini.eval.knowledge_audit import run_audit


class KnowledgeWorker:
    """Worker class managing automated knowledge audit and RAG index rebuilding."""

    def sync_and_audit(self) -> dict:
        report = run_audit()
        coverage = report.get("summary", {}).get("overall_disease_coverage_pct", 0)

        if coverage < 100.0:
            from app.agents.knowledge_adder_agent import populate_knowledge
            populate_knowledge()
            report = run_audit()

        # Re-initialize hybrid search index with new documents
        hybrid_retriever.__init__()
        return report


knowledge_worker = KnowledgeWorker()
