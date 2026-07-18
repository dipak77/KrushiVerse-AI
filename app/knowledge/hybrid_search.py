import re
from rank_bm25 import BM25Okapi

from app.config import settings
from app.knowledge.vector_store import SimpleVectorStore
from app.knowledge.dataset_loader import kb_loader
from app.knowledge.qdrant_store import create_dense_store
from app.knowledge.embeddings import embedding_provider


class HybridRAGSearch:
    """Hybrid Search: BM25 + sparse TF-IDF + dense embeddings (Qdrant/local) fused with RRF."""

    def __init__(self):
        self.docs = kb_loader.get_all_documents()
        self.vector_store = SimpleVectorStore()
        self.vector_store.build_index(self.docs)

        self.dense_store = None
        self.dense_enabled = settings.ENABLE_DENSE_RAG
        if self.dense_enabled:
            try:
                self.dense_store = create_dense_store()
                self.dense_store.build_index(self.docs)
            except Exception:
                self.dense_store = None

        self.corpus = [self._tokenize(doc["content"] + " " + doc["title"]) for doc in self.docs]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def backend_info(self) -> dict:
        dense_backend = None
        if self.dense_store is not None:
            dense_backend = getattr(self.dense_store, "backend", "dense")
        return {
            "bm25": bool(self.bm25),
            "sparse_tfidf": True,
            "dense_enabled": bool(self.dense_store),
            "dense_backend": dense_backend,
            "embedding": embedding_provider.info(),
            "document_count": len(self.docs),
        }

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[\w\u0900-\u097F]+", text.lower())

    def search_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.bm25:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0:
                results.append({"score": float(scores[idx]), "doc": self.docs[idx], "origin": "bm25"})
        return results

    def search_vector(self, query: str, top_k: int = 5) -> list[dict]:
        hits = self.vector_store.search(query, top_k=top_k)
        for h in hits:
            h["origin"] = "sparse_tfidf"
        return hits

    def search_dense(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.dense_store:
            return []
        return self.dense_store.search(query, top_k=top_k)

    def hybrid_search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list[dict]:
        """Combine BM25, sparse TF-IDF, and dense embedding search via RRF."""
        bm25_results = self.search_bm25(query, top_k=top_k * 2)
        vector_results = self.search_vector(query, top_k=top_k * 2)
        dense_results = self.search_dense(query, top_k=top_k * 2) if self.dense_enabled else []

        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}
        origins: dict[str, list[str]] = {}

        def add_list(results: list[dict], weight: float = 1.0):
            for rank, res in enumerate(results):
                doc = res["doc"]
                doc_id = doc["id"]
                doc_map[doc_id] = doc
                origin = res.get("origin", "unknown")
                origins.setdefault(doc_id, [])
                if origin not in origins[doc_id]:
                    origins[doc_id].append(origin)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + weight * (1.0 / (rrf_k + rank + 1))

        add_list(bm25_results, weight=1.0)
        add_list(vector_results, weight=1.0)
        # Dense semantic channel slightly higher weight when available
        add_list(dense_results, weight=1.15)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, score in sorted_docs[:top_k]:
            final_results.append({
                "rrf_score": score,
                "doc": doc_map[doc_id],
                "origins": origins.get(doc_id, []),
            })

        return final_results

    def rebuild(self, force_dense: bool = False):
        self.docs = kb_loader.get_all_documents()
        self.vector_store.build_index(self.docs)
        self.corpus = [self._tokenize(doc["content"] + " " + doc["title"]) for doc in self.docs]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None
        if self.dense_enabled:
            if self.dense_store is None:
                self.dense_store = create_dense_store()
            self.dense_store.build_index(self.docs, force=force_dense)


hybrid_retriever = HybridRAGSearch()
