import re
from rank_bm25 import BM25Okapi
from app.knowledge.vector_store import SimpleVectorStore
from app.knowledge.dataset_loader import kb_loader

class HybridRAGSearch:
    """Hybrid Search combining BM25, Vector Search, and Reciprocal Rank Fusion (RRF)."""

    def __init__(self):
        self.docs = kb_loader.get_all_documents()
        self.vector_store = SimpleVectorStore()
        self.vector_store.build_index(self.docs)

        # BM25 Tokenizer setup
        self.corpus = [self._tokenize(doc["content"] + " " + doc["title"]) for doc in self.docs]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())

    def search_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.bm25:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0:
                results.append({"score": float(scores[idx]), "doc": self.docs[idx]})
        return results

    def search_vector(self, query: str, top_k: int = 5) -> list[dict]:
        return self.vector_store.search(query, top_k=top_k)

    def hybrid_search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list[dict]:
        """Combine BM25 and Vector Search using Reciprocal Rank Fusion (RRF)."""
        bm25_results = self.search_bm25(query, top_k=top_k * 2)
        vector_results = self.search_vector(query, top_k=top_k * 2)

        rrf_scores = {}
        doc_map = {}

        for rank, res in enumerate(bm25_results):
            doc_id = res["doc"]["id"]
            doc_map[doc_id] = res["doc"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, res in enumerate(vector_results):
            doc_id = res["doc"]["id"]
            doc_map[doc_id] = res["doc"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, score in sorted_docs[:top_k]:
            final_results.append({
                "rrf_score": score,
                "doc": doc_map[doc_id]
            })

        return final_results

hybrid_retriever = HybridRAGSearch()
