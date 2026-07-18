"""Dense vector store: Qdrant when configured, otherwise local numpy index with disk cache."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from app.config import settings
from app.knowledge.embeddings import embedding_provider


class LocalDenseVectorStore:
    """In-process dense cosine search with optional disk cache under .cache/."""

    def __init__(self):
        self.documents: list[dict] = []
        self.vectors: np.ndarray | None = None
        self.backend = "local_dense"
        self.dim = embedding_provider.dim

    def build_index(self, docs: list[dict], force: bool = False):
        self.documents = docs
        cache_path = os.path.join(settings.CACHE_DIR, "dense_index.npz")
        meta_path = os.path.join(settings.CACHE_DIR, "dense_index_meta.json")
        doc_ids = [d.get("id") for d in docs]

        if not force and os.path.exists(cache_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if (
                    meta.get("doc_ids") == doc_ids
                    and meta.get("backend") == embedding_provider.backend
                    and meta.get("dim") == embedding_provider.dim
                ):
                    data = np.load(cache_path)
                    self.vectors = data["vectors"]
                    self.dim = int(self.vectors.shape[1]) if self.vectors is not None else embedding_provider.dim
                    return
            except Exception:
                pass

        texts = [f"{d.get('title', '')} {d.get('content', '')}" for d in docs]
        if texts:
            self.vectors = embedding_provider.embed_batch(texts)
            self.dim = int(self.vectors.shape[1])
        else:
            self.vectors = np.zeros((0, embedding_provider.dim), dtype=np.float32)

        try:
            os.makedirs(settings.CACHE_DIR, exist_ok=True)
            np.savez_compressed(cache_path, vectors=self.vectors)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "doc_ids": doc_ids,
                        "backend": embedding_provider.backend,
                        "dim": self.dim,
                        "count": len(docs),
                    },
                    f,
                )
        except Exception:
            pass

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.vectors is None or len(self.documents) == 0:
            return []
        q = embedding_provider.embed_text(query)
        # cosine since vectors are normalized
        scores = self.vectors @ q
        if top_k >= len(scores):
            idxs = np.argsort(-scores)
        else:
            idxs = np.argpartition(-scores, top_k)[:top_k]
            idxs = idxs[np.argsort(-scores[idxs])]
        results = []
        for i in idxs[:top_k]:
            score = float(scores[int(i)])
            if score > 0.01:
                results.append({"score": score, "doc": self.documents[int(i)], "origin": "dense"})
        return results


class QdrantVectorStore:
    """Qdrant-backed dense retrieval; falls back to LocalDenseVectorStore on failure."""

    def __init__(self):
        self.documents: list[dict] = []
        self.backend = "qdrant"
        self.dim = embedding_provider.dim
        self._client = None
        self._local = LocalDenseVectorStore()
        self._use_local = True
        self._init_client()

    def _init_client(self):
        if not settings.QDRANT_URL:
            self.backend = "local_dense"
            return
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qm

            self._qm = qm
            kwargs: dict[str, Any] = {"url": settings.QDRANT_URL, "timeout": 10}
            if settings.QDRANT_API_KEY:
                kwargs["api_key"] = settings.QDRANT_API_KEY
            self._client = QdrantClient(**kwargs)
            # lightweight connectivity check
            self._client.get_collections()
            self._use_local = False
            self.backend = "qdrant"
        except Exception:
            self._client = None
            self._use_local = True
            self.backend = "local_dense_fallback"

    def build_index(self, docs: list[dict], force: bool = False):
        self.documents = docs
        # Always keep local mirror for offline resilience
        self._local.build_index(docs, force=force)
        self.dim = self._local.dim

        if self._use_local or self._client is None:
            return

        try:
            from qdrant_client.http import models as qm

            collection = settings.QDRANT_COLLECTION
            exists = False
            try:
                info = self._client.get_collection(collection)
                exists = True
                current_dim = info.config.params.vectors.size
                if current_dim != self.dim or settings.QDRANT_RECREATE or force:
                    self._client.delete_collection(collection)
                    exists = False
            except Exception:
                exists = False

            if not exists:
                self._client.create_collection(
                    collection_name=collection,
                    vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
                )

            points = []
            vectors = self._local.vectors
            for i, doc in enumerate(docs):
                payload = {
                    "doc_id": doc.get("id"),
                    "title": doc.get("title"),
                    "category": doc.get("category"),
                    "source": doc.get("source"),
                    "content": (doc.get("content") or "")[:2000],
                }
                points.append(
                    qm.PointStruct(
                        id=self._stable_id(doc.get("id", str(i))),
                        vector=vectors[i].tolist(),
                        payload=payload,
                    )
                )
            # upsert in batches
            batch = 64
            for start in range(0, len(points), batch):
                self._client.upsert(collection_name=collection, points=points[start : start + batch])
            self.backend = "qdrant"
        except Exception:
            self._use_local = True
            self.backend = "local_dense_fallback"

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._use_local or self._client is None:
            return self._local.search(query, top_k=top_k)

        try:
            qvec = embedding_provider.embed_text(query).tolist()
            hits = self._client.search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=qvec,
                limit=top_k,
                with_payload=True,
            )
            # map payloads back to full docs when possible
            by_id = {d.get("id"): d for d in self.documents}
            results = []
            for h in hits:
                payload = h.payload or {}
                doc_id = payload.get("doc_id")
                doc = by_id.get(doc_id) or {
                    "id": doc_id,
                    "title": payload.get("title"),
                    "content": payload.get("content"),
                    "category": payload.get("category"),
                    "source": payload.get("source"),
                }
                results.append({"score": float(h.score), "doc": doc, "origin": "qdrant"})
            return results
        except Exception:
            return self._local.search(query, top_k=top_k)

    @staticmethod
    def _stable_id(doc_id: str) -> int:
        # Qdrant accepts unsigned int ids; hash string to 63-bit positive int
        h = hashlib_md5_int(doc_id)
        return h


def hashlib_md5_int(s: str) -> int:
    import hashlib
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:15], 16)


def create_dense_store() -> LocalDenseVectorStore | QdrantVectorStore:
    if settings.QDRANT_URL:
        return QdrantVectorStore()
    return LocalDenseVectorStore()
