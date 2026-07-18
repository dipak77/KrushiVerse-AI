"""Dense embedding providers for advanced RAG.

Backends (selected via EMBEDDING_BACKEND / auto):
  - hash:   fast local dense hashing (always available, numpy only)
  - minilm: sentence-transformers all-MiniLM-L6-v2 when installed
  - openai: OpenAI-compatible embeddings (OPENAI_API_KEY or XAI_API_KEY)
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from app.config import settings


class EmbeddingProvider:
    def __init__(self):
        self.backend = self._resolve_backend()
        self.dim = settings.EMBEDDING_DIM
        self._minilm = None
        self._openai_client = None

        if self.backend == "minilm":
            self._init_minilm()
        elif self.backend == "openai":
            self._init_openai()

    def _resolve_backend(self) -> str:
        requested = (settings.EMBEDDING_BACKEND or "auto").lower().strip()
        if requested != "auto":
            return requested
        # Prefer real neural model if present, else API keys, else hash
        try:
            import sentence_transformers  # noqa: F401
            return "minilm"
        except Exception:
            pass
        if settings.OPENAI_API_KEY or settings.XAI_API_KEY:
            return "openai"
        return "hash"

    def _init_minilm(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._minilm = SentenceTransformer(settings.EMBEDDING_MODEL)
            # Override dim from model
            test = self._minilm.encode(["test"], normalize_embeddings=True)
            self.dim = int(test.shape[1])
        except Exception as e:
            # Fall back to hash if model load fails
            self.backend = "hash"
            self._minilm = None
            self._fallback_reason = str(e)

    def _init_openai(self):
        api_key = settings.OPENAI_API_KEY or settings.XAI_API_KEY
        if not api_key:
            self.backend = "hash"
            return
        try:
            from openai import OpenAI
            base = settings.XAI_BASE_URL if settings.XAI_API_KEY and not settings.OPENAI_API_KEY else settings.OPENAI_BASE_URL
            self._openai_client = OpenAI(api_key=api_key, base_url=base)
            self._openai_model = os_getenv_embedding_model()
        except Exception:
            self.backend = "hash"
            self._openai_client = None

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "dim": self.dim,
            "model": settings.EMBEDDING_MODEL if self.backend == "minilm" else (
                getattr(self, "_openai_model", None) if self.backend == "openai" else "hash-ngram-384"
            ),
        }

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.backend == "minilm" and self._minilm is not None:
            arr = self._minilm.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(arr, dtype=np.float32)
        if self.backend == "openai" and self._openai_client is not None:
            return self._embed_openai(list(texts))
        return np.vstack([self._hash_embed(t) for t in texts])

    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic dense semantic-ish embedding via feature hashing (no network)."""
        dim = self.dim
        vec = np.zeros(dim, dtype=np.float32)
        text_l = (text or "").lower()
        tokens = re.findall(r"[\w\u0900-\u097F]+", text_l)
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h // dim) % 2 == 0 else -1.0
            vec[idx] += sign
            # bigrams
        for a, b in zip(tokens, tokens[1:]):
            bg = f"{a}_{b}"
            h = int(hashlib.md5(bg.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h // dim) % 2 == 0 else -1.0
            vec[idx] += 0.5 * sign
        # char trigrams for Marathi / morphological robustness
        compact = re.sub(r"\s+", "", text_l)
        for i in range(max(0, len(compact) - 2)):
            tri = compact[i : i + 3]
            h = int(hashlib.sha1(tri.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h // dim) % 2 == 0 else -1.0
            vec[idx] += 0.25 * sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        model = getattr(self, "_openai_model", "text-embedding-3-small")
        try:
            resp = self._openai_client.embeddings.create(model=model, input=texts)
            vectors = [np.asarray(item.embedding, dtype=np.float32) for item in resp.data]
            mat = np.vstack(vectors)
            # normalize
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
            self.dim = mat.shape[1]
            return mat
        except Exception:
            return np.vstack([self._hash_embed(t) for t in texts])


def os_getenv_embedding_model() -> str:
    import os
    return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


embedding_provider = EmbeddingProvider()
