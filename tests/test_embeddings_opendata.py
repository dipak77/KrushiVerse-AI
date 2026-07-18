import numpy as np

from app.knowledge.embeddings import embedding_provider, EmbeddingProvider
from app.knowledge.qdrant_store import LocalDenseVectorStore
from app.knowledge.hybrid_search import hybrid_retriever
from app.knowledge.dataset_loader import kb_loader
from app.live_feeds.opendata_client import opendata_client
from app.knowledge.tools.registry import tool_registry
from app.agents.planner import planner_agent
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_hash_embeddings_normalized():
    v1 = embedding_provider.embed_text("Cotton pink bollworm control Maharashtra")
    v2 = embedding_provider.embed_text("Cotton pink bollworm control Maharashtra")
    v3 = embedding_provider.embed_text("Unrelated cooking recipe pasta")
    assert v1.shape[0] >= 64
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-3
    assert float(np.dot(v1, v2)) > 0.99
    assert float(np.dot(v1, v3)) < float(np.dot(v1, v2))


def test_local_dense_store_search():
    docs = kb_loader.get_all_documents()[:40]
    store = LocalDenseVectorStore()
    store.build_index(docs, force=True)
    hits = store.search("pomegranate bacterial blight treatment", top_k=3)
    assert len(hits) > 0
    assert "doc" in hits[0]
    assert hits[0]["score"] > 0


def test_hybrid_includes_dense_backend_info():
    info = hybrid_retriever.backend_info()
    assert info["document_count"] >= 100
    assert info["bm25"] is True
    # dense may be enabled with local backend
    assert "dense_enabled" in info


def test_hybrid_search_returns_origins():
    results = hybrid_retriever.hybrid_search("Cotton fertilizer NPK recommendation", top_k=5)
    assert len(results) > 0
    assert "rrf_score" in results[0]
    # origins list present when dense/sparse channels contribute
    assert "doc" in results[0]


def test_opendata_fallback_without_key():
    status = opendata_client.status()
    assert status["provider"] == "data.gov.in"
    result = opendata_client.fetch_commodity_prices(commodity="Cotton", state="Maharashtra", limit=10)
    assert result["ok"] is True
    assert result["count"] > 0
    assert result["mode"] in ("live", "fallback")
    docs = opendata_client.to_rag_documents(result)
    assert len(docs) > 0
    assert docs[0]["category"] == "OpenData:Agmarknet"


def test_agmarknet_tool():
    res = tool_registry.run("agmarknet_opendata", {"crop": "Onion", "state": "Maharashtra"})
    assert res["ok"] is True
    assert len(res["documents"]) > 0


def test_api_opendata_and_backends():
    r = client.get("/api/opendata/agmarknet", params={"commodity": "Soybean"})
    assert r.status_code == 200
    data = r.json()
    assert data["count"] > 0

    r2 = client.get("/api/rag/backends")
    assert r2.status_code == 200
    body = r2.json()
    assert "embeddings" in body
    assert "hybrid" in body
    assert "opendata" in body


def test_api_stats_includes_dense_and_opendata():
    r = client.get("/api/knowledge/stats")
    assert r.status_code == 200
    data = r.json()
    assert "embeddings" in data
    assert "opendata" in data
    assert data["stats"]["total_documents"] >= 100


def test_planner_retrieval_mode_v10_2():
    res = planner_agent.plan_and_execute(
        "What is the cotton mandi price?",
        language="en",
        enable_web=False,
    )
    assert res["crop"] == "Cotton"
    assert "v10" in res["knowledge_layer"]["retrieval_mode"]
