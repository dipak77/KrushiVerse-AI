from app.knowledge.dataset_loader import kb_loader
from app.knowledge.vector_store import SimpleVectorStore
from app.knowledge.hybrid_search import hybrid_retriever
from app.knowledge.graph_rag import graph_rag

def test_kb_loader_docs():
    docs = kb_loader.get_all_documents()
    assert len(docs) > 0
    categories = {d["category"] for d in docs}
    assert "Crop" in categories
    assert "Disease" in categories
    assert "Government Scheme" in categories

def test_vector_store():
    store = SimpleVectorStore()
    docs = kb_loader.get_all_documents()
    store.build_index(docs)
    results = store.search("Cotton pink bollworm control", top_k=2)
    assert len(results) > 0
    assert results[0]["score"] > 0

def test_hybrid_search():
    results = hybrid_retriever.hybrid_search("Bacterial blight pomegranate", top_k=3)
    assert len(results) > 0
    assert "rrf_score" in results[0]

def test_graph_rag():
    data = graph_rag.get_crop_ecosystem("Pomegranate")
    assert data["crop"] == "Pomegranate"
    assert "pests_and_diseases" in data
    assert "applicable_schemes" in data
