from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    # Prefer JSON health; / may serve React SPA when ui/web/dist is built
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "platform" in data
    root = client.get("/")
    assert root.status_code == 200

def test_query_endpoint():
    response = client.post("/api/query", json={"query": "What fertilizers for Cotton?", "language": "mr"})
    assert response.status_code == 200
    data = response.json()
    assert "synthesized_answer" in data

def test_live_weather_endpoint():
    response = client.get("/api/live/weather?location=Pune")
    assert response.status_code == 200
    data = response.json()
    assert data["location"] == "Pune"

def test_predict_yield_endpoint():
    response = client.post("/api/predict/yield", json={"crop": "Cotton", "acreage": 2.0})
    assert response.status_code == 200
    data = response.json()
    assert data["total_predicted_yield"] > 0

def test_graph_endpoint():
    response = client.get("/api/knowledge/graph/Pomegranate")
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "Pomegranate"
