import pytest
from fastapi.testclient import TestClient
from api.server import app, initialize_database


@pytest.fixture(scope="module")
def client():
    initialize_database()
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["indices_ready"] is True


def test_stats(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_vectors" in data
    assert data["dimension"] == 384


def test_search_and_crud(client):
    # 1. Search existing vectors
    resp = client.post("/search", json={
        "query": "artificial intelligence neural networks",
        "index": "ivf",
        "top_k": 3,
        "n_probe": 8
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 3
    assert data["latency_ms"] >= 0

    # 2. Insert new vector
    resp = client.post("/insert", json={
        "id": "custom_doc_777",
        "text": "Deep Learning revolutionizes quantum computing algorithms.",
        "metadata": {"source": "test_suite"}
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # 3. Search for inserted vector
    search_resp = client.post("/search", json={
        "query": "Deep Learning quantum computing",
        "index": "ivf",
        "top_k": 1,
        "n_probe": 8
    })
    assert search_resp.status_code == 200
    top_hit = search_resp.json()["results"][0]
    assert top_hit["id"] == "custom_doc_777"

    # 4. Delete inserted vector
    del_resp = client.delete("/vectors/custom_doc_777")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "success"

    # 5. Verify deletion
    del_again = client.delete("/vectors/custom_doc_777")
    assert del_again.status_code == 404
