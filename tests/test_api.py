import os
import pytest
import numpy as np
from fastapi.testclient import TestClient
import api.server as server
from api.server import app
from vectordb.brute_force import BruteForceIndex
from vectordb.ivf_flat import IVFFlatIndex
from vectordb.distance import l2_normalize


@pytest.fixture(scope="module")
def client():
    server.initialize_database()
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
    ivf_info = data["indices"]["ivf_flat"]
    assert ivf_info["n_clusters"] in (4, 256)
    assert ivf_info["default_n_probe"] in (2, 8)
    assert ivf_info["is_trained"] is True


def test_search_and_crud(client):
    # 1. Search existing vectors
    resp = client.post("/search", json={
        "query": "artificial intelligence neural networks",
        "index": "ivf",
        "top_k": 3,
        "n_probe": 2
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 3
    assert data["latency_ms"] >= 0
    assert "total_vectors_indexed" in data
    assert "candidate_vectors_evaluated" in data
    assert data["candidate_vectors_evaluated"] > 0

    # 2. Insert new vector via text
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
        "n_probe": 4
    })
    assert search_resp.status_code == 200
    top_hit = search_resp.json()["results"][0]
    assert top_hit["id"] == "custom_doc_777"

    # 4. Search via direct dense vector using brute force
    rng = np.random.default_rng(42)
    q_vec = rng.normal(size=384).tolist()
    bf_resp = client.post("/search", json={
        "vector": q_vec,
        "index": "brute_force",
        "top_k": 2
    })
    assert bf_resp.status_code == 200
    bf_data = bf_resp.json()
    assert len(bf_data["results"]) == 2
    assert bf_data["candidate_vectors_evaluated"] == bf_data["total_vectors_indexed"]

    # 5. Delete inserted vector
    del_resp = client.delete("/vectors/custom_doc_777")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "success"

    # 6. Verify deletion
    del_again = client.delete("/vectors/custom_doc_777")
    assert del_again.status_code == 404


def test_untrained_ivf_insert_error_api(client):
    orig_ivf = server.ivf_index
    rng = np.random.default_rng(42)
    try:
        server.ivf_index = IVFFlatIndex(n_clusters=4, dimension=384)
        resp = client.post("/insert", json={
            "id": "bad_doc",
            "vector": rng.normal(size=384).tolist()
        })
        assert resp.status_code == 400
        assert "must be trained" in resp.json()["detail"]
    finally:
        server.ivf_index = orig_ivf


def test_effective_probe_reporting(client):
    # Request n_probe=999 which exceeds n_clusters
    resp = client.post("/search", json={
        "query": "artificial intelligence",
        "index": "ivf",
        "top_k": 3,
        "n_probe": 999
    })
    assert resp.status_code == 200
    data = resp.json()
    effective_k = server.ivf_index.n_clusters
    assert data["index_used"] == f"ivf_flat (n_probe={effective_k})"


def test_desynchronized_deletion_handling(client):
    # Artificially insert a vector only into BF to simulate orphaned state
    test_id = "orphan_doc_999"
    rng = np.random.default_rng(42)
    q_vec = rng.normal(size=server.dimension).astype(np.float32)
    server.bf_index.insert(test_id, q_vec)
    assert test_id in server.bf_index.id_to_idx
    assert test_id not in server.ivf_index.id_to_idx

    # Attempting to delete should detect state desynchronization, purge the orphan, and return 500
    del_resp = client.delete(f"/vectors/{test_id}")
    assert del_resp.status_code == 500
    assert "desynchronization detected" in del_resp.json()["detail"]
    assert test_id not in server.bf_index.id_to_idx
    assert test_id not in server.ivf_index.id_to_idx


