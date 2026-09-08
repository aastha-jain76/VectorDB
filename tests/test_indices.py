import pytest
import numpy as np
from vectordb.brute_force import BruteForceIndex
from vectordb.ivf_flat import IVFFlatIndex
from vectordb.hnsw import HNSWIndex
from vectordb.distance import l2_normalize


@pytest.mark.parametrize("index_cls, kwargs", [
    (BruteForceIndex, {"dimension": 16}),
    (IVFFlatIndex, {"n_clusters": 4, "n_probe": 4, "dimension": 16}),
    (HNSWIndex, {"m": 8, "ef_construction": 16, "ef_search": 16, "dimension": 16}),
])
def test_index_lifecycle(index_cls, kwargs):
    rng = np.random.default_rng(42)
    dim = 16
    index = index_cls(**kwargs)

    # 1. Empty search
    assert index.search(rng.normal(size=dim), top_k=5) == []
    assert len(index) == 0

    # 2. Ingest 50 vectors
    data = l2_normalize(rng.normal(size=(50, dim)))
    ids = [f"doc_{i}" for i in range(50)]
    metas = [{"text": f"text {i}"} for i in range(50)]
    
    if hasattr(index, 'build_index'):
        index.build_index(data, ids, metas)
    else:
        index.batch_insert(data, ids, metas)

    assert len(index) == 50

    # 3. Query with known vector (doc_7)
    q = data[7]
    results = index.search(q, top_k=5)
    assert len(results) == 5
    # The top result should be doc_7 itself
    top_id, top_score, top_meta = results[0]
    assert top_id == "doc_7"
    assert np.isclose(top_score, 1.0, atol=1e-3)
    assert top_meta["text"] == "text 7"

    # 4. Top-k > N edge case
    large_k_results = index.search(q, top_k=100)
    assert len(large_k_results) == 50

    # 5. Deletion test
    assert index.delete("doc_7") is True
    assert len(index) == 49
    # Deleting again returns False
    assert index.delete("doc_7") is False

    # Search again; doc_7 should no longer be in results
    new_results = index.search(q, top_k=5)
    assert all(r[0] != "doc_7" for r in new_results)

    # 6. Single insert
    new_vec = l2_normalize(rng.normal(size=dim))
    index.insert("new_doc", new_vec, {"text": "inserted"})
    assert len(index) == 50
    inserted_res = index.search(new_vec, top_k=1)
    assert inserted_res[0][0] == "new_doc"


def test_hnsw_compaction():
    rng = np.random.default_rng(42)
    dim = 8
    hnsw = HNSWIndex(m=4, ef_construction=8, ef_search=8, dimension=dim, random_state=42)
    data = l2_normalize(rng.normal(size=(20, dim)))
    ids = [f"id_{i}" for i in range(20)]
    hnsw.batch_insert(data, ids)

    assert len(hnsw) == 20

    # Delete 2 items (10% of dataset, below 15% threshold)
    hnsw.delete("id_0")
    hnsw.delete("id_1")
    assert len(hnsw) == 18
    assert len(hnsw.tombstones) == 2
    assert len(hnsw.vectors) == 20

    # Delete 3rd item (3/20 = 15% threshold) -> triggers automatic compaction
    hnsw.delete("id_2")
    assert len(hnsw) == 17
    assert len(hnsw.tombstones) == 0
    assert len(hnsw.vectors) == 17

    # Test explicit force compaction
    hnsw.delete("id_3")
    assert len(hnsw.tombstones) == 1
    hnsw.compact(force=True)
    assert len(hnsw.tombstones) == 0
    assert len(hnsw.vectors) == 16

    # Verify search still functions accurately
    results = hnsw.search(data[10], top_k=3)
    assert len(results) == 3
    assert results[0][0] == "id_10"


def test_hnsw_reinsert_tombstone():
    rng = np.random.default_rng(42)
    dim = 8
    hnsw = HNSWIndex(m=4, ef_construction=8, ef_search=8, dimension=dim, random_state=42)
    data = l2_normalize(rng.normal(size=(10, dim)))
    ids = [f"id_{i}" for i in range(10)]
    hnsw.batch_insert(data, ids)

    # Delete id_5
    assert hnsw.delete("id_5") is True
    assert len(hnsw) == 9

    # Re-insert id_5 with a new vector
    new_v = l2_normalize(rng.normal(size=dim))
    hnsw.insert("id_5", new_v, {"text": "re-inserted 5"})
    assert len(hnsw) == 10

    # Search with new vector, should retrieve id_5 as top 1
    res = hnsw.search(new_v, top_k=1)
    assert len(res) == 1
    assert res[0][0] == "id_5"
    assert res[0][2]["text"] == "re-inserted 5"


def test_ivf_untrained_insert_raises():
    ivf = IVFFlatIndex(n_clusters=4, n_probe=2, dimension=8)
    vec = np.random.randn(8)
    with pytest.raises(RuntimeError, match="must be trained"):
        ivf.insert("doc_1", vec)

