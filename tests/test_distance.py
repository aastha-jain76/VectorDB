import pytest
import numpy as np
from vectordb.distance import l2_normalize, cosine_similarity, squared_euclidean


def test_l2_normalize():
    v = np.array([3.0, 4.0], dtype=np.float32)
    norm_v = l2_normalize(v)
    assert np.allclose(norm_v, [0.6, 0.8], atol=1e-5)
    assert np.isclose(np.linalg.norm(norm_v), 1.0, atol=1e-5)


def test_l2_normalize_matrix():
    mat = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    norm_mat = l2_normalize(mat)
    norms = np.linalg.norm(norm_mat, axis=1)
    assert np.allclose(norms, [1.0, 1.0], atol=1e-5)


def test_cosine_similarity():
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])
    w = np.array([0.0, 1.0, 0.0])
    neg = np.array([-1.0, 0.0, 0.0])

    assert np.isclose(cosine_similarity(u, v), 1.0, atol=1e-5)
    assert np.isclose(cosine_similarity(u, w), 0.0, atol=1e-5)
    assert np.isclose(cosine_similarity(u, neg), -1.0, atol=1e-5)


def test_squared_euclidean():
    u = l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]]))
    v = l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]]))
    dist_sq = squared_euclidean(u, v)
    assert np.allclose(np.diag(dist_sq), 0.0, atol=1e-5)
    assert np.allclose(dist_sq[0, 1], 2.0, atol=1e-5)
