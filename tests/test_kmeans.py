import pytest
import numpy as np
from vectordb.kmeans import KMeansScratch
from vectordb.distance import l2_normalize


def test_kmeans_scratch_convergence():
    rng = np.random.default_rng(42)
    # Generate 3 distinct spherical clusters
    c1 = l2_normalize(rng.normal(loc=[10.0, 0.0, 0.0], scale=0.5, size=(100, 3)))
    c2 = l2_normalize(rng.normal(loc=[0.0, 10.0, 0.0], scale=0.5, size=(100, 3)))
    c3 = l2_normalize(rng.normal(loc=[0.0, 0.0, 10.0], scale=0.5, size=(100, 3)))
    data = np.vstack([c1, c2, c3])

    kmeans = KMeansScratch(n_clusters=3, max_iter=20, random_state=42)
    kmeans.fit(data)

    assert kmeans.centroids.shape == (3, 3)
    # Centroids must be unit normalized
    centroid_norms = np.linalg.norm(kmeans.centroids, axis=1)
    assert np.allclose(centroid_norms, 1.0, atol=1e-5)

    labels = kmeans.predict(data)
    assert len(np.unique(labels)) == 3
    assert kmeans.inertia_ >= 0.0
