"""K-Means clustering algorithm built entirely from scratch in pure NumPy.
Implements k-means++ initialization and Lloyd's iterative algorithm on unit sphere.
Zero external ML/ANN dependencies.
"""

from typing import Optional
import numpy as np
from .distance import l2_normalize


class KMeansScratch:
    """Spherical K-Means clustering algorithm from first principles.
    
    Optimized for unit-normalized vectors where Euclidean distance minimization
    is equivalent to Cosine Similarity maximization.
    """

    def __init__(
        self, 
        n_clusters: int = 256, 
        max_iter: int = 25, 
        tol: float = 1e-4, 
        random_state: int = 42
    ):
        """Initialize K-Means parameters.
        
        Args:
            n_clusters: Number of Voronoi partitions (centroids K).
            max_iter: Maximum number of Lloyd's iteration loops.
            tol: Minimum centroid shift tolerance for early stopping.
            random_state: Random seed for deterministic reproducibility.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids: Optional[np.ndarray] = None  # Shape (K, D)
        self.inertia_: float = 0.0

    def _init_kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using the k-means++ probabilistic algorithm.
        
        Spreads out initial centroids to accelerate convergence and avoid bad local minima.
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        k = min(self.n_clusters, n_samples)
        centroids = np.empty((k, n_features), dtype=np.float32)

        # 1. Pick first centroid uniformly at random
        first_idx = rng.integers(0, n_samples)
        centroids[0] = X[first_idx]

        # 2. For remaining centroids, compute squared distance to closest existing centroid
        # On unit sphere, squared Euclidean dist ||x - c||^2 = 2 - 2*(x · c)
        closest_sims = np.dot(X, centroids[0])
        closest_dist_sq = np.maximum(2.0 - 2.0 * closest_sims, 0.0)

        for c_idx in range(1, k):
            probs = closest_dist_sq / np.maximum(np.sum(closest_dist_sq), 1e-12)
            # Sample next centroid based on probability distribution
            next_idx = rng.choice(n_samples, p=probs)
            centroids[c_idx] = X[next_idx]

            # Update closest squared distance with the new centroid
            new_sims = np.dot(X, centroids[c_idx])
            new_dist_sq = np.maximum(2.0 - 2.0 * new_sims, 0.0)
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

        return centroids

    def fit(self, X: np.ndarray) -> "KMeansScratch":
        """Fit K-Means clustering on unit-normalized dataset X.
        
        Args:
            X: Input dataset of shape (N, D), unit normalized.
            
        Returns:
            self: The fitted KMeans instance.
        """
        X = l2_normalize(X)
        n_samples, n_features = X.shape
        
        if n_samples < self.n_clusters:
            # When samples are fewer than clusters, each sample is its own centroid
            self.centroids = X.copy()
            return self

        # Initialize centroids via k-means++
        centroids = self._init_kmeans_plus_plus(X)

        for iteration in range(self.max_iter):
            # 1. Assignment Step: assign each vector to the most similar centroid
            # Dot product (N, D) @ (K, D).T -> (N, K)
            sim_matrix = np.dot(X, centroids.T)
            labels = np.argmax(sim_matrix, axis=1)

            # 2. Update Step: compute mean of vectors in each cluster
            new_centroids = np.zeros_like(centroids)
            counts = np.bincount(labels, minlength=self.n_clusters)
            empty_clusters = []

            for cluster_id in range(self.n_clusters):
                if counts[cluster_id] > 0:
                    cluster_points = X[labels == cluster_id]
                    new_centroids[cluster_id] = np.mean(cluster_points, axis=0)
                else:
                    empty_clusters.append(cluster_id)

            # Heal any empty clusters by picking distinct points with highest residual error
            if empty_clusters:
                assigned_sims = sim_matrix[np.arange(n_samples), labels]
                worst_indices = np.argsort(assigned_sims)  # lowest similarities first
                for i, cluster_id in enumerate(empty_clusters):
                    pick_idx = worst_indices[i % n_samples]
                    new_centroids[cluster_id] = X[pick_idx]

            # Re-normalize centroids to unit sphere
            new_centroids = l2_normalize(new_centroids)

            # Check convergence
            centroid_shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids

            if centroid_shift < self.tol:
                break

        self.centroids = centroids
        # Compute final inertia (sum of squared distances to closest centroid)
        final_sims = np.max(np.dot(X, centroids.T), axis=1)
        self.inertia_ = float(np.sum(np.maximum(2.0 - 2.0 * final_sims, 0.0)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign vectors to nearest cluster centroids.
        
        Args:
            X: Input vectors of shape (M, D).
            
        Returns:
            np.ndarray: Cluster indices of shape (M,).
        """
        if self.centroids is None:
            raise RuntimeError("KMeans must be fitted before calling predict.")
        X_norm = l2_normalize(X)
        sim_matrix = np.dot(X_norm, self.centroids.T)
        return np.argmax(sim_matrix, axis=1)
