"""IVF-Flat (Inverted File Index with Flat Vectors) implemented from scratch in pure NumPy.
No FAISS, no sklearn.neighbors, strictly zero external ANN libraries.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from .base import BaseVectorIndex
from .distance import l2_normalize
from .kmeans import KMeansScratch


class IVFFlatIndex(BaseVectorIndex):
    """Inverted File Index (IVF-Flat).
    
    Partitions the vector space into K Voronoi cells via scratch K-Means.
    Vectors are stored in inverted posting lists associated with their nearest centroid.
    At query time, only the candidate vectors in the n_probe closest cells are evaluated,
    achieving massive sub-linear speedup while preserving high recall.
    """

    def __init__(
        self, 
        n_clusters: int = 256, 
        n_probe: int = 8, 
        dimension: Optional[int] = None,
        random_state: int = 42
    ):
        """Initialize IVF-Flat Index.
        
        Args:
            n_clusters: Number of Voronoi partitions (centroids K).
            n_probe: Default number of centroids to probe during search (1 <= n_probe <= n_clusters).
            dimension: Dimensionality of vectors.
            random_state: Random seed for clustering reproducibility.
        """
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.dimension = dimension
        self.random_state = random_state

        self.centroids: Optional[np.ndarray] = None  # Shape (K, D)
        self.is_trained: bool = False

        # Inverted posting lists: cluster_id -> list of internal indices
        self.inverted_lists: Dict[int, List[int]] = {c: [] for c in range(n_clusters)}
        
        # Primary storage
        self.vectors: Optional[np.ndarray] = None  # Master contiguous array (N, D)
        self.id_to_idx: Dict[Union[int, str], int] = {}
        self.idx_to_id: List[Union[int, str]] = []
        self.idx_to_cluster: List[int] = []
        self.metadatas: Dict[Union[int, str], Dict[str, Any]] = {}
        self._active_count = 0
        self.last_candidate_count: int = 0

    def train(self, vectors: np.ndarray) -> "IVFFlatIndex":
        """Train K-Means centroids on a representative sample of vectors.
        
        Args:
            vectors: 2D array of training vectors of shape (N, D).
        """
        vecs_norm = l2_normalize(vectors)
        self.dimension = vecs_norm.shape[1]
        
        # Train scratch K-Means
        kmeans = KMeansScratch(
            n_clusters=self.n_clusters, 
            max_iter=25, 
            random_state=self.random_state
        )
        kmeans.fit(vecs_norm)
        self.centroids = kmeans.centroids
        self.is_trained = True
        return self

    def build_index(
        self, 
        vectors: np.ndarray, 
        ids: List[Union[int, str]], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> "IVFFlatIndex":
        """Train centroids and ingest full dataset in one optimized pass."""
        self.train(vectors)
        self.batch_insert(vectors, ids, metadatas)
        return self

    def insert(
        self, 
        vector_id: Union[int, str], 
        vector: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Insert or update a single vector into the inverted file index."""
        if not self.is_trained:
            raise RuntimeError(
                "IVFFlatIndex must be trained via train() or build_index() before single inserts. "
                "IVF requires K centroids to partition the vector space."
            )
        
        vec_norm = l2_normalize(vector).reshape(1, -1)
        if vec_norm.shape[1] != self.dimension:
            raise ValueError(f"Vector dim {vec_norm.shape[1]} != index dim {self.dimension}")

        # If vector already exists, delete it first to cleanly update
        if vector_id in self.id_to_idx:
            self.delete(vector_id)

        # 1. Find closest centroid
        centroid_sims = np.dot(self.centroids, vec_norm[0])
        cluster_id = int(np.argmax(centroid_sims))

        # 2. Append to master storage
        idx = len(self.idx_to_id)
        if self.vectors is None:
            self.vectors = np.empty((1024, self.dimension), dtype=np.float32)
        elif idx >= len(self.vectors):
            new_capacity = max(len(self.vectors) * 2, idx + 1024)
            new_vectors = np.empty((new_capacity, self.dimension), dtype=np.float32)
            new_vectors[:idx] = self.vectors[:idx]
            self.vectors = new_vectors

        self.vectors[idx] = vec_norm[0]
        self.id_to_idx[vector_id] = idx
        self.idx_to_id.append(vector_id)
        self.idx_to_cluster.append(cluster_id)
        self.inverted_lists[cluster_id].append(idx)

        if metadata is not None:
            self.metadatas[vector_id] = metadata
        self._active_count += 1

    def batch_insert(
        self, 
        vectors: np.ndarray, 
        ids: List[Union[int, str]], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Batch insert multiple vectors into the inverted lists."""
        if len(vectors) != len(ids):
            raise ValueError(f"Length mismatch: {len(vectors)} vectors vs {len(ids)} IDs")
        
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate IDs detected in batch_insert. IDs within a batch must be unique.")
        
        vecs_norm = l2_normalize(vectors)
        n, d = vecs_norm.shape

        if not self.is_trained:
            self.train(vecs_norm)

        if d != self.dimension:
            raise ValueError(f"Batch dimension {d} != index dimension {self.dimension}")

        start_idx = len(self.idx_to_id)
        total_needed = start_idx + n

        # Allocate / resize master array
        if self.vectors is None:
            self.vectors = np.empty((total_needed, self.dimension), dtype=np.float32)
        elif len(self.vectors) < total_needed:
            new_capacity = max(len(self.vectors) * 2, total_needed)
            new_vectors = np.empty((new_capacity, self.dimension), dtype=np.float32)
            new_vectors[:start_idx] = self.vectors[:start_idx]
            self.vectors = new_vectors

        self.vectors[start_idx:start_idx + n] = vecs_norm

        # Find closest centroids in batch (N, K)
        # Dot product with centroids
        centroid_sims = np.dot(vecs_norm, self.centroids.T)
        cluster_assignments = np.argmax(centroid_sims, axis=1)

        for i, vid in enumerate(ids):
            idx = start_idx + i
            cluster_id = int(cluster_assignments[i])
            self.id_to_idx[vid] = idx
            self.idx_to_id.append(vid)
            self.idx_to_cluster.append(cluster_id)
            self.inverted_lists[cluster_id].append(idx)
            if metadatas and i < len(metadatas) and metadatas[i] is not None:
                self.metadatas[vid] = metadatas[i]

        self._active_count += n

    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 10, 
        n_probe: Optional[int] = None,
        **kwargs
    ) -> List[Tuple[Union[int, str], float, Dict[str, Any]]]:
        """Query the index using multi-probe inverted list routing.
        
        Args:
            query_vector: 1D query vector.
            top_k: Number of nearest neighbors to retrieve.
            n_probe: Number of nearest centroids to probe. If None, uses default self.n_probe.
            
        Returns:
            List of (vector_id, score, metadata) tuples.
        """
        if self._active_count == 0 or not self.is_trained:
            self.last_candidate_count = 0
            return []

        probe = n_probe if n_probe is not None else self.n_probe
        probe = max(1, min(probe, self.n_clusters))

        q_norm = l2_normalize(query_vector).reshape(-1)

        # 1. Compute dot product against all K centroids
        centroid_sims = np.dot(self.centroids, q_norm)

        # 2. Select top n_probe centroids
        if probe < self.n_clusters:
            probed_clusters = np.argpartition(centroid_sims, -probe)[-probe:]
        else:
            probed_clusters = np.arange(self.n_clusters)

        # 3. Gather candidate internal indices from the probed posting lists
        candidate_indices: List[int] = []
        for c in probed_clusters:
            candidate_indices.extend(self.inverted_lists[c])

        if not candidate_indices:
            self.last_candidate_count = 0
            return []

        self.last_candidate_count = len(candidate_indices)

        cand_indices_arr = np.array(candidate_indices, dtype=np.int64)
        # Sliced candidate matrix (M, D) where M << N
        candidate_vectors = self.vectors[cand_indices_arr]

        # 4. Compute dot products only for the candidates
        candidate_scores = np.dot(candidate_vectors, q_norm)

        # 5. Top-k selection among candidates
        m = len(candidate_scores)
        actual_k = min(top_k, m)

        if actual_k < m:
            top_cand_idx = np.argpartition(candidate_scores, -actual_k)[-actual_k:]
            sorted_order = top_cand_idx[np.argsort(-candidate_scores[top_cand_idx])]
        else:
            sorted_order = np.argsort(-candidate_scores)

        results = []
        for order_idx in sorted_order:
            idx = cand_indices_arr[order_idx]
            vid = self.idx_to_id[idx]
            score = float(candidate_scores[order_idx])
            meta = self.metadatas.get(vid, {})
            results.append((vid, score, meta))

        return results

    def delete(self, vector_id: Union[int, str]) -> bool:
        """Remove a vector by ID and prune from its inverted list."""
        if vector_id not in self.id_to_idx:
            return False

        idx_to_remove = self.id_to_idx[vector_id]
        cluster_id = self.idx_to_cluster[idx_to_remove]
        last_idx = len(self.idx_to_id) - 1

        # 1. Remove from its inverted list
        self.inverted_lists[cluster_id].remove(idx_to_remove)

        # 2. Swap-and-pop with the last vector to keep self.vectors contiguous
        if idx_to_remove != last_idx:
            last_id = self.idx_to_id[last_idx]
            last_cluster = self.idx_to_cluster[last_idx]

            # Move vector data
            self.vectors[idx_to_remove] = self.vectors[last_idx]
            self.idx_to_id[idx_to_remove] = last_id
            self.idx_to_cluster[idx_to_remove] = last_cluster
            self.id_to_idx[last_id] = idx_to_remove

            # Update last_id's index in its inverted list
            inv_list = self.inverted_lists[last_cluster]
            for i in range(len(inv_list)):
                if inv_list[i] == last_idx:
                    inv_list[i] = idx_to_remove
                    break

        # 3. Pop the last element
        self.idx_to_id.pop()
        self.idx_to_cluster.pop()
        del self.id_to_idx[vector_id]
        if vector_id in self.metadatas:
            del self.metadatas[vector_id]

        self._active_count -= 1
        return True

    def __len__(self) -> int:
        return self._active_count
