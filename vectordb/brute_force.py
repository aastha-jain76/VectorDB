"""Exact Brute-Force Vector Index using pure NumPy matrix multiplication.
Serves as the mathematical ground truth for nearest neighbor search.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from .base import BaseVectorIndex
from .distance import l2_normalize


class BruteForceIndex(BaseVectorIndex):
    """Exact nearest neighbor search via exhaustive linear scan.
    
    All vectors are unit-normalized upon ingestion so that Cosine Similarity
    is computed as a single BLAS matrix-vector dot product:
        scores = X @ q
    Followed by argpartition for O(N) top-k selection.
    """

    def __init__(self, dimension: Optional[int] = None):
        """Initialize the brute-force index.
        
        Args:
            dimension: Optional dimensionality of vectors. If None, inferred on first insert.
        """
        self.dimension = dimension
        self.vectors: Optional[np.ndarray] = None  # Contiguous float32 array (N, D)
        self.id_to_idx: Dict[Union[int, str], int] = {}
        self.idx_to_id: List[Union[int, str]] = []
        self.metadatas: Dict[Union[int, str], Dict[str, Any]] = {}
        self._active_count = 0
        self.last_candidate_count: int = 0

    def insert(
        self, 
        vector_id: Union[int, str], 
        vector: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Insert or overwrite a vector."""
        vec_norm = l2_normalize(vector).reshape(1, -1)
        if self.dimension is None:
            self.dimension = vec_norm.shape[1]
        elif vec_norm.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vec_norm.shape[1]} does not match index {self.dimension}")

        if vector_id in self.id_to_idx:
            # Update existing vector
            idx = self.id_to_idx[vector_id]
            self.vectors[idx] = vec_norm[0]
            if metadata is not None:
                self.metadatas[vector_id] = metadata
            return

        idx = len(self.idx_to_id)
        if self.vectors is None:
            # Initialize with small initial capacity
            self.vectors = np.empty((1024, self.dimension), dtype=np.float32)
        elif idx >= len(self.vectors):
            # Double array capacity
            new_capacity = max(len(self.vectors) * 2, idx + 1024)
            new_vectors = np.empty((new_capacity, self.dimension), dtype=np.float32)
            new_vectors[:idx] = self.vectors[:idx]
            self.vectors = new_vectors

        self.vectors[idx] = vec_norm[0]
        self.id_to_idx[vector_id] = idx
        self.idx_to_id.append(vector_id)
        if metadata is not None:
            self.metadatas[vector_id] = metadata
        self._active_count += 1

    def batch_insert(
        self, 
        vectors: np.ndarray, 
        ids: List[Union[int, str]], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Efficiently batch insert multiple vectors."""
        if len(vectors) != len(ids):
            raise ValueError(f"Length mismatch: {len(vectors)} vectors vs {len(ids)} IDs")
        
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate IDs detected in batch_insert. IDs within a batch must be unique.")
        
        vecs_norm = l2_normalize(vectors)
        n, d = vecs_norm.shape
        
        if self.dimension is None:
            self.dimension = d
        elif d != self.dimension:
            raise ValueError(f"Batch dimension {d} does not match index {self.dimension}")

        start_idx = len(self.idx_to_id)
        total_needed = start_idx + n
        
        if self.vectors is None:
            self.vectors = np.empty((total_needed, self.dimension), dtype=np.float32)
        elif len(self.vectors) < total_needed:
            new_capacity = max(len(self.vectors) * 2, total_needed)
            new_vectors = np.empty((new_capacity, self.dimension), dtype=np.float32)
            new_vectors[:start_idx] = self.vectors[:start_idx]
            self.vectors = new_vectors

        self.vectors[start_idx:start_idx + n] = vecs_norm
        
        for i, vid in enumerate(ids):
            self.id_to_idx[vid] = start_idx + i
            self.idx_to_id.append(vid)
            if metadatas and i < len(metadatas) and metadatas[i] is not None:
                self.metadatas[vid] = metadatas[i]

        self._active_count += n

    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 10,
        **kwargs
    ) -> List[Tuple[Union[int, str], float, Dict[str, Any]]]:
        """Exhaustive linear scan computing dot product against all active vectors."""
        if self._active_count == 0:
            self.last_candidate_count = 0
            return []

        q_norm = l2_normalize(query_vector).reshape(-1)
        if q_norm.shape[0] != self.dimension:
            raise ValueError(f"Query dim {q_norm.shape[0]} != index dim {self.dimension}")

        active_n = len(self.idx_to_id)
        self.last_candidate_count = active_n
        # Sliced view of valid active vectors
        active_matrix = self.vectors[:active_n]

        # BLAS matrix-vector dot product (pure NumPy)
        scores = np.dot(active_matrix, q_norm)

        actual_k = min(top_k, active_n)
        if actual_k <= 0:
            return []

        # Use argpartition for O(N) selection followed by sorting only top_k
        if actual_k < active_n:
            candidate_indices = np.argpartition(scores, -actual_k)[-actual_k:]
            sorted_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
        else:
            sorted_indices = np.argsort(-scores)

        results = []
        for idx in sorted_indices:
            vid = self.idx_to_id[idx]
            score = float(scores[idx])
            meta = self.metadatas.get(vid, {})
            results.append((vid, score, meta))

        return results

    def delete(self, vector_id: Union[int, str]) -> bool:
        """Remove a vector by ID using swap-and-pop for O(1) compaction."""
        if vector_id not in self.id_to_idx:
            return False

        idx_to_remove = self.id_to_idx[vector_id]
        last_idx = len(self.idx_to_id) - 1

        if idx_to_remove != last_idx:
            # Swap with last element to maintain contiguous array without reallocating
            last_id = self.idx_to_id[last_idx]
            self.vectors[idx_to_remove] = self.vectors[last_idx]
            self.idx_to_id[idx_to_remove] = last_id
            self.id_to_idx[last_id] = idx_to_remove

        # Pop the last element
        self.idx_to_id.pop()
        del self.id_to_idx[vector_id]
        if vector_id in self.metadatas:
            del self.metadatas[vector_id]

        self._active_count -= 1
        return True

    def __len__(self) -> int:
        return self._active_count
