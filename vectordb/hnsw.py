"""Hierarchical Navigable Small World (HNSW) Index implemented from scratch in pure Python and NumPy.
No external graph or ANN search libraries (zero FAISS / zero sklearn).
"""

import math
import heapq
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from .base import BaseVectorIndex
from .distance import l2_normalize


class HNSWIndex(BaseVectorIndex):
    """Hierarchical Navigable Small World (HNSW) graph index.
    
    Constructs a multi-layer graph where upper layers contain sparse long-range highway links
    for fast logarithmic navigation, and lower layers provide dense short-range local links.
    
    Deletion Strategy:
        Deletion in a graph index is genuinely awkward. If an internal routing node is
        suddenly removed, paths in the navigable small-world graph can become disconnected,
        reducing recall for distant nodes unless an expensive re-triangulation is executed.
        Therefore, this implementation uses Tombstone Masking (soft deletion):
        the node remains in the graph as a routing waypoint/bridge so navigation paths
        remain intact, but it is strictly filtered out from query search results.
    """

    def __init__(
        self, 
        m: int = 16, 
        ef_construction: int = 64, 
        ef_search: int = 32, 
        m0: Optional[int] = None,
        dimension: Optional[int] = None,
        random_state: int = 42
    ):
        """Initialize HNSW parameters.
        
        Args:
            m: Maximum number of outgoing edges per node on layers > 0.
            ef_construction: Size of dynamic candidate list during construction.
            ef_search: Size of dynamic candidate list during search.
            m0: Maximum number of outgoing edges per node on layer 0 (defaults to 2 * m).
            dimension: Dimensionality of vectors.
            random_state: Random seed for level generation reproducibility.
        """
        self.m = m
        self.m0 = m0 if m0 is not None else 2 * m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.dimension = dimension
        self.random_state = random_state
        self.rng = random.Random(random_state)
        self.m_l = 1.0 / math.log(m)

        # Graph structures: layers[level][node_id] -> set of neighbor node_ids
        self.layers: List[Dict[int, Set[int]]] = []
        self.entry_point: Optional[int] = None
        self.max_level: int = -1

        # Storage
        self.vectors: List[np.ndarray] = []  # Index -> unit-normalized vector
        self.id_to_idx: Dict[Union[int, str], int] = {}
        self.idx_to_id: List[Union[int, str]] = []
        self.metadatas: Dict[Union[int, str], Dict[str, Any]] = {}
        self.tombstones: Set[int] = set()
        self._active_count: int = 0

    def _random_level(self) -> int:
        """Sample node insertion level via exponential decay."""
        r = self.rng.random()
        while r == 0:
            r = self.rng.random()
        return int(-math.log(r) * self.m_l)

    def _distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute angular distance on unit vectors: 1.0 - (u · v)."""
        return 1.0 - float(np.dot(u, v))

    def _search_layer(
        self, 
        query: np.ndarray, 
        entry_points: List[int], 
        ef: int, 
        level: int
    ) -> List[Tuple[float, int]]:
        """Beam search exploring neighborhood at a specific graph layer."""
        visited: Set[int] = set(entry_points)
        candidates: List[Tuple[float, int]] = []
        best_w: List[Tuple[float, int]] = []

        for ep in entry_points:
            d = self._distance(query, self.vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(best_w, (-d, ep))

        layer_graph = self.layers[level]

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_best_dist = -best_w[0][0]

            if c_dist > furthest_best_dist:
                break

            neighbors = layer_graph.get(c_id, set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    n_dist = self._distance(query, self.vectors[neighbor])
                    furthest_best_dist = -best_w[0][0]

                    if n_dist < furthest_best_dist or len(best_w) < ef:
                        heapq.heappush(candidates, (n_dist, neighbor))
                        heapq.heappush(best_w, (-n_dist, neighbor))
                        if len(best_w) > ef:
                            heapq.heappop(best_w)

        return sorted([(-item[0], item[1]) for item in best_w])

    def insert(
        self, 
        vector_id: Union[int, str], 
        vector: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Insert a vector into the HNSW multi-layer graph."""
        vec_norm = l2_normalize(vector).reshape(-1)
        if self.dimension is None:
            self.dimension = vec_norm.shape[0]
        elif vec_norm.shape[0] != self.dimension:
            raise ValueError(f"Vector dim {vec_norm.shape[0]} != index dim {self.dimension}")

        if vector_id in self.id_to_idx:
            old_node_id = self.id_to_idx[vector_id]
            if old_node_id not in self.tombstones:
                self.tombstones.add(old_node_id)
                self._active_count -= 1
            self.idx_to_id[old_node_id] = None
            del self.id_to_idx[vector_id]
            if vector_id in self.metadatas:
                del self.metadatas[vector_id]

        node_id = len(self.vectors)
        self.vectors.append(vec_norm)
        self.id_to_idx[vector_id] = node_id
        self.idx_to_id.append(vector_id)
        if metadata is not None:
            self.metadatas[vector_id] = metadata

        node_level = self._random_level()

        # Ensure layers list has enough levels
        while len(self.layers) <= node_level:
            self.layers.append({})

        # Initialize node slot for all layers up to node_level
        for l in range(node_level + 1):
            self.layers[l][node_id] = set()

        curr_obj = self.entry_point
        max_l = self.max_level

        if curr_obj is None:
            # First element inserted
            self.entry_point = node_id
            self.max_level = node_level
            self._active_count += 1
            return

        # 1. Greedy routing through upper layers down to min(max_level, node_level + 1)
        for l in range(max_l, node_level, -1):
            changed = True
            while changed:
                changed = False
                curr_dist = self._distance(vec_norm, self.vectors[curr_obj])
                for neighbor in self.layers[l].get(curr_obj, set()):
                    d = self._distance(vec_norm, self.vectors[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # 2. Insert into layers from min(max_l, node_level) down to 0
        top_l = min(max_l, node_level)
        entry_points = [curr_obj]

        for l in range(top_l, -1, -1):
            m_max = self.m0 if l == 0 else self.m
            w = self._search_layer(vec_norm, entry_points, ef=self.ef_construction, level=l)
            neighbors = [item[1] for item in w[:m_max]]
            
            for n_id in neighbors:
                self.layers[l][node_id].add(n_id)
                self.layers[l][n_id].add(node_id)
                if len(self.layers[l][n_id]) > m_max:
                    n_vec = self.vectors[n_id]
                    all_nbrs = list(self.layers[l][n_id])
                    dists = [(self._distance(n_vec, self.vectors[nbr]), nbr) for nbr in all_nbrs]
                    dists.sort()
                    self.layers[l][n_id] = {nbr for _, nbr in dists[:m_max]}

            entry_points = [item[1] for item in w]

        if node_level > self.max_level:
            self.max_level = node_level
            self.entry_point = node_id

        self._active_count += 1

    def batch_insert(
        self, 
        vectors: np.ndarray, 
        ids: List[Union[int, str]], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Batch insert vectors into HNSW."""
        if len(vectors) != len(ids):
            raise ValueError(f"Length mismatch: {len(vectors)} vectors vs {len(ids)} IDs")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate IDs detected in batch_insert. IDs within a batch must be unique.")
        for i, (vid, vec) in enumerate(zip(ids, vectors)):
            meta = metadatas[i] if metadatas and i < len(metadatas) else None
            self.insert(vid, vec, meta)

    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 10, 
        ef_search: Optional[int] = None,
        **kwargs
    ) -> List[Tuple[Union[int, str], float, Dict[str, Any]]]:
        """Search nearest neighbors via HNSW greedy traversal + beam search."""
        if self._active_count == 0 or self.entry_point is None:
            return []

        q_norm = l2_normalize(query_vector).reshape(-1)
        ef = ef_search if ef_search is not None else self.ef_search
        ef = max(ef, top_k)

        curr_obj = self.entry_point

        # 1. Greedy search in upper layers
        for l in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                curr_dist = self._distance(q_norm, self.vectors[curr_obj])
                for neighbor in self.layers[l].get(curr_obj, set()):
                    d = self._distance(q_norm, self.vectors[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # 2. Beam search in bottom layer (level 0)
        candidates = self._search_layer(q_norm, [curr_obj], ef=ef, level=0)

        # 3. Filter out tombstoned nodes and format results
        results = []
        for dist, node_id in candidates:
            if node_id in self.tombstones or self.idx_to_id[node_id] is None:
                continue
            vid = self.idx_to_id[node_id]
            sim_score = 1.0 - dist
            meta = self.metadatas.get(vid, {})
            results.append((vid, sim_score, meta))
            if len(results) >= top_k:
                break

        return results

    def delete(self, vector_id: Union[int, str]) -> bool:
        """Tombstone a vector by ID (soft delete).
        
        If tombstone ratio exceeds 15%, auto-compaction is triggered to prune orphaned nodes.
        """
        if vector_id not in self.id_to_idx:
            return False

        node_id = self.id_to_idx[vector_id]
        if node_id in self.tombstones:
            return False

        self.tombstones.add(node_id)
        if vector_id in self.metadatas:
            del self.metadatas[vector_id]

        self._active_count -= 1

        # Check for periodic compaction (15% threshold)
        if len(self.vectors) > 0 and (len(self.tombstones) / len(self.vectors)) >= 0.15:
            self.compact()

        return True

    def compact(self, force: bool = False) -> int:
        """Prune tombstoned nodes and rebuild graph connections.
        
        Args:
            force: If True, forces compaction even if under the 15% threshold.
            
        Returns:
            int: Count of purged tombstone nodes.
        """
        tomb_count = len(self.tombstones)
        total = len(self.vectors)
        if total == 0 or tomb_count == 0:
            return 0
        if not force and (tomb_count / total) < 0.15:
            return 0

        # Extract active items
        active_ids = [vid for vid in self.idx_to_id if vid is not None and self.id_to_idx.get(vid) not in self.tombstones]
        active_vectors = [self.vectors[self.id_to_idx[vid]] for vid in active_ids]
        active_metas = [self.metadatas.get(vid) for vid in active_ids]

        # Reset graph state
        self.layers = []
        self.entry_point = None
        self.max_level = -1
        self.vectors = []
        self.id_to_idx = {}
        self.idx_to_id = []
        self.metadatas = {}
        self.tombstones = set()
        self._active_count = 0

        # Re-insert active elements
        for vid, vec, meta in zip(active_ids, active_vectors, active_metas):
            self.insert(vid, vec, meta)

        return tomb_count

    def __len__(self) -> int:
        return self._active_count
