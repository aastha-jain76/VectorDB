# Technical Design Document

## Project: Custom Vector Database Engine from Scratch
**Status**: Detailed Design Specification  
**File**: `docs/design.md`

---

## 1. Mathematical Foundations & Distance Metrics

### 1.1 Cosine Similarity and L2 Normalization
Given two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^D$, their cosine similarity is defined as:
$$\text{Sim}_{\cos}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^D u_i v_i}{\sqrt{\sum_{i=1}^D u_i^2} \sqrt{\sum_{i=1}^D v_i^2}}$$

By normalizing all incoming vectors to unit length ($\|\mathbf{u}\|_2 = 1, \|\mathbf{v}\|_2 = 1$) at ingestion time:
$$\mathbf{\hat{u}} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2 + \epsilon}, \quad \mathbf{\hat{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2 + \epsilon}$$
where $\epsilon = 10^{-12}$ prevents division by zero.

Under unit normalization, **Cosine Similarity simplifies strictly to the Dot Product**:
$$\text{Sim}_{\cos}(\mathbf{\hat{u}}, \mathbf{\hat{v}}) = \mathbf{\hat{u}} \cdot \mathbf{\hat{v}} = \sum_{i=1}^D \hat{u}_i \hat{v}_i$$

Furthermore, squared Euclidean distance on unit vectors is monotonically related to cosine similarity:
$$\|\mathbf{\hat{u}} - \mathbf{\hat{v}}\|_2^2 = \|\mathbf{\hat{u}}\|_2^2 + \|\mathbf{\hat{v}}\|_2^2 - 2(\mathbf{\hat{u}} \cdot \mathbf{\hat{v}}) = 2 - 2(\mathbf{\hat{u}} \cdot \mathbf{\hat{v}})$$
Thus, maximizing dot product is strictly equivalent to minimizing Euclidean distance.

---

## 2. Inverted File Index (IVF-Flat) Detailed Design

### 2.1 Custom K-Means Clustering (`KMeansScratch`)
To strictly adhere to the zero-dependency rule, $k$-means is built using pure NumPy:

1. **Initialization ($k$-means++)**:
   - Pick first centroid $\boldsymbol{\mu}_1$ uniformly at random from dataset $\mathbf{X}$.
   - For $k = 2, \dots, K$:
     - Compute squared distance $D(\mathbf{x}_i)^2 = \min_{j < k} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|_2^2$ for all $\mathbf{x}_i \in \mathbf{X}$.
     - Choose next centroid $\boldsymbol{\mu}_k$ with probability distribution:
       $$P(\mathbf{x}_i) = \frac{D(\mathbf{x}_i)^2}{\sum_{j} D(\mathbf{x}_j)^2}$$
2. **Lloyd's Iteration Loop**:
   - **Assignment Step**:
     $$\mathbf{a}_i = \arg\max_{k \in \{1, \dots, K\}} (\mathbf{x}_i \cdot \boldsymbol{\mu}_k)$$
     Computed via single matrix multiplication $\mathbf{S} = \mathbf{X} \mathbf{M}^T \in \mathbb{R}^{N \times K}$ and `np.argmax(S, axis=1)`.
   - **Centroid Update Step**:
     $$\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{|S_k|} \sum_{\mathbf{x}_i \in S_k} \mathbf{x}_i$$
     Normalize $\boldsymbol{\mu}_k^{(t+1)}$ to unit length.
   - **Empty Cluster Healing**: If any cluster has $|S_k| = 0$, re-initialize its centroid to the vector with the highest quantization residual.
   - **Convergence Check**: Terminate when centroid shift $\max_k \|\boldsymbol{\mu}_k^{(t+1)} - \boldsymbol{\mu}_k^{(t)}\| < 10^{-4}$ or maximum iterations (e.g. 25) is reached.

### 2.2 Data Structure of Inverted Lists
```
IVFFlatIndex
├── centroids: np.ndarray [K, D]          # Voronoi cell cluster centers
├── inverted_lists: dict[int, list[int]]  # cluster_id -> list of vector_ids
├── vector_store: dict[int, np.ndarray]   # vector_id -> normalized vector
└── metadata_store: dict[int, dict]       # vector_id -> arbitrary metadata
```

### 2.3 Search Mechanics ($n_{\text{probe}}$ Multi-Probing)
1. **Centroid Probing**: Compute similarity between query $\mathbf{q}$ and $K$ centroids:
   $$\mathbf{s}_{\text{centroids}} = \mathbf{C} \mathbf{q}^T \in \mathbb{R}^K$$
2. **Select Closest Clusters**: Identify top $n_{\text{probe}}$ centroids with largest dot product using `np.argpartition`.
3. **Candidate Gathering**: Concatenate candidate vectors from the selected $n_{\text{probe}}$ inverted lists.
4. **Candidate Scoring**: Compute dot product between $\mathbf{q}$ and candidate matrix $\mathbf{X}_{\text{candidates}} \in \mathbb{R}^{M \times D}$ where $M \approx \frac{n_{\text{probe}}}{K} N \ll N$.
5. **Top-$k$ Selection**: Return top-$k$ highest scoring IDs via `np.argpartition` / sort.

### 2.4 Deletion in IVF-Flat
- To delete vector $v_{\text{id}}$:
  1. Lookup vector in `vector_store` to find its cluster assignment, or search posting lists.
  2. Remove $v_{\text{id}}$ from the corresponding `inverted_lists[cluster_id]`.
  3. Delete from `vector_store` and `metadata_store`.
  4. Time complexity: $O(1)$ lookup + $O(L)$ list removal where $L \approx N/K$.

---

## 3. Hierarchical Navigable Small World (HNSW) Design

### 3.1 Graph Topology & Layering
HNSW arranges vectors into a hierarchy of proximity graphs $\mathcal{L}_0, \mathcal{L}_1, \dots, \mathcal{L}_{L_{\max}}$.
- **Layer Assignment**: Each inserted node is assigned a maximum layer $l$ using an exponential decay distribution:
  $$l = \lfloor -\ln(\text{uniform}(0, 1)) \cdot m_L \rfloor, \quad m_L = \frac{1}{\ln(M)}$$
  Where $M$ is the maximum number of bidirectional connections per node.
- **Top Layers**: Sparse graphs enabling long-range geometric hops in $O(\log N)$ steps.
- **Bottom Layer ($\mathcal{L}_0$)**: Contains all elements with dense local connections.

### 3.2 Search Traversal Algorithm
```
Algorithm: SEARCH_HNSW(query, top_k, efSearch)
1. ep = entry_point
2. For level = max_level down to 1:
3.     ep = GREEDY_SEARCH_LAYER(query, ep, ef=1, level)
4. W = BEAM_SEARCH_LAYER(query, ep, ef=efSearch, level=0)
5. Return top_k elements from W ordered by similarity
```

### 3.3 Deletion Mechanics in Graph Indices: Why Deletion is Awkward
Graph indices (like HNSW) are notoriously difficult to mutate via deletion:
1. **Connectivity Degradation**: Removing an internal node with high betweenness centrality can partition the graph or destroy the small-world navigation property.
2. **Edge Rewiring Overhead**: Reconnecting all neighbors of a deleted node to preserve the $M$-regularity requires re-running expensive local neighborhood heuristic searches.
3. **Design Solution**:
   - **Tombstone Masking (Soft Delete)**: Flag deleted nodes in a boolean mask. During search traversal, tombstoned nodes are used for graph routing (bridges) but excluded from final top-$k$ results.
   - **Periodic Compaction**: When tombstone ratio exceeds a threshold (e.g. 15%), a background compaction rewires orphaned edges and frees memory.

---

## 4. Unified API Interface Design

```python
from abc import ABC, abstractmethod
import numpy as np

class BaseVectorIndex(ABC):
    """Abstract Base Class defining the Vector Database contract."""

    @abstractmethod
    def insert(self, vector_id: int | str, vector: np.ndarray, metadata: dict | None = None) -> None:
        """Insert a single vector with optional metadata."""
        pass

    @abstractmethod
    def batch_insert(
        self, 
        vectors: np.ndarray, 
        ids: list[int | str], 
        metadatas: list[dict] | None = None
    ) -> None:
        """Batch insert multiple vectors for high throughput."""
        pass

    @abstractmethod
    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 10,
        **kwargs
    ) -> list[tuple[int | str, float, dict]]:
        """Search for top_k nearest neighbors. Returns list of (id, similarity, metadata)."""
        pass

    @abstractmethod
    def delete(self, vector_id: int | str) -> bool:
        """Remove a vector by ID. Returns True if found and removed."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the count of active vectors in the index."""
        pass
```

---

## 5. Storage, Memory Layout & Serialization

### 5.1 Memory Footprint (50,000 Vectors)
- Dimensions: $D = 384$
- Precision: IEEE 754 Single Precision (`float32`, 4 bytes/element)
- Matrix size: $50,000 \times 384 \times 4 \text{ bytes} = 76.8 \text{ MB}$.
- IVF Inverted lists (integer IDs): $50,000 \times 8 \text{ bytes} = 400 \text{ KB}$.
- Metadata storage (average 100 bytes/text): $\approx 5 \text{ MB}$.
- Total expected footprint: $< 100 \text{ MB}$.

### 5.2 Disk Persistence
The index state is saved to a compressed archive (`.npz` + `.json`):
- `centroids.npy`: Trained cluster centroids $[K, D]$.
- `vectors.npy`: Unit-normalized matrix $[N, D]$.
- `metadata.json`: Mapping of IDs to document snippets and cluster assignments.
