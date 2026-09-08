# Project Rules & Engineering Invariants

## Project: Custom Vector Database Engine from Scratch
**Status**: Mandatory Compliance Guidelines  
**File**: `docs/rules.md`

---

## 1. Zero-External-Index / Anti-Cheating Invariants

The primary educational and technical objective of this project is to build an approximate nearest neighbor vector search engine strictly from first principles. The following rules are non-negotiable:

### 1.1 Forbidden Libraries
The following packages (and any wrapper libraries over them) are **strictly forbidden** from being imported or utilized for vector indexing, distance computation, clustering, or search:
- `faiss` (FAISS CPU/GPU)
- `pinecone-client`
- `chromadb`
- `qdrant-client`
- `weaviate-client`
- `pymilvus`
- `annoy` (Spotify Annoy)
- `scann` (Google ScaNN)
- `sklearn.neighbors` (`NearestNeighbors`, `KDTree`, `BallTree`, etc.)
- `scipy.spatial.KDTree` / `scipy.spatial.cKDTree`
- `scipy.cluster.vq.kmeans` or any black-box clustering wrappers

### 1.2 Permitted Libraries
- **NumPy (`numpy`)**: The standard arithmetic workhorse for vector manipulation, linear algebra (BLAS matrix multiplications), array slicing, and sorting (`np.dot`, `np.linalg.norm`, `np.argpartition`, `np.argsort`).
- **Python Standard Library**: `math`, `heapq`, `random`, `dataclasses`, `typing`, `time`, `json`, `os`, `sys`, `collections`.
- **Embedding Generation Only**: `sentence-transformers` and `torch` are **strictly permitted only for converting raw text into initial static embedding vectors** during data preparation. They are **never** to be used inside the index, clustering, or search algorithms.
- **Demo & Visualization**: `streamlit`, `matplotlib` (for generating benchmark plots).

---

## 2. Mathematical & Algorithmic Invariants

1. **Unit Normalization Invariant**:
   Every vector ingested into any index must be normalized to unit Euclidean length:
   $$\|\mathbf{x}\|_2 = 1.0 \pm 10^{-6}$$
   This ensures that Cosine Similarity is identically equivalent to the inner product $\mathbf{x} \cdot \mathbf{y}$, maximizing BLAS efficiency.

2. **K-Means from Scratch Invariant**:
   All clustering required for IVF-Flat must be implemented directly in NumPy without external ML libraries:
   - Initialized via $k$-means++ probabilistic distance weighting.
   - Updated via Lloyd's iteration with centroid re-normalization.
   - Handled against empty clusters via residual re-seeding.

3. **Ground Truth Integrity**:
   The `BruteForceIndex` is the mathematical golden standard. All recall calculations for approximate indices (IVF-Flat, HNSW) must compare against the top-$k$ output of `BruteForceIndex` over the identical vector space.

4. **Honest Deletion Disclosure**:
   - For IVF-Flat, deletion must be exact (removing the vector from the posting list).
   - For HNSW, graph deletion is mathematically awkward (risk of disconnecting paths). Deletion must be implemented via tombstoning (soft delete) with an explicit comment explaining the trade-off.

---

## 3. Performance & Benchmark Measurement Rules

1. **Pure Index Latency Isolation**:
   When measuring search latency, the benchmark must record only the index traversal and retrieval time. **Do not include the sentence-transformer embedding model inference time** in the vector database index latency.
2. **500-Query Ground Truth Requirement**:
   Evaluation must be executed across exactly 500 distinct query vectors against a database of at least 50,000 vectors.
3. **Reproducibility**:
   All random operations (dataset shuffling, k-means centroid initialization, HNSW layer generation) must seed with `SEED = 42`.

---

## 4. Code Quality & Architectural Standards

- **Type Safety**: All functions and methods must include Python 3.10+ type hints.
- **Documentation**: All public classes and methods must include Google-style or PEP 257 docstrings.
- **Edge Case Robustness**:
  - Searching on an empty index must return an empty list `[]`, never raise an uncaught exception.
  - Requesting $k > N$ (where $N$ is total vectors) must return all $N$ vectors gracefully without crashing.
  - Deleting an unindexed ID must return `False` rather than throwing a `KeyError`.
  - Zero-norm or NaN vectors must be safely caught and rejected with an informative error.
