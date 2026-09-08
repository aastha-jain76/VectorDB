# Verification, Testing & Benchmarking Specification

## Project: Custom Vector Database Engine from Scratch
**Status**: Active Testing Specification  
**File**: `docs/testing.md`

---

## 1. Testing Strategy Overview

The testing strategy ensures both mathematical correctness and empirical performance of the custom vector database. It consists of three tiers:
1. **Unit Testing**: Testing mathematical primitives, clustering algorithms, and CRUD API contracts.
2. **Ground Truth Validation**: Verifying that Brute-Force produces exact results and that 500 test queries establish a deterministic baseline.
3. **Approximation Benchmarking**: Quantifying the Recall vs. Latency trade-offs of the approximate index (IVF-Flat / HNSW) against Ground Truth.

---

## 2. Unit Testing Suite (`tests/`)

### 2.1 Mathematical & Distance Primitives (`tests/test_distance.py`)
- **Unit Normalization**: Verify that vectors of arbitrary magnitude are scaled such that $\|\mathbf{x}\|_2 = 1.0 \pm 10^{-6}$.
- **Zero Vector Handling**: Verify that vectors with norm $< 10^{-12}$ raise a `ValueError` or fallback safely without `NaN` propagation.
- **Cosine Equivalence**: Confirm that `np.dot(u_norm, v_norm)` exactly equals $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$.

### 2.2 K-Means from Scratch (`tests/test_kmeans.py`)
- **Centroid Convergence**: Confirm that Lloyd's iteration reduces quantization error monotonically on synthetic clusters.
- **Cluster Count Invariant**: Verify that exactly $K$ non-empty centroids are returned.
- **Centroid Normalization**: Verify that all learned centroids remain unit-normalized.

### 2.3 Index API Contracts (`tests/test_indices.py`)
Both `BruteForceIndex` and `IVFFlatIndex` must pass identical contract test suites:
- **Insertion**: Single `insert` and `batch_insert` correctly increment `len(index)`.
- **Search Exactness**: Ingest small known vectors; verify that identical query vector returns itself as top-1 with similarity score $1.0 \pm 10^{-5}$.
- **Top-$k$ Bounds**: Ingest 5 items, query with $k=10$. System must return 5 items without out-of-bounds errors.
- **Empty Index**: Querying an empty index returns `[]`.
- **Deletion Integrity**:
  - Insert item $\text{ID}=101$.
  - Search returns $\text{ID}=101$.
  - Call `delete(101)` -> returns `True`.
  - Repeat search -> $\text{ID}=101$ is absent from results.
  - Call `delete(101)` again -> returns `False`.
  - `len(index)` decrements appropriately.

---

## 3. Ground Truth & 500-Query Benchmark Protocol

### 3.1 Setup
- **Database Size**: $N = 50,000$ vectors ($D = 384$).
- **Query Set**: $Q = 500$ vectors disjoint from or drawn from the corpus.
- **Ground Truth Generation**: Run `BruteForceIndex.search(q, top_k=10)` for each of the 500 queries. Store exact top-10 IDs in `ground_truth_top10.npy`.

### 3.2 Evaluation Metrics

#### A. Recall@10
Measures retrieval accuracy of the approximate index relative to ground truth:
$$\text{Recall@10} = \frac{1}{Q} \sum_{i=1}^Q \frac{|\mathcal{R}_{\text{ANN}}^{(i)} \cap \mathcal{R}_{\text{GT}}^{(i)}|}{10}$$
where $\mathcal{R}_{\text{ANN}}^{(i)}$ is the set of top-10 IDs returned by the approximate index, and $\mathcal{R}_{\text{GT}}^{(i)}$ is the set of top-10 IDs returned by Brute Force.

#### B. Latency Profiling
For each query, measure execution time using high-resolution monotonic clocks (`time.perf_counter_ns()`):
- **Mean Latency**: Average time per query in milliseconds.
- **p50 Latency**: Median response time.
- **p95 Latency**: 95th percentile worst-case latency.
- **p99 Latency**: 99th percentile worst-case latency.

#### C. Throughput & Speedup Factor
- **Queries Per Second (QPS)**:
  $$\text{QPS} = \frac{Q}{\sum_{i=1}^Q \text{Latency}(q_i)}$$
- **Speedup Factor**:
  $$\text{Speedup} = \frac{\text{Mean Latency}_{\text{BruteForce}}}{\text{Mean Latency}_{\text{ANN}}}$$

---

## 4. Hyperparameter Trade-Off Sweeps

To evaluate *"precisely what the approximation cost you"*, the test harness sweeps parameters across the 500 queries:

### IVF-Flat Parameter Sweep: $n_{\text{probe}}$
Vary $n_{\text{probe}} \in [1, 2, 4, 8, 16, 32, 64]$ (with fixed $K = 256$ clusters):
- As $n_{\text{probe}} \to 1$: Maximum speedup, lower recall.
- As $n_{\text{probe}} \to K$: Recall approaches $100\%$, speedup diminishes to brute-force levels.

Expected output visualization:
```
n_probe | Recall@10 | Mean Latency (ms) | Speedup vs BF
--------|-----------|-------------------|--------------
1       | 42.1%     | 0.35 ms           | 18.5x
4       | 78.4%     | 0.85 ms           | 7.6x
8       | 91.2%     | 1.45 ms           | 4.5x
16      | 96.8%     | 2.60 ms           | 2.5x
32      | 99.1%     | 4.80 ms           | 1.35x
BF (all)| 100.0%    | 6.50 ms           | 1.0x
```

---

## 5. Interactive Demo Verification Checklist

Before recording the working demo video:
- [ ] Database contains at least 50,000 real or synthetic text vectors.
- [ ] Any arbitrary text prompt can be submitted in the UI/CLI.
- [ ] System returns the top semantic match with its similarity score.
- [ ] Execution metrics (Brute Force time vs. Approximate Index time and Speedup) are displayed side-by-side.
- [ ] Deletion of an item from the active database can be demonstrated live.
