# Vector Database Performance Benchmark Report

**Dataset Scale**: 50,000 vectors  
**Vector Dimensionality**: 384 (`all-MiniLM-L6-v2`)  
**Evaluation Query Set**: 500 queries  
**Ground Truth Method**: Exact Brute-Force Cosine Similarity (Pure NumPy BLAS)  

### 🖥️ Test Environment & Hardware Specifications
- **CPU**: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4 Physical Cores, 8 vCPUs)
- **RAM**: 16 GB DDR4
- **Operating System**: Linux (x86_64, Ubuntu base)
- **Runtime**: Python 3.13.9, NumPy 2.5.2, PyTorch 2.14.0+cpu
> *Note: Latency and QPS figures are hardware-dependent. All metrics reported below were empirically measured on this machine under `SEED = 42`.*

---

## 1. Ground Truth Baseline (Brute-Force Linear Scan)

| Metric | Measured Value |
| :--- | :--- |
| **Recall@10** | **100.0%** (Mathematical Oracle) |
| **Mean Query Latency** | **2.95 ms** |
| **p50 Latency (Median)** | 2.40 ms |
| **p95 Latency** | 5.28 ms |
| **p99 Latency** | 9.09 ms |
| **Throughput (QPS)** | **338.6 queries/sec** |
| **Index Build Time** | 0.1106 s |

---

## 2. Inverted File Index (IVF-Flat) Sweep Results

*Index Configuration: $K = 256$ Voronoi partitions trained via custom Scratch K-Means.*

| $n_{\text{probe}}$ | Recall@10 | Mean Latency (ms) | p50 (ms) | p95 (ms) | p99 (ms) | QPS | Speedup vs BF |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **64.62%** | 0.22 ms | 0.16 ms | 0.40 ms | 0.64 ms | 4533.3 | **13.39x** |
| **2** | **76.20%** | 0.27 ms | 0.23 ms | 0.50 ms | 0.73 ms | 3767.3 | **11.13x** |
| **4** | **85.30%** | 0.70 ms | 0.54 ms | 1.00 ms | 5.83 ms | 1424.1 | **4.21x** |
| **8** | **93.14%** | 1.21 ms | 0.91 ms | 2.63 ms | 6.35 ms | 825.4 | **2.44x** |
| **16** | **96.44%** | 2.08 ms | 1.65 ms | 4.57 ms | 8.92 ms | 481.2 | **1.42x** |
| **32** | **97.96%** | 5.06 ms | 3.94 ms | 10.96 ms | 18.02 ms | 197.6 | 0.58x (slower) |
| **64** | **99.40%** | 10.91 ms | 9.42 ms | 19.77 ms | 23.12 ms | 91.6 | 0.27x (slower) |

---

## 3. Analysis: What Did the Approximation Cost?

1. **The Production "Sweet Spot" ($n_{\text{probe}} = 8$)**:
   - At $n_{\text{probe}} = 8$, IVF-Flat delivers **93.1% Recall@10** while executing in **1.21 ms** (**2.44x speedup** over Brute Force).
   - This provides the optimal balance between high semantic fidelity and sub-linear query acceleration for real-time applications.

2. **Ultra Low-Latency Mode ($n_{\text{probe}} = 1$)**:
   - Fastest possible query response: **0.22 ms** (**13.39x speedup**, 4533.3 QPS).
   - **Approximation Cost**: Recall drops to 64.6%, as candidates outside the single closest Voronoi cell are pruned without evaluation.

3. **High-Fidelity & The Algorithmic Crossover Point ($n_{\text{probe}} \ge 32$)**:
   - Probing 32 to 64 clusters yields near-perfect accuracy: **98.0%** at $n_{\text{probe}}=32$ and **99.4%** at $n_{\text{probe}}=64$.
   - **Crucial Engineering Insight**: At $n_{\text{probe}} \ge 32$, query latency increases to **5.06 ms** (0.58x — approximately 1.7x slower than Brute Force) and **10.91 ms** (3.7x slower).
   - **Why this crossover happens**: In Python, iterating through and gathering candidate indices across 32+ distinct posting lists, allocating candidate arrays, and re-indexing introduces overhead that eventually exceeds a single contiguous, heavily vectorized NumPy BLAS matrix multiplication (`X @ q`) over all 50,000 vectors. This empirically proves the classic ANN boundary: inverted file indices are most advantageous at low-to-medium probe counts ($n_{\text{probe}} \le 16$), beyond which linear scanning in contiguous memory is faster.

---

## 4. Visual Trade-Off Curve

![Trade-off Curve](../evaluation/results/tradeoff_curve.png)
