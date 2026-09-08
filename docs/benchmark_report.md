# Vector Database Performance Benchmark Report

**Dataset Scale**: 50,000 vectors  
**Vector Dimensionality**: 384 (`all-MiniLM-L6-v2`)  
**Evaluation Query Set**: 500 queries  
**Ground Truth Method**: Exact Brute-Force Cosine Similarity (Pure NumPy BLAS)  

### 🖥️ Test Environment & Hardware Specifications
- **CPU**: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (4 Physical Cores, 8 vCPUs)
- **RAM**: 8 GB DDR4
- **Operating System**: Linux (x86_64, Ubuntu base)
- **Runtime**: Python 3.13.9, NumPy 2.5.2, PyTorch 2.14.0+cpu
> *Note: Latency and QPS figures are hardware-dependent. All metrics reported below were empirically measured on this machine under `SEED = 42`.*

---

## 1. Ground Truth Baseline (Brute-Force Linear Scan)

| Metric | Measured Value |
| :--- | :--- |
| **Recall@10** | **100.0%** (Mathematical Oracle) |
| **Mean Query Latency** | **3.21 ms** |
| **p50 Latency (Median)** | 2.51 ms |
| **p95 Latency** | 5.75 ms |
| **p99 Latency** | 13.91 ms |
| **Throughput (QPS)** | **311.4 queries/sec** |
| **Index Build Time** | 0.2323 s |

---

## 2. Inverted File Index (IVF-Flat) Sweep Results

*Index Configuration: $K = 256$ Voronoi partitions trained via custom Scratch K-Means.*

| $n_{\text{probe}}$ | Recall@10 | Mean Latency (ms) | p50 (ms) | p95 (ms) | p99 (ms) | QPS | Speedup vs BF |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **64.62%** | 0.20 ms | 0.19 ms | 0.35 ms | 0.43 ms | 4907.1 | **15.76x** |
| **2** | **76.20%** | 0.29 ms | 0.25 ms | 0.49 ms | 0.70 ms | 3495.9 | **11.23x** |
| **4** | **85.30%** | 0.65 ms | 0.53 ms | 0.95 ms | 3.79 ms | 1534.5 | **4.93x** |
| **8** | **93.14%** | 1.23 ms | 0.91 ms | 3.60 ms | 5.96 ms | 813.7 | **2.61x** |
| **16** | **96.44%** | 3.20 ms | 2.08 ms | 8.07 ms | 17.53 ms | 312.9 | **1.00x** |
| **32** | **97.96%** | 7.64 ms | 6.02 ms | 16.16 ms | 23.54 ms | 130.8 | 0.42x (slower) |
| **64** | **99.40%** | 10.27 ms | 9.09 ms | 17.10 ms | 24.30 ms | 97.3 | 0.31x (slower) |

---

## 3. Analysis: What Did the Approximation Cost?

1. **The Production "Sweet Spot" ($n_{\text{probe}} = 8$)**:
   - At $n_{\text{probe}} = 8$, IVF-Flat delivers **93.1% Recall@10** while executing in **1.23 ms** (**2.61x speedup** over Brute Force).
   - This provides the optimal balance between high semantic fidelity and sub-linear query acceleration for real-time applications.

2. **Ultra Low-Latency Mode ($n_{\text{probe}} = 1$)**:
   - Fastest possible query response: **0.20 ms** (**15.76x speedup**, 4907.1 QPS).
   - **Approximation Cost**: Recall drops to 64.6%, as candidates outside the single closest Voronoi cell are pruned without evaluation.

3. **High-Fidelity & The Algorithmic Crossover Point ($n_{\text{probe}} \ge 32$)**:
   - Probing 32 to 64 clusters yields near-perfect accuracy: **98.0%** at $n_{\text{probe}}=32$ and **99.4%** at $n_{\text{probe}}=64$.
   - **Crucial Engineering Insight**: At $n_{\text{probe}} \ge 32$, query latency increases to **7.64 ms** (0.42x — approximately 2.4x slower than Brute Force) and **10.27 ms** (3.2x slower).
   - **Why this crossover happens**: In Python, iterating through and gathering candidate indices across 32+ distinct posting lists, allocating candidate arrays, and re-indexing introduces overhead that eventually exceeds a single contiguous, heavily vectorized NumPy BLAS matrix multiplication (`X @ q`) over all 50,000 vectors. This empirically proves the classic ANN boundary: inverted file indices are most advantageous at low-to-medium probe counts ($n_{\text{probe}} \le 16$), beyond which linear scanning in contiguous memory is faster.

---

## 4. Visual Trade-Off Curve

![Trade-off Curve](../evaluation/results/tradeoff_curve.png)
