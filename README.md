# ⚡ Custom Vector Database Engine from Scratch

> **"No Pinecone. No FAISS. No Chroma. No sklearn.neighbors."**  
> A high-performance Approximate Nearest Neighbor (ANN) Vector Database written entirely from first principles using **pure NumPy for arithmetic**.

---

## 📌 Features & Highlights

- **Pure First-Principles Implementation**: Zero imports of vector search engines (`faiss`, `pinecone`, `chromadb`, `sklearn.neighbors`, `scipy.spatial.KDTree`). All matrix operations, distance calculations, and clustering are written with pure NumPy.
- **Dual Index Architectures**:
  1. **Brute-Force Index (`BruteForceIndex`)**: Exact $O(N)$ linear scan serving as the mathematical ground truth.
  2. **Inverted File Index (`IVFFlatIndex`)**: Voronoi cell partitioning via custom Spherical $K$-Means clustering written from scratch, supporting dynamic multi-probe routing ($n_{\text{probe}}$).
  3. **Hierarchical Graph Index (`HNSWIndex`)**: Multi-layer skip-graph navigation with beam search and tombstone deletion.
- **Unified Vector Database API**:
  - `insert(vector_id, vector, metadata)`
  - `search(query_vector, top_k, ...)`
  - `delete(vector_id)` (with exact swap-and-pop list compaction for IVF-Flat and tombstoning for HNSW)
- **50,000 Vector Scale with Real Semantics**:
  - Encoded 5,000 real sentences from the AG News corpus using `all-MiniLM-L6-v2` (`dim=384`).
  - Expanded to 50,000 vectors via Gaussian manifold perturbations ($\sigma=0.03$) with `SEED=42` to benchmark 50k scale on CPU while maintaining realistic anisotropic semantic clusters.
- **Rigorous Evaluation Suite**:
  - 500 test queries evaluated against exact ground truth.
  - Automated measurement of Recall@10, latency percentiles ($p50, p95, p99$), QPS, and speedup trade-off curves.
- **Interactive Demonstrator & REST API**:
  - **FastAPI REST Service** (`api/server.py`) exposing `/search`, `/insert`, `/delete`, and `/stats` with Swagger OpenAPI documentation (`http://localhost:8000/docs`).
  - Sleek **Streamlit Web UI** (`demo/app.py`) for live natural language query search, side-by-side comparison, and real-time deletion testing.
  - Interactive **CLI Terminal Demo** (`demo/cli_demo.py`).

---

## 📂 Project Structure

```
IT_Geeks/
├── vectordb/                # Core Vector Database Engine (Pure NumPy)
│   ├── __init__.py
│   ├── base.py              # BaseVectorIndex abstract interface
│   ├── distance.py          # L2 normalization, cosine similarity, euclidean dist
│   ├── brute_force.py       # Exact linear scan ground truth
│   ├── ivf_flat.py          # Inverted file index with Voronoi cells & posting lists
│   ├── hnsw.py              # Hierarchical Navigable Small World graph index
│   └── kmeans.py            # Scratch K-Means (k-means++ init & Lloyd's iteration)
├── api/
│   └── server.py            # Production FastAPI REST microservice
├── data/
│   ├── prepare_data.py      # Embeds real text corpus and precomputes ground truth
│   └── corpus/              # AG News CSV dataset
├── evaluation/
│   ├── benchmark.py         # 500-query benchmark runner & trade-off curve plotter
│   ├── reporter.py          # Formats benchmark summary into Markdown
│   └── results/             # Benchmark JSONs and tradeoff_curve.png
├── demo/
│   ├── app.py               # Streamlit interactive Web UI
│   └── cli_demo.py          # Terminal interactive CLI demo
├── tests/
│   ├── test_api.py          # FastAPI REST endpoint integration tests
│   ├── test_distance.py     # Math & distance primitive unit tests
│   ├── test_kmeans.py       # Scratch K-Means convergence tests
│   └── test_indices.py      # CRUD lifecycle tests for all indices
├── run_api.sh               # One-click FastAPI server launcher
├── run_benchmark.sh         # One-click benchmark runner
├── run_demo.sh              # One-click demo launcher
└── README.md
```

---

## 🚀 Quick Start

### 1. Run Unit Tests
Verify that all mathematical operations and index CRUD APIs work properly:
```bash
pytest tests/ -v
```

### 2. Prepare Data & Vectors (Runs once)
Generates the 50,000-vector dataset from real news sentences and computes ground truth for 500 queries:
```bash
python3 data/prepare_data.py
```

### 3. Run Benchmark Suite
Executes 500 queries against both Brute-Force and IVF-Flat across varying $n_{\text{probe}}$ values, saving metrics and the trade-off plot:
```bash
./run_benchmark.sh
```

### 4. Launch FastAPI REST Server
Start the production-grade HTTP REST service with interactive Swagger UI:
```bash
./run_api.sh
```
Or directly:
```bash
python3 -m uvicorn api.server:app --port 8000
```
Interactive Swagger docs: `http://localhost:8000/docs`

### 5. Launch Interactive UI Demo
#### Web Application (Recommended for Demo Video):
```bash
./run_demo.sh
```
Or directly:
```bash
streamlit run demo/app.py
```

#### Terminal CLI Demo:
```bash
./run_demo.sh --cli
```

---

## 📊 Benchmark Summary: The Approximation Cost

> **🖥️ Test Environment & Hardware Profile**:  
> Measured on **11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz** (4 Cores, 8 vCPUs), 16 GB RAM, Ubuntu Linux, Python 3.13.9, NumPy 2.5.2, PyTorch 2.14.0 (CPU), `SEED = 42`. Latency and QPS depend on hardware.

Across **500 evaluation queries** searching over **50,000 vectors** ($D=384$):

| Index | Configuration | Recall@10 | Mean Latency | Speedup vs BF | QPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Brute-Force** | Ground Truth (BLAS Matmul) | **100.0%** | **2.95 ms** | 1.00x (Baseline) | **338.6** |
| **IVF-Flat** | $n_{\text{probe}} = 1$ | 64.62% | **0.22 ms** | **13.39x Faster** | 4,533.3 |
| **IVF-Flat** | $n_{\text{probe}} = 2$ | 76.20% | **0.27 ms** | **11.13x Faster** | 3,767.3 |
| **IVF-Flat** | $n_{\text{probe}} = 4$ | 85.30% | **0.70 ms** | **4.21x Faster** | 1,424.1 |
| **IVF-Flat** | $n_{\text{probe}} = 8$ *(Sweet Spot)* | **93.14%** | **1.21 ms** | **2.44x Faster** | 825.4 |
| **IVF-Flat** | $n_{\text{probe}} = 16$ | **96.44%** | **2.08 ms** | **1.42x Faster** | 481.2 |
| **IVF-Flat** | $n_{\text{probe}} = 32$ | **97.96%** | 5.06 ms | 0.58x *(1.7x slower)* | 197.6 |
| **IVF-Flat** | $n_{\text{probe}} = 64$ | **99.40%** | 10.91 ms | 0.27x *(3.7x slower)* | 91.6 |

> **Key Finding (The Crossover Point)**:  
> On our CPU, IVF-Flat achieves **93.14% Recall@10** at **2.44x speedup** ($n_{\text{probe}}=8$). However, at $n_{\text{probe}} \ge 32$, the Python-level overhead of aggregating and slicing 32+ posting lists exceeds a single continuous NumPy BLAS matrix multiplication on 50k vectors, marking the exact boundary where linear scan becomes faster than inverted index lookups.

---

## 🎥 Recording Your Demo Video

1. Start the Streamlit app: `streamlit run demo/app.py`
2. Open your browser at `http://localhost:8501`.
3. Type any query (e.g. *"space shuttle rocket launch"* or *"Wall Street stock market gains"*).
4. Highlight the **Top Matches**, the **Brute Force vs IVF-Flat timing**, and the **Speedup Multiplier**.
5. Test vector deletion live using the **Vector Deletion Test** panel in the sidebar to prove real-time deletion!
