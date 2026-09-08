# Product Requirements Document (PRD)

## Project Title: Custom Vector Database Engine from Scratch
**Code Name**: `vector-from-scratch` (`VFS`)  
**Status**: Approved / In Development  
**Target Delivery**: Production Demo & Benchmark Report  

---

## 1. Executive Summary & Problem Context

Modern AI and Large Language Model (LLM) applications heavily rely on Vector Databases (such as Pinecone, FAISS, Milvus, Chroma, and Qdrant) for Retrieval-Augmented Generation (RAG), semantic search, and recommendation systems. However, most software engineers and data scientists treat these systems as black boxes.

The objective of this project is to build an end-to-end, high-performance Vector Database entirely from first principles in Python, **using only NumPy for linear algebra and arithmetic**, without relying on any vector search libraries or pre-packaged nearest neighbor modules (`faiss`, `pinecone`, `chromadb`, `sklearn.neighbors`, etc.).

The system must:
1. Provide an exact **Brute-Force baseline** serving as mathematically exact ground truth over $\ge 50,000$ high-dimensional vectors.
2. Implement custom **Approximate Nearest Neighbor (ANN)** indexing algorithms (**IVF-Flat** and **HNSW**) written by hand.
3. Expose a clean, production-grade API for lifecycle management (`insert`, `search`, `delete`).
4. Rigorously evaluate and quantify the **cost of approximation** (trade-off curves between Recall@10, Latency, and Throughput across 500 test queries).
5. Deliver an interactive, real-time natural language query demonstration suitable for video demonstration.

---

## 2. Core Goals & Success Criteria

| Objective | Target Metric | Verification Method |
| :--- | :--- | :--- |
| **Strict Independence** | 0 external vector DB / ANN dependencies | Codebase audit (no `faiss`, `sklearn.neighbors`, `chromadb`, etc.) |
| **Scale Baseline** | $\ge 50,000$ stored vectors | Ingest 50,000 embedded real/synthetic text vectors |
| **Ground Truth Accuracy** | 100% exact cosine similarity | Verified via dot product on $L_2$-normalized vectors |
| **Approximate Index Performance** | $\ge 5\times - 20\times$ speedup over Brute Force | Latency benchmarking across 500 query vectors |
| **ANN Quality / Recall** | $\ge 85\%$ - $95\%$ Recall@10 | Benchmark comparison against exact Brute-Force ground truth |
| **API Completeness** | Unified `insert`, `search`, `delete` | Automated unit and integration test suite |
| **Interactive Usability** | Sub-50ms natural language query response | Interactive UI / CLI demo searching 50k document corpus |

---

## 3. User Personas & Use Cases

### Personas
- **AI Systems Engineer / Evaluator**: Needs to understand how quantization, partitioning, and graph navigation behave under the hood.
- **Interviewer / Reviewer**: Wants to verify algorithm implementation correctness, mathematical rigor, and benchmark transparency.
- **End User / Demo Viewer**: Enters a natural language sentence and expects the most semantically relevant text snippet to return in milliseconds with visible speed and recall metrics.

### Key Use Cases
1. **Semantic Text Retrieval**: User enters a query (e.g., *"What causes global climate change?"*) and the system finds the top-$k$ nearest documents from a 50,000-text corpus.
2. **Ground Truth Validation**: Developer runs 500 evaluation queries to compute exact Recall@10, latency percentiles ($p50, p95, p99$), and speedup factors.
3. **Dynamic Index Updates**: Ingesting new vectors via `insert`, querying updated states, and removing vectors via `delete`.

---

## 4. Functional Requirements

### FR-1: Data Ingestion & Representation
- **FR-1.1**: The system shall support ingesting dense float vectors of dimension $D$ (e.g., $D=384$ for `all-MiniLM-L6-v2`).
- **FR-1.2**: Each vector shall be associated with an immutable unique ID (`int` or `str`) and optional metadata (original text snippet, category, source).
- **FR-1.3**: The system shall support a dataset of at least 50,000 vectors generated from a real text corpus (e.g., AG News, Wikipedia, or Question-Answering pairs) or clustered synthetic Gaussian mixtures.
- **FR-1.4**: All vectors shall be automatically $L_2$-normalized upon insertion so that Cosine Similarity is equivalent to standard Euclidean dot product $\langle \mathbf{u}, \mathbf{v} \rangle$.

### FR-2: Exact Brute-Force Engine (`BruteForceIndex`)
- **FR-2.1**: Implement exact matrix-multiplication-based linear scan over all active vectors using only NumPy.
- **FR-2.2**: Serve as the authoritative mathematical ground truth for evaluation.
- **FR-2.3**: Support dynamic vector appending (`insert`) and deletion via tombstone masking or row deletion (`delete`).

### FR-3: Hand-Rolled Approximate Index (`IVFFlatIndex`)
- **FR-3.1 (Clustering from Scratch)**: Implement $k$-means clustering entirely from scratch in NumPy (supporting $k$-means++ initialization and Lloyd's iterative centroid update).
- **FR-3.2 (Inverted Index / Voronoi Cells)**: Partition vector space into $K$ clusters (e.g., $K=256$ or $K=512$). Maintain posting lists mapping `cluster_id -> list[vector_id]`.
- **FR-3.3 (Multi-Probe Routing)**: During search, compute distances to $K$ centroids, select top $n_{\text{probe}}$ closest clusters, and exhaustively search only vectors within those partitions.
- **FR-3.4 (Tunable Trade-offs)**: Allow dynamic tuning of $n_{\text{probe}}$ (e.g., from 1 to 32) to observe recall vs latency curves.
- **FR-3.5 (Deletion)**: Prune vector ID and representation from the respective cluster posting list in $O(1)$ to $O(N_c)$ time.

### FR-4: Graph-Based Approximate Index (`HNSWIndex` - Advanced)
- **FR-4.1 (Hierarchical Graph)**: Multi-layer navigable small-world graph structure where upper layers allow rapid geometric skips and the bottom layer provides dense neighborhood navigation.
- **FR-4.2 (Greedy & Beam Search)**: Implement greedy layer descent and beam search (`efSearch`, `efConstruction`).
- **FR-4.3 (Explicit Deletion Handling)**: Address the mathematical complexity of graph node deletion. Provide soft tombstoning with periodic compaction and document why graph deletion is inherently non-trivial.

### FR-5: Core Unified API
Each index implementation must conform to an abstract base class `BaseVectorIndex`:
```python
class BaseVectorIndex(ABC):
    def insert(self, vector_id: int | str, vector: np.ndarray, metadata: dict | None = None) -> None: ...
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[int | str, float, dict]]: ...
    def delete(self, vector_id: int | str) -> bool: ...
    def __len__(self) -> int: ...
```

### FR-6: Evaluation & Benchmarking Engine
- **FR-6.1**: Compute ground truth top-10 neighbors for 500 test queries using Brute Force.
- **FR-6.2**: Calculate **Recall@10** for the approximate index:
  $$\text{Recall@10} = \frac{|\text{Top10}_{\text{ANN}} \cap \text{Top10}_{\text{GroundTruth}}|}{10}$$
- **FR-6.3**: Profile query latency ($p50, p95, p99$, mean) and calculate speedup factor:
  $$\text{Speedup} = \frac{\text{Latency}_{\text{BruteForce}}}{\text{Latency}_{\text{ANN}}}$$
- **FR-6.4**: Export benchmark results to structured JSON / CSV and markdown tables.

### FR-7: Interactive Demonstration Interface
- **FR-7.1**: Provide a lightweight user interface (Streamlit / CLI) where any arbitrary text query can be entered.
- **FR-7.2**: Real-time embedding of the query and instantaneous retrieval of the top matching pairs.
- **FR-7.3**: Visual side-by-side comparison of results, similarity scores, and execution timings.

---

## 5. Non-Functional Requirements (NFR)

- **NFR-1 (Zero External Indexing Dependencies)**: Zero imports of vector search libraries. No FAISS, Annoy, ScaNN, Pinecone, Chroma, Milvus, Qdrant, or `sklearn.neighbors`.
- **NFR-2 (Arithmetic Restriction)**: All mathematical operations (norms, matrix multiplications, sorting, argpartition) must use pure NumPy.
- **NFR-3 (Memory Efficiency)**: In-memory footprint for 50,000 384-dimensional float32 vectors must not exceed 200 MB ($50,000 \times 384 \times 4 \text{ bytes} \approx 76.8 \text{ MB}$).
- **NFR-4 (Reproducibility)**: Deterministic random seeds (`SEED = 42`) for clustering and synthetic vector generation.
- **NFR-5 (Modularity & Extensibility)**: Clean object-oriented architecture allowing drop-in comparisons between index types.

---

## 6. Out of Scope

- Distributed vector sharding across multiple physical nodes.
- GPU CUDA kernels (CPU execution with NumPy BLAS is the target).
- Heavy enterprise persistence engines (e.g. distributed Raft consensus). Single-node serialization is sufficient.
