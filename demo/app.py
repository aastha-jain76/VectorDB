"""Streamlit Interactive Demonstration for Custom Vector Database.
Built with pure NumPy for arithmetic without external vector search libraries.
"""

import os
import sys
import time
import json
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vectordb.brute_force import BruteForceIndex
from vectordb.ivf_flat import IVFFlatIndex

st.set_page_config(
    page_title="VectorDB From Scratch",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
VECTORS_PATH = os.path.join(CACHE_DIR, "vectors_50k.npy")
METADATA_PATH = os.path.join(CACHE_DIR, "corpus_metadata.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "evaluation", "results")
BENCHMARK_JSON = os.path.join(RESULTS_DIR, "benchmark_summary.json")
TRADEOFF_IMG = os.path.join(RESULTS_DIR, "tradeoff_curve.png")


@st.cache_resource(show_spinner="Loading 50,000 Vectors & Training IVF-Flat Index...")
def load_database_and_indices():
    if not os.path.exists(VECTORS_PATH) or not os.path.exists(METADATA_PATH):
        return None, None, None, None, None

    vectors = np.load(VECTORS_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    n_vectors = len(vectors)
    dim = vectors.shape[1]
    metadatas = {m["id"]: m for m in metadata_list}
    ids = list(range(n_vectors))

    # Initialize Brute Force
    bf_index = BruteForceIndex(dimension=dim)
    bf_index.batch_insert(vectors, ids, [metadatas[i] for i in ids])

    # Initialize IVF-Flat
    ivf_index = IVFFlatIndex(n_clusters=256, n_probe=8, dimension=dim, random_state=42)
    ivf_index.build_index(vectors, ids, [metadatas[i] for i in ids])

    # Load SentenceTransformer
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    return bf_index, ivf_index, encoder, metadatas, n_vectors


def main():
    st.title("⚡ Vector Database from Scratch")
    st.markdown(
        """
        *A high-performance Vector Database engine written entirely from first principles in **pure NumPy**.*  
        **No FAISS. No Pinecone. No Chroma. No `sklearn.neighbors`.**
        """
    )

    bf_index, ivf_index, encoder, metadatas, total_vectors = load_database_and_indices()

    if bf_index is None:
        st.error("⚠️ Cached vectors not found. Please run `python3 data/prepare_data.py` first to generate embeddings.")
        return

    # Sidebar Controls
    st.sidebar.header("🛠️ Database Controls")
    st.sidebar.info(f"**Indexed Vectors**: {len(ivf_index):,} \n\n**Dimension**: 384\n\n**Voronoi Partitions (K)**: 256")

    top_k = st.sidebar.slider("Top-K Nearest Neighbors", min_value=1, max_value=20, value=5)
    n_probe = st.sidebar.slider(
        "IVF-Flat Probed Clusters (n_probe)", 
        min_value=1, 
        max_value=64, 
        value=8,
        help="Higher n_probe increases Recall towards 100% at the cost of slight latency."
    )

    # Deletion Panel in Sidebar
    st.sidebar.subheader("🗑️ Vector Deletion Test")
    del_id_input = st.sidebar.number_input("Vector ID to Delete", min_value=0, max_value=total_vectors + 1000, value=0)
    if st.sidebar.button("Delete Vector"):
        bf_ok = bf_index.delete(int(del_id_input))
        ivf_ok = ivf_index.delete(int(del_id_input))
        if bf_ok and ivf_ok:
            st.sidebar.success(f"Deleted ID {del_id_input}! Remaining: {len(ivf_index):,}")
        else:
            st.sidebar.warning(f"ID {del_id_input} not found or already deleted.")

    # Tabs
    tab_search, tab_benchmarks, tab_architecture = st.tabs(["🔍 Semantic Search Demo", "📊 Benchmark & Trade-offs", "📐 Architecture & Math"])

    with tab_search:
        # Sample prompt buttons
        st.write("**Quick Query Suggestions:**")
        cols = st.columns(4)
        sample_queries = [
            "Space exploration and rocket launch missions",
            "Wall Street tech stocks and quarterly earnings",
            "Olympic games gold medal athletics",
            "Cybersecurity threats and software vulnerabilities"
        ]
        chosen_sample = None
        for i, sq in enumerate(sample_queries):
            if cols[i].button(sq, key=f"sq_{i}"):
                chosen_sample = sq

        query_input = st.text_input(
            "Enter any natural language statement:",
            value=chosen_sample if chosen_sample else "Artificial intelligence models and neural network breakthroughs"
        )

        if st.button("🚀 Run Vector Search", type="primary") or query_input:
            # 1. Embed query
            t_emb_0 = time.perf_counter()
            q_vec = encoder.encode([query_input], normalize_embeddings=True)[0]
            t_emb = (time.perf_counter() - t_emb_0) * 1000.0

            # 2. Search Brute Force (Ground Truth)
            t_bf_0 = time.perf_counter()
            bf_results = bf_index.search(q_vec, top_k=top_k)
            t_bf = (time.perf_counter() - t_bf_0) * 1000.0

            # 3. Search IVF-Flat (Approximate)
            t_ivf_0 = time.perf_counter()
            ivf_results = ivf_index.search(q_vec, top_k=top_k, n_probe=n_probe)
            t_ivf = (time.perf_counter() - t_ivf_0) * 1000.0

            speedup = t_bf / max(t_ivf, 0.001)

            # Calculate agreement
            bf_ids = set([r[0] for r in bf_results])
            ivf_ids = set([r[0] for r in ivf_results])
            agreement = (len(bf_ids.intersection(ivf_ids)) / float(top_k)) * 100.0

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Brute Force (Ground Truth)", f"{t_bf:.2f} ms", "Exact 100%")
            m2.metric(f"IVF-Flat (n_probe={n_probe})", f"{t_ivf:.2f} ms", f"{speedup:.1f}x Faster")
            m3.metric("Top-K Agreement", f"{agreement:.1f}%")
            m4.metric("Sentence Embedding", f"{t_emb:.2f} ms")

            st.divider()

            # Side-by-side results
            c_left, c_right = st.columns(2)

            with c_left:
                st.subheader(f"⚡ IVF-Flat Results ({t_ivf:.2f} ms)")
                for rank, (vid, score, meta) in enumerate(ivf_results, start=1):
                    cat = meta.get("category", "General")
                    title = meta.get("title", "No Title")
                    text = meta.get("text", "")
                    st.markdown(
                        f"""
                        <div style="background-color: #1e2530; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #00c0f2;">
                            <div style="display: flex; justify-content: space-between;">
                                <strong>#{rank} | ID: {vid}</strong>
                                <span style="background-color: #00c0f2; color: black; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">{cat}</span>
                            </div>
                            <div style="color: #64b5f6; font-size: 13px; margin: 4px 0;">Cosine Similarity: <b>{score:.4f}</b></div>
                            <div style="font-weight: 600; margin-bottom: 4px;">{title}</div>
                            <div style="font-size: 13px; color: #cfd8dc;">{text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with c_right:
                st.subheader(f"🎯 Brute Force Results ({t_bf:.2f} ms)")
                for rank, (vid, score, meta) in enumerate(bf_results, start=1):
                    cat = meta.get("category", "General")
                    title = meta.get("title", "No Title")
                    text = meta.get("text", "")
                    st.markdown(
                        f"""
                        <div style="background-color: #1e2530; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #ff4b4b;">
                            <div style="display: flex; justify-content: space-between;">
                                <strong>#{rank} | ID: {vid}</strong>
                                <span style="background-color: #ff4b4b; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">{cat}</span>
                            </div>
                            <div style="color: #ff8a80; font-size: 13px; margin: 4px 0;">Cosine Similarity: <b>{score:.4f}</b></div>
                            <div style="font-weight: 600; margin-bottom: 4px;">{title}</div>
                            <div style="font-size: 13px; color: #cfd8dc;">{text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    with tab_benchmarks:
        st.subheader("📈 500-Query Benchmark & Approximation Cost")
        if os.path.exists(TRADEOFF_IMG):
            st.image(TRADEOFF_IMG, caption="Recall@10 vs Query Latency Trade-off across 500 Queries")

        if os.path.exists(BENCHMARK_JSON):
            with open(BENCHMARK_JSON, "r") as f:
                b_data = json.load(f)
            
            st.write("### Benchmark Summary Table (500 Queries against 50,000 Vectors)")
            sweeps = b_data["ivf_flat"]["sweeps"]
            table_rows = []
            for s in sweeps:
                table_rows.append({
                    "n_probe": s["n_probe"],
                    "Recall@10 (%)": f"{s['recall_at_10']*100:.2f}%",
                    "Mean Latency (ms)": f"{s['mean_latency_ms']:.2f} ms",
                    "p50 Latency (ms)": f"{s['p50_latency_ms']:.2f} ms",
                    "p95 Latency (ms)": f"{s['p95_latency_ms']:.2f} ms",
                    "QPS": f"{s['qps']:.1f}",
                    "Speedup vs BF": f"{s['speedup']:.2f}x"
                })
            try:
                st.dataframe(table_rows, width="stretch")
            except TypeError:
                st.dataframe(table_rows, use_container_width=True)

    with tab_architecture:
        st.subheader("📐 System Design & Mathematical Principles")
        st.markdown(
            r"""
            ### 1. Mathematical Ground Truth
            Given normalized vectors $\|\mathbf{u}\|_2 = 1$, cosine similarity simplifies to the Euclidean dot product:
            $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v}$$
            
            ### 2. Scratch K-Means & Voronoi Partitioning
            - Centroids are initialized via $k$-means++ probabilistic distance distribution.
            - Iterative Lloyd's update normalizes centroids after each epoch.
            - Search probes only the $n_{\text{probe}}$ nearest clusters out of $K=256$, reducing the search space from $N$ down to $\frac{n_{\text{probe}}}{K} N$.
            
            ### 3. Exact CRUD & Compaction
            - Inverted lists prune deleted vector pointers in $O(1)$ to $O(N/K)$.
            - Primary memory arrays use swap-and-pop compaction to avoid memory reallocation overhead.
            """
        )


if __name__ == "__main__":
    main()
