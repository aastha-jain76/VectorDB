"""Interactive Command-Line Vector Database Demonstration.
Allows real-time semantic search, index comparison, and deletion testing.
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vectordb.brute_force import BruteForceIndex
from vectordb.ivf_flat import IVFFlatIndex

CACHE_DIR = "cache"
VECTORS_PATH = os.path.join(CACHE_DIR, "vectors_50k.npy")
METADATA_PATH = os.path.join(CACHE_DIR, "corpus_metadata.json")


def main():
    print("=" * 65)
    print("      CUSTOM VECTOR DATABASE ENGINE FROM SCRATCH (CLI DEMO)")
    print("=" * 65)

    if not os.path.exists(VECTORS_PATH) or not os.path.exists(METADATA_PATH):
        print(f"Error: Cache not found at {VECTORS_PATH}.")
        print("Please run: python3 data/prepare_data.py")
        sys.exit(1)

    print("\n[1/3] Loading cached 50,000 vectors and metadata...")
    vectors = np.load(VECTORS_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
    n_vectors = len(vectors)
    dim = vectors.shape[1]
    print(f"Loaded {n_vectors:,} vectors with dimension {dim}.")

    # Build metadata lookup dict
    metadatas = {m["id"]: m for m in metadata_list}
    ids = list(range(n_vectors))

    print("\n[2/3] Initializing indexes (pure NumPy)...")
    t0 = time.time()
    bf_index = BruteForceIndex(dimension=dim)
    bf_index.batch_insert(vectors, ids, [metadatas[i] for i in ids])
    print(f"✓ Brute-Force Index ready ({time.time() - t0:.2f}s).")

    t0 = time.time()
    ivf_index = IVFFlatIndex(n_clusters=256, n_probe=8, dimension=dim, random_state=42)
    ivf_index.build_index(vectors, ids, [metadatas[i] for i in ids])
    print(f"✓ IVF-Flat Index ready ({time.time() - t0:.2f}s).")

    print("\n[3/3] Loading sentence encoder for real-time natural language query embedding...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓ Model loaded successfully.")

    print("\n" + "=" * 65)
    print("  System Ready! Enter any sentence to search across 50,000 documents.")
    print("  Type 'delete <id>' to remove a document.")
    print("  Type 'exit' or 'quit' to exit.")
    print("=" * 65 + "\n")

    while True:
        try:
            query = input("\n🔍 Enter search query > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            if query.lower().startswith("delete "):
                try:
                    del_id = int(query.split()[1])
                    bf_deleted = bf_index.delete(del_id)
                    ivf_deleted = ivf_index.delete(del_id)
                    if bf_deleted and ivf_deleted:
                        print(f"✅ Successfully deleted Document ID {del_id} from both indexes.")
                        print(f"   Remaining vectors: {len(ivf_index):,}")
                    else:
                        print(f"❌ Document ID {del_id} not found.")
                except Exception as e:
                    print(f"Error executing delete: {e}")
                continue

            # Real-time embedding
            t_emb_0 = time.perf_counter()
            q_vec = encoder.encode([query], normalize_embeddings=True)[0]
            t_emb = (time.perf_counter() - t_emb_0) * 1000.0

            # 1. Search Brute Force
            t0 = time.perf_counter()
            bf_results = bf_index.search(q_vec, top_k=5)
            t_bf = (time.perf_counter() - t0) * 1000.0

            # 2. Search IVF-Flat
            t0 = time.perf_counter()
            ivf_results = ivf_index.search(q_vec, top_k=5, n_probe=8)
            t_ivf = (time.perf_counter() - t0) * 1000.0

            speedup = t_bf / max(t_ivf, 0.001)

            print(f"\n⏱️  Timing Breakdown (Over {len(ivf_index):,} vectors):")
            print(f"   • Sentence Embedding Time : {t_emb:.2f} ms")
            print(f"   • Brute-Force Search Time : {t_bf:.2f} ms (100% exact ground truth)")
            print(f"   • IVF-Flat Search Time    : {t_ivf:.2f} ms (n_probe=8)")
            print(f"   ⚡ Speedup Factor         : {speedup:.1f}x FASTER than Brute Force\n")

            print(f"🏆 Top 5 Semantic Matches (IVF-Flat):")
            print("-" * 65)
            for rank, (vid, score, meta) in enumerate(ivf_results, start=1):
                category = meta.get("category", "N/A")
                title = meta.get("title", "No Title")
                text = meta.get("text", "")
                if len(text) > 120:
                    text = text[:117] + "..."
                print(f" [{rank}] ID: {vid:<6} | Cosine Sim: {score:.4f} | [{category}]")
                print(f"     Title: {title}")
                print(f"     Text : {text}")
                print("-" * 65)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
