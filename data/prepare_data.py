import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""Data Preparation Pipeline.
- Embeds 5,000 real text snippets from AG News.
- Embeds 500 real query texts from AG News.
- Expands to 50,000 vectors via clustered semantic perturbations (preserving lumpy geometry).
- Precomputes exact Ground Truth top-10 for 500 queries over all 50,000 vectors.
"""

import csv
import json
import time
import argparse
import numpy as np
import torch

from vectordb.distance import l2_normalize
from vectordb.brute_force import BruteForceIndex

CACHE_DIR = "cache"
DATA_DIR = "data/corpus"
AG_NEWS_CSV = os.path.join(DATA_DIR, "ag_news_train.csv")
VECTORS_PATH = os.path.join(CACHE_DIR, "vectors_50k.npy")
METADATA_PATH = os.path.join(CACHE_DIR, "corpus_metadata.json")
QUERIES_PATH = os.path.join(CACHE_DIR, "queries_500.npy")
QUERY_TEXTS_PATH = os.path.join(CACHE_DIR, "query_texts.json")
GROUND_TRUTH_PATH = os.path.join(CACHE_DIR, "ground_truth_top10.npy")

CATEGORY_MAP = {
    "1": "World",
    "2": "Sports",
    "3": "Business",
    "4": "Sci/Tech"
}


def load_raw_texts(num_corpus: int = 5000, num_queries: int = 500):
    """Load real sentences from AG News dataset."""
    if not os.path.exists(AG_NEWS_CSV):
        raise FileNotFoundError(f"Corpus file not found: {AG_NEWS_CSV}")

    corpus_texts = []
    corpus_metadata = []
    query_texts = []
    query_metadata = []

    print(f"Reading texts from {AG_NEWS_CSV}...")
    with open(AG_NEWS_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 3:
                continue
            cat_id, title, desc = row[0], row[1], row[2]
            category = CATEGORY_MAP.get(cat_id, "General")
            
            clean_text = f"{title.strip()}. {desc.strip()}".replace("\\", " ")
            meta = {
                "id": i,
                "title": title.strip(),
                "text": clean_text,
                "category": category
            }

            if len(corpus_texts) < num_corpus:
                corpus_texts.append(clean_text)
                corpus_metadata.append(meta)
            elif len(query_texts) < num_queries:
                query_texts.append(clean_text)
                query_metadata.append(meta)
            else:
                break

    print(f"Loaded {len(corpus_texts)} corpus texts and {len(query_texts)} query texts.")
    return corpus_texts, corpus_metadata, query_texts, query_metadata


def generate_embeddings(corpus_texts, query_texts, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 128):
    """Embed real texts using SentenceTransformer."""
    from sentence_transformers import SentenceTransformer
    
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"Loading embedding model '{model_name}' on CPU...")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(corpus_texts)} real corpus texts (batch_size={batch_size})...")
    t0 = time.time()
    corpus_vecs = model.encode(
        corpus_texts, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        normalize_embeddings=True
    )
    t_corpus = time.time() - t0
    print(f"5,000 real corpus texts encoded in {t_corpus:.2f}s ({len(corpus_texts)/t_corpus:.1f} sent/s).")

    print(f"Encoding {len(query_texts)} query texts...")
    t0 = time.time()
    query_vecs = model.encode(
        query_texts, 
        batch_size=batch_size, 
        show_progress_bar=False, 
        normalize_embeddings=True
    )
    print(f"500 query texts encoded in {time.time() - t0:.2f}s.")

    return np.asarray(corpus_vecs, dtype=np.float32), np.asarray(query_vecs, dtype=np.float32)


def expand_to_50k(base_vecs: np.ndarray, base_meta: list, target_size: int = 50000, seed: int = 42):
    """Expand real embeddings to at least 50,000 vectors with clustered semantic geometry."""
    print(f"Expanding {len(base_vecs)} real embeddings to {target_size} vectors with SEED={seed}...")
    rng = np.random.default_rng(seed)
    n_base, dim = base_vecs.shape
    repeats_needed = target_size // n_base
    remainder = target_size % n_base

    all_vecs = [base_vecs]
    all_meta = [m.copy() for m in base_meta]

    curr_id = n_base
    for rep in range(1, repeats_needed):
        # Add clustered perturbation (sigma=0.03) preserving manifold geometry
        noise = rng.normal(loc=0.0, scale=0.03, size=(n_base, dim)).astype(np.float32)
        augmented = l2_normalize(base_vecs + noise)
        all_vecs.append(augmented)
        for i in range(n_base):
            meta_copy = base_meta[i].copy()
            meta_copy["id"] = curr_id
            all_meta.append(meta_copy)
            curr_id += 1

    if remainder > 0:
        noise = rng.normal(loc=0.0, scale=0.03, size=(remainder, dim)).astype(np.float32)
        augmented = l2_normalize(base_vecs[:remainder] + noise)
        all_vecs.append(augmented)
        for i in range(remainder):
            meta_copy = base_meta[i].copy()
            meta_copy["id"] = curr_id
            all_meta.append(meta_copy)
            curr_id += 1

    final_vecs = np.vstack(all_vecs)
    print(f"Expanded to {len(final_vecs):,} vectors with unit L2 norm.")
    return final_vecs, all_meta


def compute_ground_truth(corpus_vecs: np.ndarray, query_vecs: np.ndarray, top_k: int = 10):
    """Compute exact ground truth top-k neighbors for 500 queries using BruteForceIndex."""
    print(f"Computing exact Ground Truth for {len(query_vecs)} queries over {len(corpus_vecs)} vectors...")
    bf_index = BruteForceIndex(dimension=corpus_vecs.shape[1])
    ids = list(range(len(corpus_vecs)))
    bf_index.batch_insert(corpus_vecs, ids)

    ground_truth_top10 = np.empty((len(query_vecs), top_k), dtype=np.int64)
    
    t0 = time.time()
    for q_idx, q_vec in enumerate(query_vecs):
        results = bf_index.search(q_vec, top_k=top_k)
        ground_truth_top10[q_idx] = [r[0] for r in results]

    dt = time.time() - t0
    print(f"Ground truth computed in {dt:.2f}s ({len(query_vecs)/dt:.1f} queries/s).")
    return ground_truth_top10


def main(force_recompute: bool = False):
    os.makedirs(CACHE_DIR, exist_ok=True)

    if (
        not force_recompute 
        and os.path.exists(VECTORS_PATH) 
        and os.path.exists(METADATA_PATH)
        and os.path.exists(QUERIES_PATH)
        and os.path.exists(GROUND_TRUTH_PATH)
    ):
        print("Cached vectors, metadata, and ground truth already exist. Skipping recomputation.")
        print(f"Vectors: {VECTORS_PATH} ({os.path.getsize(VECTORS_PATH)/(1024*1024):.2f} MB)")
        return

    corpus_texts, corpus_metadata, query_texts, query_metadata = load_raw_texts(
        num_corpus=5000, 
        num_queries=500
    )

    base_vecs, query_vecs = generate_embeddings(corpus_texts, query_texts)
    final_vecs, final_metadata = expand_to_50k(base_vecs, corpus_metadata, target_size=50000, seed=42)
    gt_top10 = compute_ground_truth(final_vecs, query_vecs, top_k=10)

    print("Saving cache files to disk...")
    np.save(VECTORS_PATH, final_vecs)
    np.save(QUERIES_PATH, query_vecs)
    np.save(GROUND_TRUTH_PATH, gt_top10)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(final_metadata, f)

    with open(QUERY_TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(query_texts, f)

    print("Data preparation successfully completed! All artifacts cached in cache/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force recompute embeddings")
    args = parser.parse_args()
    main(force_recompute=args.force)
