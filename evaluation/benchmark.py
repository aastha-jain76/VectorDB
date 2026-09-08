"""Comprehensive Benchmarking Suite for Custom Vector Database.
Measures Recall@10, latency (mean, p50, p95, p99), QPS, and speedup trade-off curves.
Strictly zero external ANN libraries.
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vectordb.brute_force import BruteForceIndex
from vectordb.ivf_flat import IVFFlatIndex

CACHE_DIR = "cache"
VECTORS_PATH = os.path.join(CACHE_DIR, "vectors_50k.npy")
QUERIES_PATH = os.path.join(CACHE_DIR, "queries_500.npy")
GROUND_TRUTH_PATH = os.path.join(CACHE_DIR, "ground_truth_top10.npy")
RESULTS_DIR = "evaluation/results"


def compute_recall_at_k(retrieved_ids: list, ground_truth_ids: list, k: int = 10) -> float:
    """Compute Recall@k for a single query."""
    retrieved_set = set(retrieved_ids[:k])
    gt_set = set(ground_truth_ids[:k])
    intersection = len(retrieved_set.intersection(gt_set))
    return intersection / float(k)


def run_benchmarks():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading cached vectors and ground truth...")
    if not os.path.exists(VECTORS_PATH) or not os.path.exists(QUERIES_PATH):
        raise FileNotFoundError("Cache not found. Run python3 data/prepare_data.py first.")

    vectors = np.load(VECTORS_PATH)
    queries = np.load(QUERIES_PATH)
    gt_top10 = np.load(GROUND_TRUTH_PATH)
    n_vectors = len(vectors)
    n_queries = len(queries)
    dim = vectors.shape[1]

    print(f"Loaded {n_vectors} vectors (dim={dim}), {n_queries} queries, and ground truth.")

    # 1. Benchmark Brute Force (Ground Truth Oracle)
    print("\n--- Benchmarking Brute-Force Index ---")
    bf_index = BruteForceIndex(dimension=dim)
    t0 = time.time()
    bf_index.batch_insert(vectors, list(range(n_vectors)))
    bf_build_time = time.time() - t0
    print(f"Brute-Force index built in {bf_build_time:.4f}s.")

    # Warm-up CPU caches and BLAS threads (unmeasured)
    for wq in queries[:10]:
        _ = bf_index.search(wq, top_k=10)

    bf_latencies = []
    for q in queries:
        t_start = time.perf_counter_ns()
        _ = bf_index.search(q, top_k=10)
        t_end = time.perf_counter_ns()
        bf_latencies.append((t_end - t_start) / 1e6)  # ms

    bf_latencies = np.array(bf_latencies)
    bf_mean = np.mean(bf_latencies)
    bf_p50 = np.median(bf_latencies)
    bf_p95 = np.percentile(bf_latencies, 95)
    bf_p99 = np.percentile(bf_latencies, 99)
    bf_qps = 1000.0 / bf_mean

    print(f"Brute-Force Latency: Mean={bf_mean:.2f}ms | p50={bf_p50:.2f}ms | p95={bf_p95:.2f}ms | p99={bf_p99:.2f}ms | QPS={bf_qps:.1f}")

    # 2. Benchmark IVF-Flat with varying n_probe
    print("\n--- Training and Building IVF-Flat Index (K=256) ---")
    ivf_index = IVFFlatIndex(n_clusters=256, n_probe=8, dimension=dim, random_state=42)
    t0 = time.time()
    ivf_index.build_index(vectors, list(range(n_vectors)))
    ivf_build_time = time.time() - t0
    print(f"IVF-Flat Index built in {ivf_build_time:.2f}s.")

    # Warm-up IVF search
    for wq in queries[:10]:
        _ = ivf_index.search(wq, top_k=10, n_probe=8)

    probe_list = [1, 2, 4, 8, 16, 32, 64]
    ivf_results = []

    for probe in probe_list:
        latencies = []
        recalls = []
        for i, q in enumerate(queries):
            t_start = time.perf_counter_ns()
            res = ivf_index.search(q, top_k=10, n_probe=probe)
            t_end = time.perf_counter_ns()

            latencies.append((t_end - t_start) / 1e6)
            retrieved_ids = [r[0] for r in res]
            recalls.append(compute_recall_at_k(retrieved_ids, gt_top10[i], k=10))

        lat_arr = np.array(latencies)
        mean_lat = float(np.mean(lat_arr))
        p50 = float(np.median(lat_arr))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        mean_recall = float(np.mean(recalls))
        qps = float(1000.0 / mean_lat)
        speedup = float(bf_mean / mean_lat)

        print(f"IVF-Flat (n_probe={probe:2d}) -> Recall@10: {mean_recall*100:5.2f}% | Latency: {mean_lat:5.2f}ms | Speedup: {speedup:5.2f}x | QPS: {qps:6.1f}")

        ivf_results.append({
            "n_probe": probe,
            "recall_at_10": mean_recall,
            "mean_latency_ms": mean_lat,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "qps": qps,
            "speedup": speedup
        })

    # Save summary report
    summary = {
        "dataset_size": n_vectors,
        "query_count": n_queries,
        "dimension": dim,
        "brute_force": {
            "mean_latency_ms": float(bf_mean),
            "p50_latency_ms": float(bf_p50),
            "p95_latency_ms": float(bf_p95),
            "p99_latency_ms": float(bf_p99),
            "qps": float(bf_qps),
            "build_time_s": float(bf_build_time)
        },
        "ivf_flat": {
            "n_clusters": 256,
            "build_time_s": float(ivf_build_time),
            "sweeps": ivf_results
        }
    }

    with open(os.path.join(RESULTS_DIR, "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plot Recall vs Latency trade-off curve
    plt.figure(figsize=(9, 5), dpi=150)
    recalls_pct = [r["recall_at_10"] * 100 for r in ivf_results]
    lats = [r["mean_latency_ms"] for r in ivf_results]

    plt.plot(lats, recalls_pct, marker='o', color='#1f77b4', linewidth=2.5, label='IVF-Flat Index')
    plt.axvline(x=bf_mean, color='#d62728', linestyle='--', label=f'Brute-Force ({bf_mean:.2f}ms, 100% Recall)')
    
    for r in ivf_results:
        plt.annotate(
            f"n_probe={r['n_probe']}\n({r['speedup']:.1f}x)",
            xy=(r["mean_latency_ms"], r["recall_at_10"] * 100),
            xytext=(r["mean_latency_ms"] + 0.1, r["recall_at_10"] * 100 - 3),
            fontsize=9
        )

    plt.title(f"Approximation Cost: Recall@10 vs Query Latency ({n_vectors:,} Vectors)", fontsize=13, fontweight='bold')
    plt.xlabel("Mean Query Latency (ms)", fontsize=11)
    plt.ylabel("Recall@10 (%)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    chart_path = os.path.join(RESULTS_DIR, "tradeoff_curve.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"\nTrade-off curve saved to {chart_path}")
    print("Benchmark completed successfully!")


if __name__ == "__main__":
    run_benchmarks()
