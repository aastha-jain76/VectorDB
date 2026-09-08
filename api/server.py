"""FastAPI REST Service for Custom Vector Database.
Exposes insert, search, delete, stats, and health endpoints over pure NumPy indices.
"""

import os
import sys
import time
import json
from typing import Any, Dict, List, Optional, Union
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vectordb.brute_force import BruteForceIndex
from vectordb.ivf_flat import IVFFlatIndex
from vectordb.distance import l2_normalize

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
VECTORS_PATH = os.path.join(CACHE_DIR, "vectors_50k.npy")
METADATA_PATH = os.path.join(CACHE_DIR, "corpus_metadata.json")

from contextlib import asynccontextmanager

def initialize_database(force_synthetic: bool = False):
    global bf_index, ivf_index, dimension
    print("Initializing Vector Database API...")

    if not force_synthetic and os.path.exists(VECTORS_PATH) and os.path.exists(METADATA_PATH):
        vectors = np.load(VECTORS_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)

        n_vectors = len(vectors)
        dimension = vectors.shape[1]
        metadatas = {m["id"]: m for m in metadata_list}
        ids = list(range(n_vectors))

        bf_index = BruteForceIndex(dimension=dimension)
        bf_index.batch_insert(vectors, ids, [metadatas.get(i) for i in ids])

        ivf_index = IVFFlatIndex(n_clusters=256, n_probe=8, dimension=dimension, random_state=42)
        ivf_index.build_index(vectors, ids, [metadatas.get(i) for i in ids])
        print(f"Loaded {n_vectors:,} vectors into Brute-Force and IVF-Flat indices.")
    else:
        # If cache is missing (e.g. fresh clone before data/prepare_data.py is run),
        # initialize with a small valid trained index (50 synthetic vectors, dim=384)
        # so all API endpoints, searches, and test suites are fully functional out-of-the-box.
        dimension = 384
        rng = np.random.default_rng(42)
        n_samples = 50
        synthetic_vecs = l2_normalize(rng.normal(size=(n_samples, dimension)).astype(np.float32))
        synthetic_ids = [f"sample_{i}" for i in range(n_samples)]
        synthetic_meta = [
            {"id": f"sample_{i}", "text": f"Synthetic news article {i} about technology and AI", "category": "Sci/Tech"}
            for i in range(n_samples)
        ]

        bf_index = BruteForceIndex(dimension=dimension)
        bf_index.batch_insert(synthetic_vecs, synthetic_ids, synthetic_meta)

        ivf_index = IVFFlatIndex(n_clusters=4, n_probe=2, dimension=dimension, random_state=42)
        ivf_index.build_index(synthetic_vecs, synthetic_ids, synthetic_meta)
        print("Initialized Vector Database with small trained synthetic index (run data/prepare_data.py for full 50k corpus).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="VectorDB From Scratch API",
    description="Lightweight Educational REST API over pure NumPy Brute-Force and IVF-Flat vector indices.",
    version="1.0.0",
    lifespan=lifespan
)

# Global in-memory state
bf_index: Optional[BruteForceIndex] = None
ivf_index: Optional[IVFFlatIndex] = None
encoder = None
dimension: int = 384


def get_encoder():
    global encoder
    if encoder is None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return encoder


# Pydantic Request & Response Models
class SearchRequest(BaseModel):
    query: Optional[str] = Field(None, description="Natural language statement to search")
    vector: Optional[List[float]] = Field(None, description="Pre-computed dense vector")
    index: str = Field("ivf", description="Index to query: 'ivf' (approximate) or 'brute_force' (exact)")
    top_k: int = Field(10, ge=1, le=100, description="Number of nearest neighbors to retrieve")
    n_probe: Optional[int] = Field(8, ge=1, le=256, description="Voronoi clusters to probe for IVF")


class SearchResultItem(BaseModel):
    id: Union[int, str]
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: Optional[str]
    index_used: str
    top_k: int
    latency_ms: float
    total_vectors_indexed: int
    candidate_vectors_evaluated: int
    results: List[SearchResultItem]


class InsertRequest(BaseModel):
    id: Union[int, str] = Field(..., description="Unique vector ID")
    text: Optional[str] = Field(None, description="Text to encode and index")
    vector: Optional[List[float]] = Field(None, description="Pre-computed vector of dimension D")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary metadata attributes")


class GenericResponse(BaseModel):
    status: str
    message: str
    total_vectors: int


# Endpoints
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "VectorDB from Scratch",
        "indices_ready": bf_index is not None and ivf_index is not None
    }


@app.get("/stats")
def get_stats():
    total_bf = len(bf_index) if bf_index else 0
    total_ivf = len(ivf_index) if ivf_index else 0
    return {
        "total_vectors": total_ivf,
        "dimension": dimension,
        "indices": {
            "brute_force": {"active_vectors": total_bf, "type": "exact_linear_scan"},
            "ivf_flat": {"active_vectors": total_ivf, "n_clusters": 256, "default_n_probe": 8}
        }
    }


@app.post("/search", response_model=SearchResponse)
def search_vectors(req: SearchRequest):
    if bf_index is None or ivf_index is None:
        raise HTTPException(status_code=503, detail="Indices not initialized")

    # Resolve query vector
    if req.vector is not None:
        q_vec = np.array(req.vector, dtype=np.float32)
        if len(q_vec) != dimension:
            raise HTTPException(status_code=400, detail=f"Vector length {len(q_vec)} != dimension {dimension}")
    elif req.query is not None:
        enc = get_encoder()
        q_vec = enc.encode([req.query], normalize_embeddings=True)[0]
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'query' string or 'vector' float array")

    idx_type = req.index.lower()
    t0 = time.perf_counter()

    if idx_type in ("ivf", "ivf_flat", "approximate"):
        probe = req.n_probe if req.n_probe is not None else 8
        raw_results = ivf_index.search(q_vec, top_k=req.top_k, n_probe=probe)
        index_used = f"ivf_flat (n_probe={probe})"
        candidates_evaluated = getattr(ivf_index, "last_candidate_count", len(raw_results))
    elif idx_type in ("brute_force", "bf", "exact"):
        raw_results = bf_index.search(q_vec, top_k=req.top_k)
        index_used = "brute_force (exact ground truth)"
        candidates_evaluated = getattr(bf_index, "last_candidate_count", len(bf_index))
    else:
        raise HTTPException(status_code=400, detail=f"Unknown index type: {req.index}. Use 'ivf' or 'brute_force'.")

    latency = (time.perf_counter() - t0) * 1000.0

    items = [
        SearchResultItem(id=vid, score=float(score), metadata=meta)
        for vid, score, meta in raw_results
    ]

    return SearchResponse(
        query=req.query,
        index_used=index_used,
        top_k=req.top_k,
        latency_ms=round(latency, 3),
        total_vectors_indexed=len(ivf_index),
        candidate_vectors_evaluated=candidates_evaluated,
        results=items
    )


@app.post("/insert", response_model=GenericResponse)
def insert_vector(req: InsertRequest):
    if bf_index is None or ivf_index is None:
        raise HTTPException(status_code=503, detail="Indices not initialized")

    if req.vector is not None:
        vec = np.array(req.vector, dtype=np.float32)
        if len(vec) != dimension:
            raise HTTPException(status_code=400, detail=f"Vector length {len(vec)} != dimension {dimension}")
    elif req.text is not None:
        enc = get_encoder()
        vec = enc.encode([req.text], normalize_embeddings=True)[0]
        if "text" not in req.metadata:
            req.metadata["text"] = req.text
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'text' or 'vector'")

    try:
        bf_index.insert(req.id, vec, req.metadata)
        ivf_index.insert(req.id, vec, req.metadata)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return GenericResponse(
        status="success",
        message=f"Vector ID '{req.id}' inserted into BruteForce and IVF-Flat indices.",
        total_vectors=len(ivf_index)
    )


@app.delete("/vectors/{vector_id}", response_model=GenericResponse)
def delete_vector(vector_id: Union[int, str]):
    if bf_index is None or ivf_index is None:
        raise HTTPException(status_code=503, detail="Indices not initialized")

    # Try integer conversion if possible
    del_id = vector_id
    if isinstance(vector_id, str) and vector_id.isdigit():
        del_id = int(vector_id)

    bf_ok = bf_index.delete(del_id)
    ivf_ok = ivf_index.delete(del_id)

    # Fallback to string if int failed
    if not (bf_ok or ivf_ok) and del_id != str(vector_id):
        bf_ok = bf_index.delete(str(vector_id))
        ivf_ok = ivf_index.delete(str(vector_id))
        del_id = str(vector_id)

    if not (bf_ok or ivf_ok):
        raise HTTPException(status_code=404, detail=f"Vector ID '{vector_id}' not found")

    return GenericResponse(
        status="success",
        message=f"Vector ID '{del_id}' deleted from both indices.",
        total_vectors=len(ivf_index)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
