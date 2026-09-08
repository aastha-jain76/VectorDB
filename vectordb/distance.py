"""Mathematical distance metrics and vector normalization using pure NumPy.
Strictly zero external ML/ANN library dependencies.
"""

from typing import Union
import numpy as np


def l2_normalize(vectors: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Normalize vectors to unit Euclidean length (L2 norm = 1.0).
    
    Under unit normalization, Cosine Similarity simplifies to the matrix dot product:
        Sim_cos(u, v) = u_norm · v_norm
        
    Args:
        vectors: NumPy array of shape (D,) or (N, D).
        axis: Axis along which to compute the L2 norm.
        eps: Small epsilon to prevent division by zero.
        
    Returns:
        np.ndarray: Unit-normalized vectors with the same shape and float32 dtype.
    """
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return arr / norms


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> Union[float, np.ndarray]:
    """Compute exact cosine similarity between vector(s) u and v.
    
    Args:
        u: Query vector of shape (D,) or matrix of shape (M, D).
        v: Candidate vector of shape (D,) or matrix of shape (N, D).
        
    Returns:
        float or np.ndarray: Cosine similarity values bounded in [-1.0, 1.0].
    """
    u_norm = l2_normalize(u)
    v_norm = l2_normalize(v)
    
    if u_norm.ndim == 1 and v_norm.ndim == 1:
        return float(np.dot(u_norm, v_norm))
    elif u_norm.ndim == 1 and v_norm.ndim == 2:
        return np.dot(v_norm, u_norm)
    elif u_norm.ndim == 2 and v_norm.ndim == 2:
        return np.dot(u_norm, v_norm.T)
    else:
        raise ValueError(f"Incompatible shapes: u={u.shape}, v={v.shape}")


def squared_euclidean(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distance between two unit-normalized sets:
    ||u - v||^2 = ||u||^2 + ||v||^2 - 2*(u · v) = 2 - 2*(u · v)
    
    Args:
        u: Shape (N, D) normalized vectors.
        v: Shape (K, D) normalized vectors (e.g. centroids).
        
    Returns:
        np.ndarray: Pairwise distances of shape (N, K).
    """
    # Using matrix multiplication: 2 - 2 * (u @ v.T)
    sims = np.dot(u, v.T)
    return np.maximum(2.0 - 2.0 * sims, 0.0)
