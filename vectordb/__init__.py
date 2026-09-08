"""Custom Vector Database Engine from Scratch.
Built with pure NumPy for arithmetic, without external ANN libraries.
"""

from .base import BaseVectorIndex
from .distance import l2_normalize, cosine_similarity, squared_euclidean
from .brute_force import BruteForceIndex
from .ivf_flat import IVFFlatIndex
from .hnsw import HNSWIndex
from .kmeans import KMeansScratch

__all__ = [
    "BaseVectorIndex",
    "BruteForceIndex",
    "IVFFlatIndex",
    "HNSWIndex",
    "KMeansScratch",
    "l2_normalize",
    "cosine_similarity",
    "squared_euclidean",
]
