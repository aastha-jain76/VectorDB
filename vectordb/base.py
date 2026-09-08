"""Base abstract class for all Vector Database Index implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class BaseVectorIndex(ABC):
    """Abstract Base Class defining the standard Vector Database contract."""

    @abstractmethod
    def insert(
        self, 
        vector_id: Union[int, str], 
        vector: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Insert a single vector with an ID and optional metadata.
        
        Args:
            vector_id: Unique identifier for the vector.
            vector: 1D NumPy array representing the vector.
            metadata: Optional dictionary with document text or attributes.
        """
        pass

    @abstractmethod
    def batch_insert(
        self, 
        vectors: np.ndarray, 
        ids: List[Union[int, str]], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Batch insert multiple vectors.
        
        Args:
            vectors: 2D NumPy array of shape (N, D).
            ids: List of N unique IDs.
            metadatas: Optional list of N metadata dictionaries.
        """
        pass

    @abstractmethod
    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 10,
        **kwargs
    ) -> List[Tuple[Union[int, str], float, Dict[str, Any]]]:
        """Search for top_k nearest neighbors by cosine similarity.
        
        Args:
            query_vector: 1D NumPy array representing the query.
            top_k: Number of nearest neighbors to retrieve.
            **kwargs: Index-specific hyperparameters (e.g., n_probe, ef_search).
            
        Returns:
            List of tuples: (vector_id, similarity_score, metadata)
            sorted in descending order of similarity.
        """
        pass

    @abstractmethod
    def delete(self, vector_id: Union[int, str]) -> bool:
        """Remove a vector by its ID.
        
        Args:
            vector_id: The ID of the vector to remove.
            
        Returns:
            bool: True if vector was found and removed, False otherwise.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of active vectors in the index."""
        pass
