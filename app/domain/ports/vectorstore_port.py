from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorStorePort(ABC):
    @abstractmethod
    async def search(
        self, 
        query_vector: List[float], 
        top_k: int = 5,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors and return documents with metadata."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        pass