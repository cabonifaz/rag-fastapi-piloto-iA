from abc import ABC, abstractmethod
from typing import List


class EmbeddingsPort(ABC):
    """
    Puerto (interfaz) para servicios de embeddings.
    Permite implementar distintos proveedores (OpenAI, Cohere, AWS, etc.)
    """

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Retorna el vector embedding de un texto.
        """
        pass
