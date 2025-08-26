from abc import ABC, abstractmethod
from typing import AsyncGenerator


class LLMPort(ABC):
    """
    Puerto (interfaz) para servicios de LLM.
    Define cómo la aplicación interactúa con cualquier proveedor de LLM.
    """

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Genera texto a partir de un prompt.
        """
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> AsyncGenerator[str, None]:
        """
        Genera texto a partir de un prompt con streaming.
        """
        pass
