from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional


class LLMPort(ABC):
    """
    Puerto (interfaz) para servicios de LLM.
    Define cómo la aplicación interactúa con cualquier proveedor de LLM.
    """

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, system_prompt: Optional[str] = None) -> str:
        """
        Genera texto a partir de un prompt.

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            system_prompt: Optional system prompt
        """
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Genera texto a partir de un prompt con streaming.

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            system_prompt: Optional system prompt
        """
        pass
