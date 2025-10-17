from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, List, Dict


class LLMPort(ABC):
    """
    Puerto (interfaz) para servicios de LLM.
    Define cómo la aplicación interactúa con cualquier proveedor de LLM.
    """

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Genera texto a partir de un prompt con streaming.

        Args:
            prompt: User prompt (used if messages is None)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            system_prompt: Optional system prompt
            messages: Optional conversation history in format [{"role": "user/assistant", "content": "..."}]
                     If provided, prompt will be ignored and messages will be used instead
        """
        pass
