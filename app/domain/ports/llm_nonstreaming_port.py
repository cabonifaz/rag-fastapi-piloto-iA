from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class LLMNonStreamingPort(ABC):
    """
    Puerto (interfaz) para servicios de LLM sin streaming.
    Define cómo la aplicación interactúa con proveedores de LLM que generan respuestas completas.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        role_behavior: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        fallback_models: Optional[List[str]] = None
    ) -> str:
        """
        Genera texto a partir de un prompt sin streaming (respuesta completa).

        Args:
            prompt: User prompt (used if messages is None)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            role_behavior: Optional role behavior (system prompt)
            messages: Optional conversation history in format [{"role": "user/assistant", "content": "..."}]
                     If provided, prompt will be ignored and messages will be used instead
            fallback_models: Optional list of fallback model IDs to try if primary is saturated

        Returns:
            Complete generated text as a single string
        """
        pass
