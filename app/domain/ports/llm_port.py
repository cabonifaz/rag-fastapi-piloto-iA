from abc import ABC, abstractmethod


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
