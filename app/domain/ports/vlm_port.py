from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, List, Dict, Any


class VLMPort(ABC):
    """
    Port (interface) for Vision Language Model providers with streaming.
    Accepts multimodal input: a list of S3 attachment keys + a text message.
    """

    @abstractmethod
    async def generate_stream(
        self,
        model_id: str,
        message: str,
        attachment_keys: List[str],
        max_tokens: int = 2048,
        temperature: float = 0.3,
        top_p: float = 0.9,
        role_behavior: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        fallback_models: Optional[List[str]] = None,
        request_timezone: Optional[str] = None,
        utc_formatted: Optional[str] = None,
        local_formatted: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a multimodal response.

        Args:
            model_id: Bedrock model ID (must support vision)
            message: Text part of the user turn
            attachment_keys: S3 object keys for attached files (images, etc.)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            role_behavior: System prompt / role behavior string
            messages: Optional prior conversation history
            fallback_models: Vision-capable fallback model IDs
            request_timezone: Timezone string for time context
            utc_formatted: Formatted UTC timestamp
            local_formatted: Formatted local timestamp

        Yields:
            Text chunks as they are generated
        """
        pass
