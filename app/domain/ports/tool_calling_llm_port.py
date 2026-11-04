from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ToolCallingLLMPort(ABC):
    """Port for a Bedrock model turn that can request tool execution."""

    @abstractmethod
    async def converse(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        pass
