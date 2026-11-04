from abc import ABC, abstractmethod
from typing import Any, Dict, List


class MCPClientPort(ABC):
    """Port for interacting with an MCP server."""

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass
