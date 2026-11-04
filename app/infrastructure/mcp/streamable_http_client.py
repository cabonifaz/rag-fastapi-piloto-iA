import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from app.domain.ports.mcp_client_port import MCPClientPort

logger = logging.getLogger(__name__)


class StreamableHTTPMCPClient(MCPClientPort):
    """Stateful MCP client for the Streamable HTTP transport."""

    PROTOCOL_VERSION = "2025-06-18"

    def __init__(
        self,
        server_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.server_url = server_url
        self._api_key = api_key
        self._session_id: Optional[str] = None
        self._request_id = 0
        self._client = httpx.AsyncClient(timeout=timeout)

    async def initialize(self) -> None:
        response = await self._post(
            method="initialize",
            params={
                "protocolVersion": self.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "qamaq-mcp-client", "version": "1.0.0"},
            },
            include_id=True,
        )
        result = response.get("result", {})
        if not result.get("protocolVersion"):
            raise ConnectionError(
                "El servidor MCP no devolvio una inicializacion valida"
            )

        await self._post(
            method="notifications/initialized",
            params={},
            include_id=False,
            expect_response=False,
        )

    async def list_tools(self) -> List[Dict[str, Any]]:
        response = await self._post("tools/list", {}, include_id=True)
        tools = response.get("result", {}).get("tools", [])
        if not isinstance(tools, list):
            raise ConnectionError(
                "El servidor MCP devolvio una lista de herramientas invalida"
            )
        return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._post(
            "tools/call",
            {"name": name, "arguments": arguments},
            include_id=True,
        )
        result = response.get("result")
        if not isinstance(result, dict):
            raise ConnectionError(
                f"La herramienta MCP '{name}' devolvio una respuesta invalida"
            )
        return result

    async def close(self) -> None:
        try:
            if self._session_id:
                await self._client.delete(self.server_url, headers=self._headers())
        except Exception as exc:
            logger.debug("MCP session could not be closed: %s", exc)
        finally:
            await self._client.aclose()
            self._session_id = None

    async def _post(
        self,
        method: str,
        params: Dict[str, Any],
        include_id: bool,
        expect_response: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        if include_id:
            self._request_id += 1
            payload["id"] = self._request_id

        try:
            response = await self._client.post(
                self.server_url,
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                "Tiempo de espera agotado al conectar con el servidor MCP"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ConnectionError(
                f"El servidor MCP respondio HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectionError(
                "No fue posible conectar con el servidor MCP"
            ) from exc

        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id

        if not expect_response or not response.content:
            return {}

        data = self._parse_response(response)
        if "error" in data:
            error = data["error"]
            message = (
                error.get("message", "Error MCP")
                if isinstance(error, dict)
                else str(error)
            )
            raise ConnectionError(message)
        return data

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self.PROTOCOL_VERSION,
        }
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    @staticmethod
    def _parse_response(response: httpx.Response) -> Dict[str, Any]:
        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" not in content_type:
            return response.json()

        for line in response.text.splitlines():
            if line.startswith("data:"):
                data = line[5:].strip()
                if data:
                    return json.loads(data)
        raise ConnectionError("El servidor MCP devolvio un evento sin datos")
