import asyncio
import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from app.domain.ports.mcp_client_port import MCPClientPort
from app.domain.ports.tool_calling_llm_port import ToolCallingLLMPort
from app.infrastructure.repositories.area_mcp_repository import AreaMCPRepository
from app.infrastructure.repositories.chat_repository import ChatRepository
from app.services.message_service import MessageService
from app.utils.query_utils import clean_user_query
from app.workflows.helpers.llm_common import update_chat_last_message_date


class MCPChatService:
    """Orchestrates chat history, Bedrock tool use and MCP tool execution."""

    MAX_TOOL_ROUNDS = 5

    def __init__(
        self,
        mcp_client_factory: Callable[[str, str], MCPClientPort],
        tool_llm_provider: ToolCallingLLMPort,
        message_service: MessageService,
    ):
        self.mcp_client_factory = mcp_client_factory
        self.tool_llm_provider = tool_llm_provider
        self.message_service = message_service

    @staticmethod
    def is_available(db: Session, company_id: int, area_id: int) -> bool:
        return AreaMCPRepository(db).is_available(company_id, area_id)

    async def process(
        self,
        user_id: int,
        message: str,
        company_id: int,
        area_id: int,
        created_at: str,
        db: Session,
        chat_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        cleaned_message = clean_user_query(message)
        if not cleaned_message:
            raise ValueError("El mensaje no puede estar vacio")
        try:
            user_timestamp = int(created_at)
        except ValueError as exc:
            raise ValueError(
                "created_at debe ser un timestamp en milisegundos"
            ) from exc

        repository = AreaMCPRepository(db)
        if not repository.is_available(company_id, area_id):
            raise ValueError(
                "El MCP no esta habilitado para la empresa y area indicadas"
            )

        if chat_id is None:
            chat_id = await self._create_chat(db, user_id, company_id, area_id)

        mcp_config = repository.get_chat_config(user_id, area_id, chat_id)
        if not mcp_config:
            raise ValueError("No se encontro una configuracion MCP valida para el chat")

        history = await self._get_history(chat_id)
        messages = self._to_bedrock_messages(history)
        messages.append({"role": "user", "content": [{"text": cleaned_message}]})

        saved = await self.message_service.create_message(
            chat_id=chat_id,
            created_at=created_at,
            sender=0,
            message=cleaned_message,
        )
        if not saved:
            raise ConnectionError("No fue posible guardar el mensaje del usuario")

        mcp_client = self.mcp_client_factory(
            mcp_config["URL"],
            mcp_config["TOKEN_AUTH"],
        )
        try:
            await mcp_client.initialize()
            mcp_tools = await mcp_client.list_tools()
            bedrock_tools = self._to_bedrock_tools(mcp_tools)
            if not bedrock_tools:
                raise ConnectionError(
                    "El servidor MCP no publico herramientas disponibles"
                )

            assistant_response = await self._run_tool_loop(
                mcp_client=mcp_client,
                messages=messages,
                tools=bedrock_tools,
                config=mcp_config,
                model_id=mcp_config["LLM_MODEL"],
            )
        finally:
            await mcp_client.close()

        assistant_timestamp = str(max(int(time.time() * 1000), user_timestamp + 1000))

        saved = await self.message_service.create_message(
            chat_id=chat_id,
            created_at=assistant_timestamp,
            sender=1,
            message=assistant_response,
        )
        if not saved:
            raise ConnectionError("No fue posible guardar la respuesta del asistente")
        await update_chat_last_message_date(db, chat_id)

        return {
            "response": assistant_response,
            "chat_id": chat_id,
            "result": {
                "idTipoMensaje": 2,
                "mensaje": "Respuesta generada correctamente",
            },
        }

    async def _run_tool_loop(
        self,
        mcp_client: MCPClientPort,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        config: Dict[str, Any],
        model_id: str,
    ) -> str:
        for round_index in range(self.MAX_TOOL_ROUNDS + 1):
            turn = await self.tool_llm_provider.converse(
                model_id=model_id,
                messages=messages,
                tools=tools,
                max_tokens=int(config["MAX_TOKENS"]),
                temperature=float(config["LLM_TEMPERATURE"]),
                system_prompt=config.get("SYSTEM_PROMPT"),
            )
            assistant_message = turn["message"]
            messages.append(assistant_message)

            tool_uses = [
                block["toolUse"]
                for block in assistant_message.get("content", [])
                if "toolUse" in block
            ]
            if not tool_uses:
                text_parts = [
                    block["text"]
                    for block in assistant_message.get("content", [])
                    if block.get("text")
                ]
                response = "".join(text_parts).strip()
                if not response:
                    raise ValueError("El modelo no genero una respuesta")
                return response

            if round_index == self.MAX_TOOL_ROUNDS:
                break

            tool_results = []
            for tool_use in tool_uses:
                result = await mcp_client.call_tool(
                    tool_use["name"], tool_use.get("input", {})
                )
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_use["toolUseId"],
                            "content": self._tool_result_content(result),
                            "status": "error" if result.get("isError") else "success",
                        }
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        raise ValueError("Se alcanzo el limite de llamadas a herramientas MCP")

    async def _create_chat(
        self, db: Session, user_id: int, company_id: int, area_id: int
    ) -> int:
        title = f"Nueva conversacion {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        repository = ChatRepository(db)
        chat_id = await asyncio.to_thread(
            repository.create_chat,
            id_usuario=user_id,
            id_area=area_id,
            id_empresa=company_id,
            titulo=title,
        )
        if not chat_id:
            raise ConnectionError("No fue posible crear el chat")
        return int(chat_id)

    async def _get_history(self, chat_id: int) -> List[Any]:
        return await self.message_service.get_last_n_messages(
            chat_id=f"chat-{chat_id}", n=8
        )

    @staticmethod
    def _to_bedrock_messages(history: List[Any]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for item in history:
            role = "user" if item.sender == 0 else "assistant"
            content = item.message
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].append({"text": content})
            else:
                messages.append({"role": role, "content": [{"text": content}]})

        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        if messages and messages[-1]["role"] == "user":
            messages.pop()
        return messages

    @staticmethod
    def _to_bedrock_tools(mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "toolSpec": {
                    "name": tool["name"],
                    "description": tool.get("description") or tool["name"],
                    "inputSchema": {
                        "json": tool.get("inputSchema") or {"type": "object"}
                    },
                }
            }
            for tool in mcp_tools
            if tool.get("name")
        ]

    @staticmethod
    def _tool_result_content(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        if result.get("structuredContent") is not None:
            return [{"json": result["structuredContent"]}]

        content = result.get("content", [])
        text_parts = [
            item.get("text", "") for item in content if item.get("type") == "text"
        ]
        if text_parts:
            return [{"text": "\n".join(text_parts)}]
        return [{"text": json.dumps(content, ensure_ascii=False)}]
