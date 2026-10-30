import logging
from typing import Any, Dict, List, Optional

from botocore.exceptions import ClientError

from app.core.aws_clients import get_bedrock_client
from app.domain.ports.tool_calling_llm_port import ToolCallingLLMPort
from app.infrastructure.llm.model_factory import ModelConfigFactory

logger = logging.getLogger(__name__)


class AWSBedrockToolProvider(ToolCallingLLMPort):
    """Bedrock Converse adapter with client-side tool calling support."""

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
        model_config = ModelConfigFactory.get_model_config(model_id)
        inference_config: Dict[str, Any] = {
            "maxTokens": max_tokens,
            "temperature": float(temperature),
        }
        if top_p is not None and getattr(
            model_config, "supports_both_temp_and_top_p", True
        ):
            inference_config["topP"] = float(top_p)

        request: Dict[str, Any] = {
            "modelId": model_config.model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
            "toolConfig": {"tools": tools},
        }
        if system_prompt and system_prompt.strip():
            request["system"] = [{"text": system_prompt.strip()}]

        try:
            async with get_bedrock_client() as client:
                response = await client.converse(**request)
        except ClientError as exc:
            error = exc.response.get("Error", {})
            logger.error(
                "Bedrock tool-use error: %s - %s",
                error.get("Code"),
                error.get("Message"),
            )
            raise ConnectionError(
                "Error al invocar el modelo configurado en Bedrock"
            ) from exc

        message = response.get("output", {}).get("message")
        if not message:
            raise ConnectionError("Bedrock no devolvio un mensaje")

        return {
            "message": message,
            "stop_reason": response.get("stopReason"),
        }
