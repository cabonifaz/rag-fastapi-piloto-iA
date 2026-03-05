import aioboto3
import logging
import os
import asyncio
from typing import Optional, AsyncGenerator, List, Dict, Any
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from botocore.config import Config
from app.domain.ports.vlm_port import VLMPort
from app.infrastructure.vlm.model_factory import ModelConfigFactory
from app.infrastructure.vlm.model_saturation_tracker import ModelSaturationTracker
from app.core.config import settings

logger = logging.getLogger(__name__)

# Global saturation tracker shared across all VLM provider instances
_saturation_tracker: Optional[ModelSaturationTracker] = None

# Vision-capable fallback models only — text-only models cannot handle image content
VLM_FALLBACK_MODELS = [
    {"model_id": "qwen.qwen3-vl-235b-a22b", "max_tokens": 131072},
]

_IMAGE_FORMAT_MAP = {
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "png": "png",
    "webp": "webp",
}


def _get_image_format(s3_key: str) -> str:
    """Derive Bedrock Converse image format from S3 key extension."""
    ext = s3_key.rsplit(".", 1)[-1].lower() if "." in s3_key else ""
    return _IMAGE_FORMAT_MAP.get(ext, "jpeg")


def get_saturation_tracker() -> ModelSaturationTracker:
    global _saturation_tracker
    if _saturation_tracker is None:
        timeout_minutes = getattr(settings, "llm_saturation_timeout_minutes", 60)
        _saturation_tracker = ModelSaturationTracker(
            saturation_timeout_minutes=timeout_minutes
        )
    return _saturation_tracker


class AWSBedrockVLMProvider(VLMPort):
    """
    AWS Bedrock VLM provider using the Converse API with aioboto3.

    Handles multimodal requests: builds image content blocks from S3 keys
    (s3Location format) and appends the text message. Only vision-capable
    models are used — text-only fallbacks are excluded.

    Usage:
        provider = AWSBedrockVLMProvider(region="us-east-1")

        async for chunk in provider.generate_stream(
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            message="What do you see in these images?",
            attachment_keys=["uploads/chat-5/img1.jpg", "uploads/chat-5/img2.png"],
            role_behavior="You are a helpful assistant",
        ):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        region: str,
        model_id: Optional[str] = None,  # defaults to settings.vlm_model_id if not provided
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self.session_params = {"region_name": region}

        if profile_name and os.getenv("ENVIRONMENT", "").lower() != "production":
            self.session_params["profile_name"] = profile_name
        elif aws_access_key_id and aws_secret_access_key:
            self.session_params["aws_access_key_id"] = aws_access_key_id
            self.session_params["aws_secret_access_key"] = aws_secret_access_key

        self.boto_config = Config(
            connect_timeout=10,
            read_timeout=60,  # VLM requests take longer due to image processing
            retries={"max_attempts": 0},
        )

        self.model_id = model_id or settings.vlm_model_id
        self.region = region
        self.model_config = ModelConfigFactory.get_model_config(self.model_id) if self.model_id else None
        self.saturation_tracker = get_saturation_tracker()

        self.session = aioboto3.Session(**self.session_params)
        session_info = {
            k: "***" if "key" in k.lower() or "secret" in k.lower() else v
            for k, v in self.session_params.items()
        }
        logger.info(f"✨ Created VLM aioboto3.Session (id: {id(self.session)}) | Config: {session_info}")

    def _build_system_config(
        self,
        role_behavior: str = "",
        request_timezone: Optional[str] = None,
        utc_formatted: Optional[str] = None,
        local_formatted: Optional[str] = None,
    ) -> Optional[List[Dict[str, str]]]:
        system_text = f"""{role_behavior}

Time context:
- UTC: {utc_formatted}
- Local: {local_formatted}
- Timezone: {request_timezone}

Temporal rules:
- Treat this time context as the single source of truth.
- Use local time for all time-sensitive reasoning.
- Apply time filtering ONLY when the task involves scheduling, reminders, events, or availability.
- Do not invent or infer external dates or times.
- If time context is insufficient, ask for clarification.

Response rules:
- Use the conversation history to interpret the user's question in context. If the question references or continues a previous topic, infer the full meaning from the history.
- Answer directly and concisely, but do not remove essential information required for accuracy.
- Do not reveal internal reasoning.
- Match the user's language. If unclear or mixed, default to Spanish.

Formatting rules:
- Render structured/API data as Markdown tables.
- Convert data with headers + rows-like structure into a table."""

        return [{"text": system_text}] if system_text else None

    def _build_image_block(self, s3_key: str) -> Dict[str, Any]:
        """Build a Bedrock Converse image content block from an S3 key."""
        return {
            "image": {
                "format": _get_image_format(s3_key),
                "source": {
                    "s3Location": {
                        "uri": f"s3://{settings.s3_images_bucket}/{s3_key}"
                    }
                },
            }
        }

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
        Stream a multimodal response using Bedrock Converse Stream API with fallback.

        Args:
            model_id: Primary vision-capable model ID
            message: Text part of the user turn
            attachment_keys: S3 object keys — converted to s3Location image blocks
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            role_behavior: System prompt string
            messages: Optional prior conversation history
            fallback_models: Vision-capable fallback model IDs (must support images)
            request_timezone: Timezone string for time context
            utc_formatted: Formatted UTC timestamp
            local_formatted: Formatted local timestamp

        Yields:
            Text chunks as they are generated
        """
        models_to_try = [model_id]
        if fallback_models:
            models_to_try.extend(fallback_models)
        else:
            fallback_list = [
                m["model_id"] for m in VLM_FALLBACK_MODELS if m["model_id"] != model_id
            ]
            models_to_try.extend(fallback_list)

        available_models = []
        for model in models_to_try:
            if not await self.saturation_tracker.is_saturated(model):
                available_models.append(model)
            else:
                logger.debug(f"⏭️ Skipping {model} (saturated)")

        if not available_models:
            saturated = await self.saturation_tracker.get_active_saturated_models()
            raise ConnectionError(
                f"All VLM models saturated: {', '.join(saturated)}. Please try again in a few minutes."
            )

        last_error = None
        for attempt, current_model in enumerate(available_models):
            try:
                logger.info(f"🔄 VLM generating with {current_model} (attempt {attempt + 1}/{len(available_models)})")

                async for chunk in self._stream_from_model(
                    model_id=current_model,
                    message=message,
                    attachment_keys=attachment_keys,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    role_behavior=role_behavior,
                    messages=messages,
                    request_timezone=request_timezone,
                    utc_formatted=utc_formatted,
                    local_formatted=local_formatted,
                ):
                    yield chunk

                logger.info(f"✅ VLM successfully generated with {current_model}")
                return

            except ConnectionError as e:
                error_msg = str(e).lower()
                last_error = str(e)

                saturation_patterns = getattr(settings, "llm_saturation_patterns", [
                    "throttling", "quota exceeded", "service quota", "read timeout", "rate limit"
                ])
                is_saturation = any(p.lower() in error_msg for p in saturation_patterns)

                if is_saturation:
                    await self.saturation_tracker.mark_saturated(current_model)
                    if attempt < len(available_models) - 1:
                        next_model = available_models[attempt + 1]
                        logger.warning(f"VLM model {current_model} SATURATED. Falling back to {next_model}")
                        continue
                    else:
                        logger.error(f"VLM model {current_model} saturated and no fallbacks available")
                        raise
                else:
                    logger.error(f"Non-recoverable VLM error from {current_model}: {error_msg}")
                    raise

            except Exception as e:
                logger.error(f"Unexpected VLM error from {current_model}: {str(e)}")
                raise

        raise ConnectionError(f"All VLM models exhausted. Last error: {last_error}")

    async def _stream_from_model(
        self,
        model_id: str,
        message: str,
        attachment_keys: List[str],
        max_tokens: int = 2048,
        temperature: float = 0.3,
        top_p: float = 0.9,
        role_behavior: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        request_timezone: Optional[str] = None,
        utc_formatted: Optional[str] = None,
        local_formatted: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        try:
            # Build multimodal content for the current user turn:
            # images first (s3Location blocks), text message last
            user_content = [self._build_image_block(key) for key in attachment_keys]
            user_content.append({"text": message})

            converse_messages: List[Dict[str, Any]] = []

            if messages is not None:
                # Prepend conversation history as plain text turns
                for msg in messages:
                    converse_messages.append({
                        "role": msg["role"],
                        "content": [{"text": msg["content"]}],
                    })

            # Current user turn with attachments
            converse_messages.append({"role": "user", "content": user_content})

            model_config = ModelConfigFactory.get_model_config(model_id)

            inference_config = {
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            }
            if hasattr(model_config, "supports_both_temp_and_top_p") and not model_config.supports_both_temp_and_top_p:
                del inference_config["topP"]

            request_params = {
                "modelId": model_config.model_id,
                "messages": converse_messages,
                "inferenceConfig": inference_config,
                "system": self._build_system_config(
                    role_behavior, request_timezone, utc_formatted, local_formatted
                ),
            }

            if hasattr(model_config, "get_converse_additional_fields"):
                request_params["additionalModelRequestFields"] = model_config.get_converse_additional_fields()

            logger.info(
                f"♻️ VLM reusing session (id: {id(self.session)}) | "
                f"model={model_id}, attachments={len(attachment_keys)}, max_tokens={max_tokens}"
            )

            async with self.session.client("bedrock-runtime", config=self.boto_config) as client:
                response = await client.converse_stream(**request_params)
                stream = response.get("stream")
                if stream is None:
                    return

                try:
                    yielded_count = 0
                    async for event in stream:
                        if "contentBlockDelta" in event:
                            delta = event["contentBlockDelta"].get("delta", {})
                            if "text" in delta:
                                text = delta["text"]
                                if text:
                                    yielded_count += 1
                                    yield text
                        elif "messageStop" in event:
                            stop_reason = event["messageStop"].get("stopReason")
                            if yielded_count == 0 and stop_reason == "max_tokens":
                                yield f"__STOP_REASON__:{stop_reason}"
                            break
                finally:
                    if hasattr(stream, "close"):
                        stream.close()

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            logger.error(f"AWS ClientError in VLM _stream_from_model ({model_id}): {error_code} - {e}")
            if error_code == "ValidationException":
                raise ValueError(f"Invalid parameters for model {model_id}: {str(e)}")
            elif error_code == "ThrottlingException":
                raise ConnectionError(f"Rate limit exceeded for model {model_id}")
            elif error_code == "ServiceQuotaExceededException":
                raise ConnectionError(f"Service quota exceeded for model {model_id}")
            elif error_code == "ModelNotReadyException":
                raise ValueError(f"Model {model_id} is not ready")
            elif error_code == "ResourceNotFoundException":
                raise ValueError(f"Model {model_id} not found or not accessible")
            else:
                raise ConnectionError(f"AWS Bedrock error: {error_code}")

        except NoCredentialsError as e:
            logger.error(f"AWS credentials error in VLM _stream_from_model ({model_id}): {e}")
            raise ConnectionError("AWS credentials not configured or invalid")

        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error in VLM _stream_from_model ({model_id}): {e}")
            raise ConnectionError("Unable to connect to AWS Bedrock service")

        except asyncio.TimeoutError:
            logger.error(f"Timeout in VLM _stream_from_model ({model_id})")
            raise ConnectionError("AWS Bedrock VLM request timed out")

        except Exception as e:
            error_message = str(e)
            if "timed out" in error_message.lower() or "timeout" in error_message.lower():
                raise ConnectionError(f"VLM request timed out: {error_message}")
            logger.error(f"Unexpected error in VLM _stream_from_model ({model_id}): {e}")
            raise ConnectionError(f"VLM streaming service error: {error_message}")
