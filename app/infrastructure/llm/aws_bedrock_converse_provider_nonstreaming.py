import aioboto3
import json
import logging
import os
import asyncio
from typing import Optional, List, Dict, Any
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from botocore.config import Config
from app.domain.ports.llm_nonstreaming_port import LLMNonStreamingPort
from app.infrastructure.llm.model_factory import ModelConfigFactory
from app.infrastructure.llm.model_saturation_tracker import ModelSaturationTracker
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Global saturation tracker - shared across all provider instances
_saturation_tracker: Optional[ModelSaturationTracker] = None

# Hardcoded list of 3 models with max tokens
MODELS = [
    {"model_id": "us.meta.llama4-maverick-17b-instruct-v1:0", "max_tokens": 4096},
    {"model_id": "qwen.qwen3-32b-v1:0", "max_tokens": 16384},
    {"model_id": "us.amazon.nova-premier-v1:0", "max_tokens": 16000},
]


def get_saturation_tracker() -> ModelSaturationTracker:
    """Get or create the global saturation tracker instance."""
    global _saturation_tracker
    if _saturation_tracker is None:
        timeout_minutes = getattr(settings, 'llm_saturation_timeout_minutes', 60)
        _saturation_tracker = ModelSaturationTracker(
            saturation_timeout_minutes=timeout_minutes
        )
    return _saturation_tracker


class AWSBedrockConverseNonStreamingProvider(LLMNonStreamingPort):
    """
    AWS Bedrock LLM provider using the Converse API with aioboto3.

    NON-STREAMING VERSION: Returns complete message at the end instead of chunks.

    Provides unified interface for all supported models (Claude, Llama, OpenAI, etc.)
    with built-in system prompt support and automatic fallback on model saturation.

    Features:
    - Automatic fallback to alternative models when primary is saturated (429 errors)
    - Circuit breaker pattern to track saturated models globally
    - Configurable fallback chains per model
    - Support for per-request model selection via fallback_models parameter
    - Transparent error handling with saturation detection
    - Fully async with aioboto3 (no thread pool blocking)
    - Returns complete message at once (no streaming)

    Usage:
        provider = AWSBedrockConverseNonStreamingProvider(
            region="us-east-1",
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0"
        )

        # Generate with role behavior provided per request
        response = await provider.generate(
            prompt="What is AI?",
            role_behavior="You are a helpful assistant"
        )
        print(response)

        # Or with custom fallback models
        response = await provider.generate(
            prompt="What is AI?",
            role_behavior="You are a helpful assistant",
            fallback_models=["meta.llama3-1-70b-instruct-v1:0"]
        )
        print(response)
    """

    def __init__(
        self,
        region: str,
        model_id: Optional[str] = None,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        """
        Initialize AWS Bedrock Converse client with saturation tracking.

        Args:
            region: AWS region
            model_id: Bedrock model ID (optional, will be provided per-request from database)
            profile_name: AWS profile name (optional)
            aws_access_key_id: AWS access key ID (optional)
            aws_secret_access_key: AWS secret access key (optional)
        """
        self.session_params = {"region_name": region}

        # Use profile only in development, not in production with IAM roles
        if profile_name and os.getenv('ENVIRONMENT', '').lower() != 'production':
            self.session_params["profile_name"] = profile_name
        # If no profile, use direct credentials if available
        elif aws_access_key_id and aws_secret_access_key:
            self.session_params["aws_access_key_id"] = aws_access_key_id
            self.session_params["aws_secret_access_key"] = aws_secret_access_key

        # Configure boto3 with connection and read timeouts
        # IMPORTANT: Disable retries - we handle retries via fallback logic
        self.boto_config = Config(
            connect_timeout=10,
            read_timeout=30,
            retries={'max_attempts': 0}
        )

        self.model_id = model_id
        self.region = region
        self.profile_name = profile_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key

        # Get model-specific configuration for optimized prompts (will be retrieved per-request)
        self.model_config = ModelConfigFactory.get_model_config(model_id) if model_id else None

        # Get reference to global saturation tracker
        self.saturation_tracker = get_saturation_tracker()

        # ✨ Create aioboto3 session ONCE for reuse across all requests
        # This improves performance by 30-40% (connection pooling, reduced overhead)
        self.session = aioboto3.Session(**self.session_params)
        # Log session parameters (credentials are masked for security)
        session_info = {k: '***' if 'key' in k.lower() or 'secret' in k.lower() else v
                       for k, v in self.session_params.items()}
        logger.info(f"✨ Created NEW aioboto3.Session (id: {id(self.session)}) [Non-streaming] | Config: {session_info}")

    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt for this provider instance."""
        self.system_prompt = system_prompt

    def _build_system_config(self, custom_system: str = "", request_timezone: Optional[str] = None, utc_formatted: str = None, local_formatted: str = None) -> Optional[List[Dict[str, str]]]:
        """Build system configuration for Converse API."""
        # Use custom role behavior (always provided from workflow now)
        role_behavior = custom_system

        # Concatenate role behavior with formatting instructions
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
- Do NOT use Markdown tables.
- Prefer simple Markdown-style lists.
- Use plain-text formatting only (no code blocks).
- When data is tabular (headers + rows-like), convert it into a list format."""

        if system_text:
            return [{"text": system_text}]
        return None

    async def generate(
        self,
        model_id: str,
        prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        top_p: float = 0.9,
        role_behavior: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        fallback_models: Optional[List[str]] = None,
        request_timezone: Optional[str] = None,
        utc_formatted: str = None,
        local_formatted: str = None
    ) -> str:
        """
        Generate text using AWS Bedrock Converse API with automatic fallback.

        NON-STREAMING: Returns the complete response at once.

        Attempts to generate using the primary model. If the model is saturated
        (rate limit/quota exceeded), automatically tries fallback models.

        Uses aioboto3 for fully async, non-blocking AWS API calls.

        Args:
            model_id: Model ID to use as primary model (from database config)
            prompt: User prompt (used if messages is None)
            max_tokens: Maximum tokens to generate (default: 2048)
            temperature: Temperature for sampling (default: 0.3)
            top_p: Top-p (nucleus) sampling parameter (default: 0.9)
            role_behavior: Optional role behavior (overrides instance system_prompt)
            messages: Optional conversation history in format [{"role": "user/assistant", "content": "..."}]
                     If provided, prompt will be ignored and messages will be used instead
            fallback_models: Optional list of fallback model IDs to try if primary is saturated.
                           If not provided, uses hardcoded MODELS list.
            request_timezone: Optional timezone string for the request
            utc_formatted: Formatted UTC timestamp string
            local_formatted: Formatted local timestamp string

        Returns:
            Complete generated text as a single string
        """
        # Build list of models to try
        models_to_try = [model_id]
        if fallback_models:
            models_to_try.extend(fallback_models)
        else:
            # Use hardcoded MODELS list as fallback (all models except current one)
            fallback_list = [m["model_id"] for m in MODELS if m["model_id"] != model_id]
            models_to_try.extend(fallback_list)

        # Filter out saturated models
        available_models = []
        for model in models_to_try:
            if not await self.saturation_tracker.is_saturated(model):
                available_models.append(model)
            else:
                logger.debug(f"⏭️ Skipping {model} (currently marked as saturated)")

        if not available_models:
            saturated = await self.saturation_tracker.get_active_saturated_models()
            raise ConnectionError(
                f"All models saturated: {', '.join(saturated)}. Please try again in a few minutes."
            )

        # Try each available model
        last_error = None
        for attempt, current_model in enumerate(available_models):
            try:
                logger.info(
                    f"🔄 Generating with {current_model} "
                    f"(attempt {attempt + 1}/{len(available_models)})"
                )

                # Generate from this model (accumulate all chunks)
                complete_response = await self._generate_from_model(
                    model_id=current_model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    role_behavior=role_behavior,
                    messages=messages,
                    request_timezone=request_timezone,
                    utc_formatted=utc_formatted,
                    local_formatted=local_formatted
                )

                # Success - return complete response
                logger.info(f"✅ Successfully generated with {current_model}")
                return complete_response

            except ConnectionError as e:
                error_msg = str(e).lower()
                last_error = str(e)

                # Check if this is a saturation error
                saturation_patterns = getattr(settings, 'llm_saturation_patterns', [
                    "throttling",
                    "quota exceeded",
                    "service quota",
                    "read timeout",
                    "rate limit"
                ])
                is_saturation = any(p.lower() in error_msg for p in saturation_patterns)

                if is_saturation:
                    # Mark model as saturated
                    await self.saturation_tracker.mark_saturated(current_model)

                    if attempt < len(available_models) - 1:
                        # Try next available model
                        next_model = available_models[attempt + 1]
                        logger.warning(
                            f"Model {current_model} is SATURATED. "
                            f"Falling back to {next_model}"
                        )
                        continue
                    else:
                        # No more fallbacks
                        logger.error(f"Model {current_model} is saturated and no fallbacks available")
                        raise

                else:
                    # Not saturation error - don't retry
                    logger.error(f"Non-recoverable error from {current_model}: {error_msg}")
                    raise

            except Exception as e:
                # Unexpected error - don't try fallback
                logger.error(f"Unexpected error from {current_model}: {str(e)}")
                raise

        # Should not reach here
        raise ConnectionError(f"All models exhausted. Last error: {last_error}")

    async def _generate_from_model(
        self,
        model_id: str,
        prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        top_p: float = 0.9,
        role_behavior: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        request_timezone: Optional[str] = None,
        utc_formatted: str = None,
        local_formatted: str = None
    ) -> str:
        """
        Generate from a specific model (internal helper).

        NON-STREAMING: Accumulates all chunks and returns complete response.

        This contains the core generation logic using aioboto3.
        Called by generate for each model attempt.

        Args:
            model_id: The specific model ID to use for this attempt
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p (nucleus) sampling parameter
            role_behavior: Optional role behavior override
            messages: Optional conversation history
            request_timezone: Optional timezone string for the request
            utc_formatted: Formatted UTC timestamp string
            local_formatted: Formatted local timestamp string

        Returns:
            Complete generated text as a single string
        """
        try:
            # Build messages array - use provided messages or create from prompt
            if messages is not None:
                # Use provided conversation history
                # Convert messages to Converse API format
                converse_messages = []
                for msg in messages:
                    converse_messages.append({
                        "role": msg["role"],
                        "content": [{"text": msg["content"]}]
                    })

                # Append current prompt as the latest user message (if provided)
                if prompt and prompt.strip():
                    converse_messages.append({
                        "role": "user",
                        "content": [{"text": prompt}]
                    })
            else:
                # Use single prompt (backward compatibility)
                if prompt is None:
                    raise ValueError("Either prompt or messages must be provided")
                converse_messages = [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ]

            # Get model-specific config for this attempt
            model_config = ModelConfigFactory.get_model_config(model_id)

            # Build inference config - always include temperature and topP by default
            inference_config = {
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p
            }

            # Remove topP if the model doesn't support both parameters
            if hasattr(model_config, 'supports_both_temp_and_top_p') and not model_config.supports_both_temp_and_top_p:
                del inference_config["topP"]

            # Build request parameters - use model_id from config (normalized for AWS)
            request_params = {
                "modelId": model_config.model_id,
                "messages": converse_messages,
                "inferenceConfig": inference_config
            }

            # Configure model-specific additional parameters (e.g., OpenAI reasoning)
            if hasattr(model_config, 'get_converse_additional_fields'):
                additional_fields = model_config.get_converse_additional_fields()
                request_params["additionalModelRequestFields"] = additional_fields

            # Add system prompt with timestamp and timezone context
            request_params["system"] = self._build_system_config(role_behavior, request_timezone, utc_formatted, local_formatted)

            # Use pre-initialized session (created once in __init__) for better performance
            # Only the modelId changes per request - session/client are reused
            logger.info(
                f"♻️ Reusing session (id: {id(self.session)}) [Non-streaming] | "
                f"Request params: model={model_id}, max_tokens={max_tokens}, "
                f"temp={temperature}, top_p={top_p}, messages={len(converse_messages)}"
            )
            async with self.session.client("bedrock-runtime", config=self.boto_config) as client:
                # Use non-streaming converse API - returns complete response at once
                response = await client.converse(**request_params)

                # Extract the complete response text from Converse API format
                output = response.get("output", {})
                message = output.get("message", {})
                content_blocks = message.get("content", [])

                # Concatenate all text content blocks
                accumulated_text = ""
                for block in content_blocks:
                    if "text" in block:
                        accumulated_text += block["text"]

                # Check stop reason
                stop_reason = response.get("stopReason")
                if not accumulated_text and stop_reason == "max_tokens":
                    logger.warning(f"Model {model_id} stopped at max_tokens with no content")

                # Log usage metadata if available
                usage = response.get("usage", {})
                if usage:
                    logger.debug(
                        f"Token usage - Input: {usage.get('inputTokens', 0)}, "
                        f"Output: {usage.get('outputTokens', 0)}, "
                        f"Total: {usage.get('totalTokens', 0)}"
                    )

                return accumulated_text

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"AWS ClientError in _generate_from_model ({model_id}): {error_code} - {e}")
            if error_code == 'ValidationException':
                raise ValueError(f"Invalid parameters for model {model_id}: {str(e)}")
            elif error_code == 'ThrottlingException':
                raise ConnectionError(f"Rate limit exceeded for model {model_id}")
            elif error_code == 'ServiceQuotaExceededException':
                raise ConnectionError(f"Service quota exceeded for model {model_id}")
            elif error_code == 'ModelNotReadyException':
                raise ValueError(f"Model {model_id} is not ready")
            elif error_code == 'ResourceNotFoundException':
                raise ValueError(f"Model {model_id} not found or not accessible")
            else:
                raise ConnectionError(f"AWS Bedrock error: {error_code}")

        except NoCredentialsError as e:
            logger.error(f"AWS credentials error in _generate_from_model ({model_id}): {e}")
            raise ConnectionError("AWS credentials not configured or invalid")

        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error in _generate_from_model ({model_id}): {e}")
            raise ConnectionError("Unable to connect to AWS Bedrock service - check internet connection")

        except asyncio.TimeoutError as e:
            logger.error(f"Timeout error in _generate_from_model ({model_id}): {e}")
            raise ConnectionError("AWS Bedrock request timed out - network may be slow or unstable")

        except Exception as e:
            # Check if it's a botocore timeout exception
            error_message = str(e)
            if "timed out" in error_message.lower() or "timeout" in error_message.lower():
                logger.error(f"Timeout error in _generate_from_model ({model_id}): {e}")
                raise ConnectionError(f"Request timed out after waiting for response: {str(e)}")

            logger.error(f"Unexpected error in _generate_from_model ({model_id}): {e}")
            raise ConnectionError(f"LLM generation service error: {str(e)}")

    def get_model_config(self):
        """
        Return model-specific configuration for optimized prompts.
        Uses the same configs as invoke_model (anthropic_models, meta_models, openai_models).
        """
        return self.model_config
