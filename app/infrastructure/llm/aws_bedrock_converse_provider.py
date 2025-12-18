import aioboto3
import json
import logging
import os
import asyncio
from typing import Optional, AsyncGenerator, List, Dict, Any
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from botocore.config import Config
from app.domain.ports.llm_port import LLMPort
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


class AWSBedrockConverseProvider(LLMPort):
    """
    AWS Bedrock LLM provider using the Converse API with aioboto3.

    Provides unified interface for all supported models (Claude, Llama, OpenAI, etc.)
    with built-in system prompt support and automatic fallback on model saturation.

    Features:
    - Automatic fallback to alternative models when primary is saturated (429 errors)
    - Circuit breaker pattern to track saturated models globally
    - Configurable fallback chains per model
    - Support for per-request model selection via fallback_models parameter
    - Transparent error handling with saturation detection
    - Fully async with aioboto3 (no thread pool blocking)

    Usage:
        provider = AWSBedrockConverseProvider(
            region="us-east-1",
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            role_behavior="You are a helpful assistant"
        )

        # Generate with automatic fallback
        async for chunk in provider.generate_stream(prompt="What is AI?"):
            print(chunk, end="", flush=True)

        # Or with custom fallback models
        async for chunk in provider.generate_stream(
            prompt="What is AI?",
            fallback_models=["meta.llama3-1-70b-instruct-v1:0"]
        ):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        region: str,
        model_id: str,
        role_behavior: str,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        """
        Initialize AWS Bedrock Converse client with saturation tracking.

        Args:
            region: AWS region
            model_id: Bedrock model ID
            role_behavior: Role behavior instructions for the model
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
            connect_timeout=30,
            read_timeout=120,
            retries={'max_attempts': 0}
        )

        self.model_id = model_id
        self.region = region
        self.default_role_behavior = role_behavior
        self.profile_name = profile_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key

        # Get model-specific configuration for optimized prompts
        self.model_config = ModelConfigFactory.get_model_config(model_id)

        # Get reference to global saturation tracker
        self.saturation_tracker = get_saturation_tracker()

    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt for this provider instance."""
        self.system_prompt = system_prompt

    def _build_system_config(self, custom_system: Optional[str] = None, timestamp_utc: Optional[str] = None, request_timezone: Optional[str] = None) -> Optional[List[Dict[str, str]]]:
        """Build system configuration for Converse API."""
        # Use custom role behavior or default
        role_behavior = custom_system or self.default_role_behavior

        # Build time context text if timestamp and timezone are provided
        time_context = ""
        if timestamp_utc is not None and request_timezone is not None:
            time_context = f"\nCurrent Time Context: The current timestamp is {timestamp_utc} (Unix UTC format) and the user's timezone is {request_timezone}. Use this information to provide accurate temporal references."
        elif timestamp_utc is not None:
            time_context = f"\nCurrent Time Context: The current timestamp is {timestamp_utc} (Unix UTC format). Use this information to provide accurate temporal references."
        elif request_timezone is not None:
            time_context = f"\nCurrent Time Context: The user's timezone is {request_timezone}. Use this information to provide accurate temporal references."

        # Concatenate role behavior with formatting instructions
        system_text = f"""{role_behavior}{time_context}
Use a natural, human-like tone in responses. Maintain conversational and engaging style throughout.
When providing data or structured information, prioritize technical accuracy and formatting:
- Always render JSON with "table", "headers", and "rows" as a **Markdown table**.
- If the context comes from an API call, render it as a Markdown table and omit references.
Answer directly and briefly. You may include short natural phrases **before or after** the main answer, but not inside technical tables or structured data.
Do not overthink, speculate, or explain your internal reasoning.
Always mirror the user's language exactly in your response. If the input language is unclear, mixed,
or contains spelling errors, default to Spanish. Format responses in Markdown when relevant."""

        if system_text:
            return [{"text": system_text}]
        return None

    async def generate_stream(
        self,
        prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        role_behavior: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        fallback_models: Optional[List[str]] = None,
        timestamp_utc: Optional[str] = None,
        request_timezone: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate text using AWS Bedrock Converse Stream API with automatic fallback.

        Attempts to generate using the primary model. If the model is saturated
        (rate limit/quota exceeded), automatically tries fallback models.

        Uses aioboto3 for fully async, non-blocking AWS API calls.

        Args:
            prompt: User prompt (used if messages is None)
            max_tokens: Maximum tokens to generate (default: 2048)
            temperature: Temperature for sampling (default: 0.3)
            role_behavior: Optional role behavior (overrides instance system_prompt)
            messages: Optional conversation history in format [{"role": "user/assistant", "content": "..."}]
                     If provided, prompt will be ignored and messages will be used instead
            fallback_models: Optional list of fallback model IDs to try if primary is saturated.
                           If not provided, uses hardcoded MODELS list.
            timestamp_utc: Optional Unix timestamp in UTC format (as string)
            request_timezone: Optional timezone string for the request

        Yields:
            Text chunks as they are generated
        """
        # Build list of models to try
        models_to_try = [self.model_id]
        if fallback_models:
            models_to_try.extend(fallback_models)
        else:
            # Use hardcoded MODELS list as fallback (all models except current one)
            fallback_list = [m["model_id"] for m in MODELS if m["model_id"] != self.model_id]
            models_to_try.extend(fallback_list)

        # Filter out saturated models
        available_models = []
        for model_id in models_to_try:
            if not await self.saturation_tracker.is_saturated(model_id):
                available_models.append(model_id)
            else:
                logger.debug(f"⏭️ Skipping {model_id} (currently marked as saturated)")

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

                # Stream from this model
                async for chunk in self._stream_from_model(
                    model_id=current_model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    role_behavior=role_behavior,
                    messages=messages,
                    timestamp_utc=timestamp_utc,
                    request_timezone=request_timezone
                ):
                    yield chunk

                # Success - exit
                logger.info(f"✅ Successfully generated with {current_model}")
                return

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

    async def _stream_from_model(
        self,
        model_id: str,
        prompt: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        role_behavior: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        timestamp_utc: Optional[str] = None,
        request_timezone: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream from a specific model (internal helper).

        This contains the core streaming logic using aioboto3.
        Called by generate_stream for each model attempt.

        Args:
            model_id: The specific model ID to use for this attempt
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            role_behavior: Optional role behavior override
            messages: Optional conversation history
            timestamp_utc: Optional Unix timestamp in UTC format (as string)
            request_timezone: Optional timezone string for the request

        Yields:
            Text chunks as they are generated
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

            # Build request parameters
            request_params = {
                "modelId": model_id,
                "messages": converse_messages,
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": getattr(settings, 'llm_top_p', 0.9)
                }
            }

            # Configure model-specific additional parameters (e.g., OpenAI reasoning)
            if hasattr(model_config, 'get_converse_additional_fields'):
                additional_fields = model_config.get_converse_additional_fields()
                request_params["additionalModelRequestFields"] = additional_fields

            # Add system prompt with timestamp and timezone context
            request_params["system"] = self._build_system_config(role_behavior, timestamp_utc, request_timezone)

            # Use aioboto3 async client for truly non-blocking Bedrock calls
            session = aioboto3.Session(**self.session_params)
            async with session.client("bedrock-runtime", config=self.boto_config) as client:
                # Fully async API call - no thread pool needed!
                response = await client.converse_stream(**request_params)

                # Process streaming response - ensure proper cleanup
                stream = response.get("stream")
                if stream is None:
                    return

                try:
                    yielded_count = 0

                    # Iterate through stream events asynchronously
                    async for event in stream:
                        # Handle content block delta (text chunks)
                        if "contentBlockDelta" in event:
                            delta = event["contentBlockDelta"].get("delta", {})
                            if "text" in delta:
                                text = delta["text"]
                                # Filter out empty chunks from AWS Bedrock streaming protocol
                                if text:
                                    yielded_count += 1
                                    yield text

                        # Handle metadata events (optional logging)
                        elif "metadata" in event:
                            metadata = event["metadata"]

                        # Handle message stop event
                        elif "messageStop" in event:
                            stop_reason = event["messageStop"].get("stopReason")
                            # Yield stop reason info if no content was generated
                            if yielded_count == 0 and stop_reason == "max_tokens":
                                yield f"__STOP_REASON__:{stop_reason}"
                            break
                finally:
                    # Ensure stream is properly closed to avoid resource leaks
                    if hasattr(stream, 'close'):
                        stream.close()

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"AWS ClientError in _stream_from_model ({model_id}): {error_code} - {e}")
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
            logger.error(f"AWS credentials error in _stream_from_model ({model_id}): {e}")
            raise ConnectionError("AWS credentials not configured or invalid")

        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error in _stream_from_model ({model_id}): {e}")
            raise ConnectionError("Unable to connect to AWS Bedrock service - check internet connection")

        except asyncio.TimeoutError as e:
            logger.error(f"Timeout error in _stream_from_model ({model_id}): {e}")
            raise ConnectionError("AWS Bedrock request timed out - network may be slow or unstable")

        except Exception as e:
            # Check if it's a botocore timeout exception
            error_message = str(e)
            if "timed out" in error_message.lower() or "timeout" in error_message.lower():
                logger.error(f"Timeout error in _stream_from_model ({model_id}): {e}")
                raise ConnectionError(f"Request timed out after waiting for response: {str(e)}")

            logger.error(f"Unexpected error in _stream_from_model ({model_id}): {e}")
            raise ConnectionError(f"LLM streaming service error: {str(e)}")

    def get_model_config(self):
        """
        Return model-specific configuration for optimized prompts.
        Uses the same configs as invoke_model (anthropic_models, meta_models, openai_models).
        """
        return self.model_config
