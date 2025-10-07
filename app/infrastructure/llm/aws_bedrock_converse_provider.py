import boto3
import json
import logging
import os
from typing import Optional, AsyncGenerator, List, Dict, Any
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from app.domain.ports.llm_port import LLMPort
from app.infrastructure.llm.model_factory import ModelConfigFactory

# Configure logging
logger = logging.getLogger(__name__)


class AWSBedrockConverseProvider(LLMPort):
    """
    AWS Bedrock LLM provider using the Converse API.
    Provides unified interface for all supported models (Claude, Llama, OpenAI, etc.)
    with built-in system prompt support.
    """

    def __init__(
        self,
        region: str,
        model_id: str,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize AWS Bedrock Converse client.

        Args:
            region: AWS region (e.g., "us-east-1")
            model_id: Bedrock model ID (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
            profile_name: AWS profile name (optional, for local development)
            aws_access_key_id: AWS access key ID (optional)
            aws_secret_access_key: AWS secret access key (optional)
            system_prompt: System prompt to use for all conversations (optional)
        """
        session_params = {"region_name": region}

        # Use profile only in development, not in production with IAM roles
        if profile_name and os.getenv('ENVIRONMENT', '').lower() != 'production':
            session_params["profile_name"] = profile_name
        # If no profile, use direct credentials if available
        elif aws_access_key_id and aws_secret_access_key:
            session_params["aws_access_key_id"] = aws_access_key_id
            session_params["aws_secret_access_key"] = aws_secret_access_key

        session = boto3.Session(**session_params)
        self.client = session.client("bedrock-runtime")
        self.model_id = model_id
        self.system_prompt = system_prompt

        # Get model-specific configuration for optimized prompts
        self.model_config = ModelConfigFactory.get_model_config(model_id)

        logger.info(f"AWS Bedrock Converse provider initialized with model: {model_id}")
        logger.info(f"Model provider: {ModelConfigFactory.get_model_provider(model_id)}")
        if system_prompt:
            logger.info(f"System prompt configured: {system_prompt[:100]}...")

    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt for this provider instance."""
        self.system_prompt = system_prompt
        logger.info(f"System prompt updated: {system_prompt[:100]}...")

    def _build_system_config(self, custom_system: Optional[str] = None) -> Optional[List[Dict[str, str]]]:
        """Build system configuration for Converse API."""
        system_text = custom_system or self.system_prompt
        if system_text:
            return [{"text": system_text}]
        return None

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text using AWS Bedrock Converse API.

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            system_prompt: Optional system prompt (overrides instance system_prompt)

        Returns:
            Generated text
        """
        try:
            from app.core.config import settings

            # Build request parameters
            request_params = {
                "modelId": self.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": getattr(settings, 'llm_top_p', 0.4)
                }
            }

            # Configure reasoning for OpenAI models only
            is_openai = "openai" in self.model_id.lower() or "gpt" in self.model_id.lower()
            if is_openai:
                request_params["additionalModelRequestFields"] = {
                    "reasoning_effort": "medium"
                }

            # Add system prompt if configured
            system_config = self._build_system_config(system_prompt)
            if system_config:
                request_params["system"] = system_config

            response = self.client.converse(**request_params)

            # Extract text from unified response format
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])

            if content_blocks and len(content_blocks) > 0:
                return content_blocks[0].get("text", "").strip()

            return ""

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"AWS ClientError in generate: {error_code} - {e}")
            if error_code == 'ValidationException':
                raise ValueError(f"Invalid parameters for model {self.model_id}: {str(e)}")
            elif error_code == 'ThrottlingException':
                raise ConnectionError(f"Rate limit exceeded for model {self.model_id}")
            elif error_code == 'ServiceQuotaExceededException':
                raise ConnectionError(f"Service quota exceeded for model {self.model_id}")
            elif error_code == 'ModelNotReadyException':
                raise ValueError(f"Model {self.model_id} is not ready")
            elif error_code == 'ResourceNotFoundException':
                raise ValueError(f"Model {self.model_id} not found or not accessible")
            else:
                raise ConnectionError(f"AWS Bedrock error: {error_code}")

        except NoCredentialsError as e:
            logger.error(f"AWS credentials error in generate: {e}")
            raise ConnectionError("AWS credentials not configured or invalid")

        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error in generate: {e}")
            raise ConnectionError("Unable to connect to AWS Bedrock service")

        except KeyError as e:
            logger.error(f"Missing key in response: {e}")
            raise ValueError("Unexpected response format from LLM service")

        except Exception as e:
            logger.error(f"Unexpected error in generate: {e}")
            raise ConnectionError(f"LLM service error: {str(e)}")

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate text using AWS Bedrock Converse Stream API.

        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            system_prompt: Optional system prompt (overrides instance system_prompt)

        Yields:
            Text chunks as they are generated
        """
        try:
            from app.core.config import settings

            # Build request parameters
            request_params = {
                "modelId": self.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": getattr(settings, 'llm_top_p', 0.9)
                }
            }

            # Configure reasoning for OpenAI models only
            is_openai = "openai" in self.model_id.lower() or "gpt" in self.model_id.lower()
            if is_openai:
                request_params["additionalModelRequestFields"] = {
                    "reasoning_effort": "medium"
                }

            # Add system prompt if configured
            system_config = self._build_system_config(system_prompt)
            if system_config:
                request_params["system"] = system_config

            response = self.client.converse_stream(**request_params)

            # Process streaming response - ensure proper cleanup
            stream = response.get("stream")
            if stream is None:
                return

            try:
                yielded_count = 0
                for event in stream:
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
                        logger.debug(f"Stream metadata: {metadata}")

                    # Handle message stop event
                    elif "messageStop" in event:
                        stop_reason = event["messageStop"].get("stopReason")
                        logger.debug(f"Stream stopped: {stop_reason}")
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
            logger.error(f"AWS ClientError in generate_stream: {error_code} - {e}")
            if error_code == 'ValidationException':
                raise ValueError(f"Invalid parameters for model {self.model_id}: {str(e)}")
            elif error_code == 'ThrottlingException':
                raise ConnectionError(f"Rate limit exceeded for model {self.model_id}")
            elif error_code == 'ServiceQuotaExceededException':
                raise ConnectionError(f"Service quota exceeded for model {self.model_id}")
            elif error_code == 'ModelNotReadyException':
                raise ValueError(f"Model {self.model_id} is not ready")
            elif error_code == 'ResourceNotFoundException':
                raise ValueError(f"Model {self.model_id} not found or not accessible")
            else:
                raise ConnectionError(f"AWS Bedrock error: {error_code}")

        except NoCredentialsError as e:
            logger.error(f"AWS credentials error in generate_stream: {e}")
            raise ConnectionError("AWS credentials not configured or invalid")

        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error in generate_stream: {e}")
            raise ConnectionError("Unable to connect to AWS Bedrock service")

        except Exception as e:
            logger.error(f"Unexpected error in generate_stream: {e}")
            raise ConnectionError(f"LLM streaming service error: {str(e)}")

    def get_model_config(self):
        """
        Return model-specific configuration for optimized prompts.
        Uses the same configs as invoke_model (anthropic_models, meta_models, openai_models).
        """
        return self.model_config
