import boto3
import json
import logging
import os
import asyncio
from typing import Optional, Dict, List
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from botocore.config import Config
from app.core.config import settings
from app.infrastructure.context_counter.nova_models import NovaCounterConfig

# Configure logging
logger = logging.getLogger(__name__)


class ContextMessageCounter:
    """
    Analyzes a user query and conversation context to:
    1. Determine the number of past messages needed for context
    2. Infer the main topic when the user makes implicit references
    3. Decide whether to use the inferred topic for retrieval

    Uses AWS Bedrock Converse API (non-streaming version) with Nova models.
    """

    def __init__(
        self,
        region: Optional[str] = None,
        model_id: Optional[str] = None,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        """
        Initialize AWS Bedrock Converse client for the context counter.

        Args:
            region: AWS region (defaults to settings.aws_region)
            model_id: Bedrock model ID (defaults to settings.context_counter_model_id)
            profile_name: AWS profile name
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
        """
        self.region = region or settings.aws_region
        self.model_id = model_id or settings.context_counter_model_id

        session_params = {"region_name": self.region}

        # Use profile only in development, not in production with IAM roles
        profile = profile_name or settings.aws_profile
        if profile and os.getenv('ENVIRONMENT', settings.environment).lower() != 'production':
            session_params["profile_name"] = profile
        # If no profile, use direct credentials if available
        else:
            access_key = aws_access_key_id or settings.aws_access_key_id
            secret_key = aws_secret_access_key or settings.aws_secret_access_key
            if access_key and secret_key:
                session_params["aws_access_key_id"] = access_key
                session_params["aws_secret_access_key"] = secret_key

        # Configure boto3 with connection and read timeouts to prevent blocking
        boto_config = Config(
            connect_timeout=30,  # 30 seconds to establish connection
            read_timeout=120,    # 2 minutes max for reading response
            retries={'max_attempts': 2, 'mode': 'standard'}  # Retry failed requests
        )

        try:
            session = boto3.Session(**session_params)
            self.client = session.client("bedrock-runtime", config=boto_config)

            # Get model-specific configuration
            self.model_config = NovaCounterConfig

            logger.info(f"ContextMessageCounter initialized with model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize ContextMessageCounter: {e}")
            raise RuntimeError(f"Could not connect to AWS Bedrock: {str(e)}")

    def _build_system_config(self) -> list:
        """Build system configuration for Converse API."""
        system_prompt = self.model_config.get_system_prompt()
        return [{"text": system_prompt}]

    async def analyze_query(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, any]:
        """
        Asynchronously analyzes the user query and conversation context.

        Uses asyncio.to_thread to run blocking boto3 calls in a thread pool,
        preventing the FastAPI event loop from blocking.

        Args:
            user_query: The user's query text.
            conversation_history: Optional list of recent message dicts with 'role' and 'content'.

        Returns:
            Dictionary with:
                - context_messages: int (number of past messages needed)
                - inferred_topic: str (inferred topic if applicable)
                - use_inferred_topic: bool (whether to use inferred topic for retrieval)
        """
        try:
            # Build the user prompt with query and conversation history (limited to last 3 messages)
            prompt = self.model_config.build_user_prompt(
                user_query,
                conversation_history if conversation_history else []
            )

            # Build request parameters
            request_params = {
                "modelId": self.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "system": self._build_system_config(),
                "inferenceConfig": {
                    "maxTokens": 200,  # Enough for response with inferred topic
                    "temperature": 0.0,  # Deterministic output
                    "topP": 1.0
                }
            }

            # Run the blocking boto3 call in a thread pool to avoid blocking the event loop
            response = await asyncio.to_thread(
                self.client.converse,
                **request_params
            )

            # Extract the full response (count, topic, and flag)
            result = self._extract_result(response)

            logger.info(
                f"Context analysis - Messages: {result['context_messages']}, "
                f"Use inferred: {result['use_inferred_topic']}, "
                f"Topic: {result['inferred_topic'] if result['use_inferred_topic'] else 'N/A'}"
            )

            return result

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"AWS ClientError in ContextMessageCounter: {error_code} - {e}")

            if error_code == 'ValidationException':
                logger.error(f"Invalid parameters for model {self.model_id}: {str(e)}")
            elif error_code == 'ThrottlingException':
                logger.error(f"Rate limit exceeded for model {self.model_id}")
            elif error_code == 'ServiceQuotaExceededException':
                logger.error(f"Service quota exceeded for model {self.model_id}")
            elif error_code == 'ModelNotReadyException':
                logger.error(f"Model {self.model_id} is not ready")
            elif error_code == 'ResourceNotFoundException':
                logger.error(f"Model {self.model_id} not found or not accessible")

            return {"context_messages": 0, "inferred_topic": "", "use_inferred_topic": False}

        except NoCredentialsError as e:
            logger.error(f"AWS credentials error in ContextMessageCounter: {e}")
            return {"context_messages": 0, "inferred_topic": "", "use_inferred_topic": False}

        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error in ContextMessageCounter: {e}")
            return {"context_messages": 0, "inferred_topic": "", "use_inferred_topic": False}

        except asyncio.TimeoutError as e:
            logger.error(f"Timeout error in ContextMessageCounter: {e}")
            return {"context_messages": 0, "inferred_topic": "", "use_inferred_topic": False}

        except Exception as e:
            # Check if it's a timeout exception
            error_message = str(e)
            if "timed out" in error_message.lower() or "timeout" in error_message.lower():
                logger.error(f"Timeout error in ContextMessageCounter: {e}")
            else:
                logger.error(f"Unexpected error in ContextMessageCounter: {e}")

            return {"context_messages": 0, "inferred_topic": "", "use_inferred_topic": False}

    def _extract_result(self, response) -> Dict[str, any]:
        """
        Extract the context count, inferred topic, and usage flag from the Converse API response.

        Args:
            response: The response from bedrock_client.converse()

        Returns:
            Dictionary with context_messages, inferred_topic, and use_inferred_topic.
        """
        # Delegate to the model config's extract_response method
        return self.model_config.extract_response(response)
