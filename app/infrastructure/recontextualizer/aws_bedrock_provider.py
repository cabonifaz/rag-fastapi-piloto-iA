import aioboto3
import json
import logging
import os
import asyncio
from typing import Optional, Dict, List
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from botocore.config import Config
from app.core.config import settings
from app.domain.ports.recontextualizer_port import RecontextualizerPort
from app.infrastructure.recontextualizer.model_factory import ModelFactory

# Configure logging
logger = logging.getLogger(__name__)


class QueryRecontextualizer(RecontextualizerPort):
    """
    Recontextualizes user queries by analyzing conversation history to:
    1. Resolve pronouns and implicit references
    2. Include necessary context from previous messages
    3. Create standalone, self-contained queries for better RAG retrieval

    Uses AWS Bedrock Converse API (non-streaming version).
    Supports multiple models: Claude (Anthropic), Llama (Meta), Nova (Amazon), GPT (OpenAI).
    Model configuration is automatically selected based on model_id.
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
        Initialize AWS Bedrock Converse client using aioboto3 (async) for query recontextualization.

        Args:
            region: AWS region (defaults to settings.aws_region)
            model_id: Bedrock model ID (defaults to settings.recontextualizer_model_id)
            profile_name: AWS profile name
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
        """
        self.region = region or settings.aws_region
        self.model_id = model_id or settings.recontextualizer_model_id

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

        # Configure botocore with connection and read timeouts
        self.boto_config = Config(
            connect_timeout=30,  # 30 seconds to establish connection
            read_timeout=120,    # 2 minutes max for reading response
            retries={'max_attempts': 2, 'mode': 'standard'}  # Retry failed requests
        )

        try:
            # Create aioboto3 session (don't create client yet)
            self.session = aioboto3.Session(**session_params)

            # Get model-specific configuration based on model_id
            self.model_config = ModelFactory.get_model_config(self.model_id)

            logger.info(f"QueryRecontextualizer initialized with model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize QueryRecontextualizer: {e}")
            raise RuntimeError(f"Could not connect to AWS Bedrock: {str(e)}")

    def _build_system_config(self) -> list:
        """Build system configuration for Converse API."""
        system_prompt = self.model_config.get_system_prompt()
        return [{"text": system_prompt}]

    async def recontextualize_query(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, any]:
        """
        Asynchronously recontextualizes the user query using conversation history with aioboto3 (truly async).

        Args:
            user_query: The user's query text.
            conversation_history: Optional list of recent message dicts with 'role' and 'content'.

        Returns:
            Dictionary with:
                - needs_context: bool (whether the query needed context)
                - response: str (the recontextualized query)
                - summary_intent: bool (whether user is asking for a summary)
            Returns a default dict with the original query if recontextualization fails.
        """
        # Default response if no conversation history
        if not conversation_history or len(conversation_history) == 0:
            logger.info("No conversation history provided, returning original query")
            return {
                "needs_context": False,
                "response": user_query,
                "summary_intent": False
            }

        try:
            # Build messages array using proper Converse API format
            # Convert conversation history to Converse messages format
            converse_messages = []
            for msg in conversation_history:
                converse_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}]
                })

            # Build the current query prompt with any model-specific instructions
            prompt = self.model_config.build_user_prompt(user_query, conversation_history)

            # Append current query as the latest user message
            converse_messages.append({
                "role": "user",
                "content": [{"text": prompt}]
            })

            # Build request parameters
            request_params = {
                "modelId": self.model_id,
                "messages": converse_messages,
                "system": self._build_system_config(),
                "inferenceConfig": {
                    "maxTokens": 1024,  # Sufficient for recontextualized queries
                    "temperature": 0.0,  # Low temperature for consistent recontextualization
                    "topP": 0.1
                }
            }

            # Use aioboto3 async client for truly non-blocking Bedrock calls
            async with self.session.client("bedrock-runtime", config=self.boto_config) as client:
                response = await client.converse(**request_params)

                # Extract the recontextualized query result
                result = self._extract_result(response)

            if result:
                logger.info(
                    f"Query recontextualized:\n"
                    f"  Original: {user_query}\n"
                    f"  Recontextualized: {result['response']}\n"
                    f"  Needs context: {result['needs_context']}\n"
                    f"  Summary intent: {result['summary_intent']}"
                )
                return result
            else:
                logger.warning("Failed to extract recontextualized query, returning original")
                return {
                    "needs_context": False,
                    "response": user_query,
                    "summary_intent": False
                }

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"AWS ClientError in QueryRecontextualizer: {error_code} - {e}")

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

            return {"needs_context": False, "response": user_query, "summary_intent": False}

        except NoCredentialsError as e:
            logger.error(f"AWS credentials error in QueryRecontextualizer: {e}")
            return {"needs_context": False, "response": user_query, "summary_intent": False}

        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error in QueryRecontextualizer: {e}")
            return {"needs_context": False, "response": user_query, "summary_intent": False}

        except asyncio.TimeoutError as e:
            logger.error(f"Timeout error in QueryRecontextualizer: {e}")
            return {"needs_context": False, "response": user_query, "summary_intent": False}

        except Exception as e:
            # Check if it's a timeout exception
            error_message = str(e)
            if "timed out" in error_message.lower() or "timeout" in error_message.lower():
                logger.error(f"Timeout error in QueryRecontextualizer: {e}")
            else:
                logger.error(f"Unexpected error in QueryRecontextualizer: {e}")

            return {"needs_context": False, "response": user_query, "summary_intent": False}

    def _extract_result(self, response) -> Optional[Dict[str, any]]:
        """
        Extract the recontextualized query from the Converse API response.

        Args:
            response: The response from bedrock_client.converse()

        Returns:
            Dictionary with needs_context, response, and summary_intent, or None if extraction fails.
        """
        # Delegate to the model config's extract_response method
        return self.model_config.extract_response(response)
