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
        # read_timeout is higher to account for reasoning/thinking time
        self.boto_config = Config(
            connect_timeout=10,
            read_timeout=120,
            retries={'max_attempts': 0}
        )

        try:
            # Create aioboto3 session (don't create client yet)
            self.session = aioboto3.Session(**session_params)

            # Log session creation
            session_info = {k: '***' if 'key' in k.lower() or 'secret' in k.lower() else v
                           for k, v in session_params.items()}
            logger.info(f"✨ Created NEW aioboto3.Session (id: {id(self.session)}) [Recontextualizer] | Config: {session_info}")

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
        conversation_history: Optional[List[str]] = None
    ) -> str:
        """
        Asynchronously recontextualizes the user query using conversation history with aioboto3 (truly async).

        Args:
            user_query: The user's current query text (unused, kept for interface compatibility).
            conversation_history: List of 4 user message strings ordered oldest to newest.
                                 [0]=Turn -3, [1]=Turn -2, [2]=Turn -1, [3]=Turn 0.

        Returns:
            The rewritten query string. Returns the original query if recontextualization fails.
        """
        # Need at least 2 messages (one previous + current) to have something to recontextualize
        if not conversation_history:
            logger.info("No previous messages available, returning original query")
            return user_query

        try:
            # Build the user prompt with turn format
            prompt = self.model_config.build_user_prompt(user_query, conversation_history)

            # Single user message with the turn-formatted prompt
            converse_messages = [{
                "role": "user",
                "content": [{"text": prompt}]
            }]

            # Build request parameters
            request_params = {
                "modelId": self.model_id,
                "messages": converse_messages,
                "system": self._build_system_config(),
                "inferenceConfig": {
                    "maxTokens": 16000,
                    "temperature": 1,
                },
                "additionalModelRequestFields": {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 10000
                    }
                }
            }

            # Use aioboto3 async client for truly non-blocking Bedrock calls
            logger.info(
                f"♻️ Reusing session (id: {id(self.session)}) [Recontextualizer] | "
                f"Request params: model={self.model_id}, max_tokens=2048, temp=0.1, top_p=1"
            )
            async with self.session.client("bedrock-runtime", config=self.boto_config) as client:
                response = await client.converse(**request_params)

                # Extract the recontextualized query result
                result = self._extract_result(response)
                print(result)

            if result:
                logger.info(
                    f"Query recontextualized:\n"
                    f"  Original:    {user_query}\n"
                    f"  Rewritten:   {result}\n"
                )
                return result
            else:
                logger.warning("Failed to extract recontextualized query, returning original")
                return user_query

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

            return user_query

        except NoCredentialsError as e:
            logger.error(f"AWS credentials error in QueryRecontextualizer: {e}")
            return user_query

        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error in QueryRecontextualizer: {e}")
            return user_query

        except asyncio.TimeoutError as e:
            logger.error(f"Timeout error in QueryRecontextualizer: {e}")
            return user_query

        except Exception as e:
            # Check if it's a timeout exception
            error_message = str(e)
            if "timed out" in error_message.lower() or "timeout" in error_message.lower():
                logger.error(f"Timeout error in QueryRecontextualizer: {e}")
            else:
                logger.error(f"Unexpected error in QueryRecontextualizer: {e}")

            return user_query

    def _extract_result(self, response) -> Optional[str]:
        """
        Extract the recontextualized query from the Converse API response.

        The model returns QUERY::[rewritten_query]. The model config's extract_response
        parses this format and returns the query string.

        Args:
            response: The response from bedrock_client.converse()

        Returns:
            The rewritten query string, or None if extraction fails.
        """
        return self.model_config.extract_response(response)
