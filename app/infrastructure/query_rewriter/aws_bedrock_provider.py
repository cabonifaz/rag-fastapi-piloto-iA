import aioboto3
import json
import logging
import os
import asyncio
from typing import Optional, Dict, List
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from botocore.config import Config
from app.core.config import settings
from app.domain.ports.query_rewriter import QueryRewriterPort
from app.infrastructure.query_rewriter.model_factory import ModelFactory

# Configure logging
logger = logging.getLogger(__name__)


class QueryRewriter(QueryRewriterPort):
    """
    Query rewriter that optimizes user queries for better RAG retrieval by:
    1. Expanding abbreviations and technical terms
    2. Adding relevant domain-specific keywords
    3. Reformulating complex questions into clearer search queries
    4. Generating multiple query variations for better coverage

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
        Initialize AWS Bedrock Converse client using aioboto3 (async) for query rewriting.

        Args:
            region: AWS region (defaults to settings.aws_region)
            model_id: Bedrock model ID (defaults to settings.query_rewriter_model_id)
            profile_name: AWS profile name
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
        """
        self.region = region or settings.aws_region
        self.model_id = model_id or settings.query_rewriter_model_id

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

            # Log session creation
            session_info = {k: '***' if 'key' in k.lower() or 'secret' in k.lower() else v
                           for k, v in session_params.items()}
            logger.info(f"✨ Created NEW aioboto3.Session (id: {id(self.session)}) [Query Rewriter] | Config: {session_info}")

            # Get model-specific configuration based on model_id
            self.model_config = ModelFactory.get_model_config(self.model_id)

            logger.info(f"QueryRewriter initialized with model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize QueryRewriter: {e}")
            raise RuntimeError(f"Could not connect to AWS Bedrock: {str(e)}")

    def _build_system_config(self) -> list:
        """Build system configuration for Converse API."""
        system_prompt = self.model_config.get_system_prompt()
        return [{"text": system_prompt}]

    async def rewrite_query(
        self,
        user_query: str,
        state: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """
        Rewrites user query using conversation state for better RAG retrieval with aioboto3 (truly async).

        Args:
            user_query: The user's query text to be rewritten.
            state: Optional conversation state from state builder with {topic, entities, goal}.

        Returns:
            Dictionary with:
                - needs_rewrite: bool (whether the query needed rewriting)
                - rewritten_query: str (the optimized query for RAG)
                - is_summary_request: bool (whether user is requesting a summary)
            Returns a default dict with the original query if rewriting fails.
        """
        # If query is too short, return as-is
        if len(user_query.strip()) < 3:
            logger.info("Query too short, returning original query")
            return {
                "needs_rewrite": False,
                "rewritten_query": user_query,
                "is_summary_request": False
            }

        try:
            # Build the query rewriting prompt with state
            prompt = self.model_config.build_user_prompt(user_query, state)

            # Build messages array using proper Converse API format
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
                    "maxTokens": 2048,  # Sufficient for rewritten queries
                    "temperature": 0.3,  # Slightly higher for creative query variations
                    "topP": 0.5
                }
            }

            # Use aioboto3 async client for truly non-blocking Bedrock calls
            logger.info(
                f"♻️ Reusing session (id: {id(self.session)}) [Query Rewriter] | "
                f"Request params: model={self.model_id}, max_tokens=2048, temp=0.3, top_p=0.5"
            )
            async with self.session.client("bedrock-runtime", config=self.boto_config) as client:
                response = await client.converse(**request_params)

                # Extract the rewritten query result
                result = self._extract_result(response)

            if result:
                logger.info(
                    f"Query rewriter result:\n"
                    f"  Original: {user_query}\n"
                    f"  Needs rewrite: {result.get('needs_rewrite', False)}\n"
                    f"  Rewritten: {result.get('rewritten_query', user_query)}\n"
                    f"  Is summary request: {result.get('is_summary_request', False)}"
                )
                return result
            else:
                logger.warning("Failed to extract rewritten query, returning original")
                return {
                    "needs_rewrite": False,
                    "rewritten_query": user_query,
                    "is_summary_request": False
                }

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"AWS ClientError in QueryRewriter: {error_code} - {e}")

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

            return {"needs_rewrite": False, "rewritten_query": user_query, "is_summary_request": False}

        except NoCredentialsError as e:
            logger.error(f"AWS credentials error in QueryRewriter: {e}")
            return {"needs_rewrite": False, "rewritten_query": user_query, "is_summary_request": False}

        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error in QueryRewriter: {e}")
            return {"needs_rewrite": False, "rewritten_query": user_query, "is_summary_request": False}

        except asyncio.TimeoutError as e:
            logger.error(f"Timeout error in QueryRewriter: {e}")
            return {"needs_rewrite": False, "rewritten_query": user_query, "is_summary_request": False}

        except Exception as e:
            # Check if it's a timeout exception
            error_message = str(e)
            if "timed out" in error_message.lower() or "timeout" in error_message.lower():
                logger.error(f"Timeout error in QueryRewriter: {e}")
            else:
                logger.error(f"Unexpected error in QueryRewriter: {e}")

            return {"needs_rewrite": False, "rewritten_query": user_query, "is_summary_request": False}

    def _extract_result(self, response) -> Optional[Dict[str, any]]:
        """
        Extract the rewritten query from the Converse API response.

        Args:
            response: The response from bedrock_client.converse()

        Returns:
            Dictionary with needs_rewrite, rewritten_query, and is_summary_request, or None if extraction fails.
        """
        # Delegate to the model config's extract_response method
        return self.model_config.extract_response(response)
