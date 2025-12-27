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
        domain_context: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Rewrites user query to optimize it for RAG retrieval with aioboto3 (truly async).
        Expands terms, adds keywords, and creates query variations.

        Args:
            user_query: The user's query text to be rewritten.
            domain_context: Optional domain/industry context for better query optimization.

        Returns:
            Dictionary with:
                - rewritten_query: str (the optimized query for RAG)
                - variations: List[str] (alternative query formulations)
                - keywords: List[str] (extracted/added keywords)
            Returns a default dict with the original query if rewriting fails.
        """
        # If query is too short, return as-is
        if len(user_query.strip()) < 3:
            logger.info("Query too short, returning original query")
            return {
                "rewritten_query": user_query,
                "variations": [user_query],
                "keywords": []
            }

        try:
            # Build the query rewriting prompt
            if domain_context:
                prompt = f"Rewrite this query to optimize it for document retrieval in the domain of {domain_context}:\n\nQuery: {user_query}\n\nProvide the rewritten query, alternative variations, and key search terms."
            else:
                prompt = f"Rewrite this query to optimize it for document retrieval:\n\nQuery: {user_query}\n\nProvide the rewritten query, alternative variations, and key search terms."

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
                    "maxTokens": 512,  # Sufficient for rewritten queries
                    "temperature": 0.3,  # Slightly higher for creative query variations
                    "topP": 0.5
                }
            }

            # Use aioboto3 async client for truly non-blocking Bedrock calls
            logger.info(
                f"♻️ Reusing session (id: {id(self.session)}) [Query Rewriter] | "
                f"Request params: model={self.model_id}, max_tokens=512, temp=0.3, top_p=0.5"
            )
            async with self.session.client("bedrock-runtime", config=self.boto_config) as client:
                response = await client.converse(**request_params)

                # Extract the rewritten query result
                result = self._extract_result(response)

            if result:
                logger.info(
                    f"Query rewritten:\n"
                    f"  Original: {user_query}\n"
                    f"  Rewritten: {result.get('rewritten_query', user_query)}\n"
                    f"  Variations count: {len(result.get('variations', []))}\n"
                    f"  Keywords: {', '.join(result.get('keywords', []))}"
                )
                return result
            else:
                logger.warning("Failed to extract rewritten query, returning original")
                return {
                    "rewritten_query": user_query,
                    "variations": [user_query],
                    "keywords": []
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

            return {"rewritten_query": user_query, "variations": [user_query], "keywords": []}

        except NoCredentialsError as e:
            logger.error(f"AWS credentials error in QueryRewriter: {e}")
            return {"rewritten_query": user_query, "variations": [user_query], "keywords": []}

        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error in QueryRewriter: {e}")
            return {"rewritten_query": user_query, "variations": [user_query], "keywords": []}

        except asyncio.TimeoutError as e:
            logger.error(f"Timeout error in QueryRewriter: {e}")
            return {"rewritten_query": user_query, "variations": [user_query], "keywords": []}

        except Exception as e:
            # Check if it's a timeout exception
            error_message = str(e)
            if "timed out" in error_message.lower() or "timeout" in error_message.lower():
                logger.error(f"Timeout error in QueryRewriter: {e}")
            else:
                logger.error(f"Unexpected error in QueryRewriter: {e}")

            return {"rewritten_query": user_query, "variations": [user_query], "keywords": []}

    def _extract_result(self, response) -> Optional[Dict[str, any]]:
        """
        Extract the rewritten query from the Converse API response.

        Args:
            response: The response from bedrock_client.converse()

        Returns:
            Dictionary with rewritten_query, variations, and keywords, or None if extraction fails.
        """
        # Delegate to the model config's extract_response method
        return self.model_config.extract_response(response)
