import aioboto3
import logging
import os
import asyncio
from typing import Optional, Dict, List
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from botocore.config import Config
from app.core.config import settings
from app.domain.ports.comparator_port import ComparatorPort
from app.infrastructure.query_comparator.model_factory import ModelFactory

# Configure logging
logger = logging.getLogger(__name__)


class QueryComparator(ComparatorPort):
    """
    Comparates user queries: the original and the recontextualized ones:
    1. Checks if both are the same for the vectorial search
    2. Checks if the user asks for a summary of the previous conversation

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
        Initialize AWS Bedrock Converse client using aioboto3 (async) for query comparison.

        Args:
            region: AWS region (defaults to settings.aws_region)
            model_id: Bedrock model ID (defaults to settings.recontextualizer_model_id)
            profile_name: AWS profile name
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
        """
        self.region = region or settings.aws_region
        self.model_id = model_id or settings.query_comparator_model_id

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
            connect_timeout=10,
            read_timeout=30,
            retries={'max_attempts': 0}
        )

        try:
            # Create aioboto3 session (don't create client yet)
            self.session = aioboto3.Session(**session_params)

            # Log session creation
            session_info = {k: '***' if 'key' in k.lower() or 'secret' in k.lower() else v
                           for k, v in session_params.items()}
            logger.info(f"✨ Created NEW aioboto3.Session (id: {id(self.session)}) [QueryComparator] | Config: {session_info}")

            # Get model-specific configuration based on model_id
            self.model_config = ModelFactory.get_model_config(self.model_id)

            logger.info(f"QueryComparator initialized with model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize QueryComparator: {e}")
            raise RuntimeError(f"Could not connect to AWS Bedrock: {str(e)}")

    def _build_system_config(self) -> list:
        """Build system configuration for Converse API."""
        system_prompt = self.model_config.get_system_prompt()
        return [{"text": system_prompt}]

    async def comparate_query(
        self,
        original_query: str,
    ) -> Dict:
        """
        Asynchronously classifies a query using AWS Bedrock.

        Args:
            original_query: The user's original query.

        Returns:
            Dict with keys:
                - needs_context: bool — True if the query requires prior context.
                - is_summary: bool — True if user is asking for a conversation recap.
            Returns {"needs_context": False, "is_summary": False} if the classification fails.
        """
        try:
            prompt = self.model_config.build_user_prompt(original_query)

            converse_messages = [{
                "role": "user",
                "content": [{"text": prompt}]
            }]

            request_params = {
                "modelId": self.model_id,
                "messages": converse_messages,
                "system": self._build_system_config(),
                "inferenceConfig": {
                    "maxTokens": 512,
                    "temperature": 0.1,
                    "topP": 0.9
                }
            }

            logger.info(
                f"QueryComparator | session={id(self.session)} | model={self.model_id}"
            )
            async with self.session.client("bedrock-runtime", config=self.boto_config) as client:
                response = await client.converse(**request_params)

            result = self._extract_result(response)

            if result:
                logger.info(
                    f"Query gatekeeper result: needs_context={result['needs_context']}, "
                    f"is_summary={result['is_summary']}"
                )
                return result

            logger.warning("Failed to extract gatekeeper result, using default")
            return {"needs_context": False, "is_summary": False}

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"AWS ClientError in QueryComparator: {error_code} - {e}")
            return {"needs_context": False, "is_summary": False}

        except NoCredentialsError as e:
            logger.error(f"AWS credentials error in QueryComparator: {e}")
            return {"needs_context": False, "is_summary": False}

        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error in QueryComparator: {e}")
            return {"needs_context": False, "is_summary": False}

        except asyncio.TimeoutError:
            logger.error("Timeout in QueryComparator")
            return {"needs_context": False, "is_summary": False}

        except Exception as e:
            logger.error(f"Unexpected error in QueryComparator: {e}")
            return {"needs_context": False, "is_summary": False}

    def _extract_result(self, response) -> Optional[Dict]:
        """
        Delegate response extraction to the model config.

        Args:
            response: The response from bedrock_client.converse()

        Returns:
            Dict with needs_context and is_summary, or None if extraction fails.
        """
        return self.model_config.extract_response(response)
