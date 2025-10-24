import json
import logging
from typing import List, Dict, Any, Optional
import aioboto3
from botocore.exceptions import ClientError

from app.infrastructure.task_decomposition.orchestrator_factory import OrchestratorConfigFactory
from app.core.config import settings

logger = logging.getLogger(__name__)


class OrchestratorQueryAnalyzer:
    """
    Orchestrator implementation for query analysis and task generation using aioboto3 (async).

    Supports multiple model families via factory pattern:
    - OpenAI: openai.gpt-oss-20b-1:0, openai.gpt-oss-120b-1:0
    - Amazon Nova: us.amazon.nova-micro-v1:0, us.amazon.nova-lite-v1:0, etc.
    """

    def __init__(self):
        self.model_id = settings.orchestrator_model_id
        self.region = settings.aws_region

        # Get model-specific configuration using factory pattern
        self.model_config = OrchestratorConfigFactory.get_config(self.model_id)

        try:
            # Create aioboto3 session (don't create client yet)
            session_params = {
                "region_name": self.region
            }

            if settings.aws_profile:
                session_params["profile_name"] = settings.aws_profile
            elif settings.aws_access_key_id and settings.aws_secret_access_key:
                session_params["aws_access_key_id"] = settings.aws_access_key_id
                session_params["aws_secret_access_key"] = settings.aws_secret_access_key

            self.session = aioboto3.Session(**session_params)

            logger.info(f"Orchestrator query analyzer initialized with model: {self.model_id} "
                       f"(Provider: {OrchestratorConfigFactory.get_model_provider(self.model_id)})")

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator provider: {e}")
            raise RuntimeError(f"Could not connect to AWS Bedrock: {str(e)}")

    async def analyze_query(
        self,
        user_query: str,
        available_apis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze user query using orchestrator model and return structured analysis (async)."""
        try:
            # Use aioboto3 async client for Bedrock calls
            async with self.session.client("bedrock-runtime") as bedrock_client:
                # Use model-specific analyze method (now async with await)
                json_response = await self.model_config.analyze(
                    bedrock_client=bedrock_client,
                    model_id=self.model_id,
                    user_query=user_query,
                    available_apis=available_apis
                )

                if not json_response.strip():
                    raise ValueError("Empty response from model")

                analysis = json.loads(json_response)
                return analysis

        except (ClientError, json.JSONDecodeError) as e:
            logger.error(f"Error in query analysis: {e}")
            return self._get_fallback_analysis(user_query)

    def _get_fallback_analysis(self, user_query: str) -> Dict[str, Any]:
        """Fallback analysis when orchestrator fails."""
        return {
            "needs_context": False,
            "context_messages": 0,
            "needs_system_data": False,
            "system_calls": [],
            "needs_external_knowledge": True,
            "semantic_query": user_query,
            "format": None,
            "query_clean": user_query
        }