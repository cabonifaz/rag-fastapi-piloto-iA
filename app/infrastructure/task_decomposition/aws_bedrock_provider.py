import json
import logging
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

from app.infrastructure.task_decomposition.orchestrator_factory import OrchestratorConfigFactory
from app.core.config import settings

logger = logging.getLogger(__name__)


class OrchestratorQueryAnalyzer:
    """
    Orchestrator implementation for query analysis and task generation.

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
            session = boto3.Session(
                profile_name=settings.aws_profile if settings.aws_profile else None,
                aws_access_key_id=settings.aws_access_key_id if settings.aws_access_key_id else None,
                aws_secret_access_key=settings.aws_secret_access_key if settings.aws_secret_access_key else None
            )

            self.bedrock_client = session.client(
                service_name="bedrock-runtime",
                region_name=self.region
            )

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
        """Analyze user query using orchestrator model and return structured analysis."""
        try:
            # Use model-specific configuration to build request
            request_body = self.model_config.get_request_body(user_query, available_apis)

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response["body"].read())

            # Use model-specific extraction
            generated_text = self.model_config.extract_response(response_body)

            logger.info(f"Orchestrator response: {generated_text}")

            if not generated_text.strip():
                raise ValueError("Empty response from model")

            # Clean the generated text to remove smart quotes and other problematic characters
            cleaned_text = generated_text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")

            # Remove potential markdown code blocks if present (common in OpenAI responses)
            if cleaned_text.strip().startswith("```"):
                lines = cleaned_text.strip().split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned_text = '\n'.join(lines).strip()

            analysis = json.loads(cleaned_text)
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