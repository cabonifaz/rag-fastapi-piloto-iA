import json
import logging
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

from app.infrastructure.task_decomposition.nova_models import OrchestratorConfig
from app.core.config import settings

logger = logging.getLogger(__name__)


class OrchestratorQueryAnalyzer:
    """Orchestrator implementation for query analysis and task generation."""

    def __init__(self):
        self.model_id = OrchestratorConfig.MODEL_ID
        self.region = settings.aws_region

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

            logger.info(f"Orchestrator query analyzer initialized with model: {self.model_id}")

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
            request_body = OrchestratorConfig.get_request_body(user_query, available_apis)

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response["body"].read())

            # Handle Nova Micro response format: output.message.content[0].text
            if "output" in response_body and "message" in response_body["output"]:
                content = response_body["output"]["message"].get("content", [])
                generated_text = content[0].get("text", "") if content else ""
            else:
                # Fallback to old format
                generated_text = response_body.get("content", [{}])[0].get("text", "")


            logger.info(f"Orchestrator response: {generated_text}")

            if not generated_text.strip():
                raise ValueError("Empty response from model")

            # Clean the generated text to remove smart quotes and other problematic characters
            cleaned_text = generated_text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")

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