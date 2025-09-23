import json
import logging
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

from app.domain.ports.task_decomposition_port import TaskDecompositionPort
from app.infrastructure.task_decomposition.mistral_models import MistralTaskDecompositionConfig
from app.core.config import settings

logger = logging.getLogger(__name__)


class AWSBedrockTaskDecompositionProvider(TaskDecompositionPort):
    """
    Generic AWS Bedrock implementation for task decomposition.
    Uses model-specific configurations for different LLMs.
    """

    def __init__(self, model_config_class=MistralTaskDecompositionConfig):
        self.model_config = model_config_class()
        self.model_id = model_config_class.MODEL_ID
        self.region = settings.aws_region

        # Initialize AWS Bedrock client
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

            logger.info(f"AWS Bedrock task decomposition provider initialized with model: {self.model_id}")

        except Exception as e:
            logger.error(f"Failed to initialize AWS Bedrock provider: {e}")
            raise RuntimeError(f"Could not connect to AWS Bedrock: {str(e)}")

    async def decompose_query(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        available_schemas: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Decompose user query into executable tasks using the configured model.
        """
        try:
            # Build the decomposition prompt using model-specific configuration
            prompt = self.model_config.build_decomposition_prompt(
                user_query, conversation_history, user_context, available_schemas
            )

            # Prepare the request body using model-specific configuration
            request_body = self.model_config.get_request_body(prompt)

            # Call model via Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )

            # Parse response
            response_body = json.loads(response["body"].read())
            generated_text = response_body.get("outputs", [{}])[0].get("text", "")

            logger.info(f"Model response: {generated_text}")

            # Extract and validate task chain using model-specific method
            tasks = self.model_config.extract_task_chain(generated_text)

            # Validate task chain
            validation = await self.validate_task_chain(tasks)
            if not validation["valid"]:
                logger.warning(f"Invalid task chain generated: {validation['errors']}")
                # Return model-specific fallback task chain
                return self.model_config.get_fallback_task_chain(user_query, available_schemas)

            return tasks

        except ClientError as e:
            logger.error(f"AWS Bedrock error in task decomposition: {e}")
            return self.model_config.get_fallback_task_chain(user_query, available_schemas)
        except Exception as e:
            logger.error(f"Error in task decomposition: {e}")
            return self.model_config.get_fallback_task_chain(user_query, available_schemas)

    async def analyze_intent(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze user intent using the configured model.
        """
        try:
            # Build intent analysis prompt using model-specific configuration
            prompt = self.model_config.build_intent_analysis_prompt(user_query, conversation_history)

            # Use model-specific request body but with reduced tokens for intent analysis
            request_body = self.model_config.get_request_body(prompt)
            request_body["max_tokens"] = 300  # Reduce tokens for intent analysis

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response["body"].read())
            generated_text = response_body.get("outputs", [{}])[0].get("text", "")

            # Parse intent from response using model-specific method
            intent = self.model_config.parse_intent_response(generated_text)
            return intent

        except Exception as e:
            logger.error(f"Error in intent analysis: {e}")
            # Return model-specific fallback intent
            return self.model_config.get_fallback_intent()

    async def validate_task_chain(
        self,
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate task chain structure and logic.
        """
        errors = []
        warnings = []

        if not tasks:
            errors.append("Task chain is empty")
            return {"valid": False, "errors": errors, "warnings": warnings}

        # Check if last task is llm_response
        if tasks[-1].get("action") != "llm_response":
            errors.append("Task chain must end with llm_response")

        # Check task order logic
        has_embedding = any(task.get("action") == "embedding" for task in tasks)
        has_retrieval = any(task.get("action") == "retrieval" for task in tasks)

        if has_retrieval and not has_embedding:
            errors.append("Retrieval task requires embedding task")

        # Check for required fields
        for i, task in enumerate(tasks):
            action = task.get("action")
            if not action:
                errors.append(f"Task {i} missing action field")

            if action == "embedding" and not task.get("input"):
                errors.append(f"Embedding task {i} missing input field")

            if action == "sql_select" and not task.get("query"):
                errors.append(f"SQL task {i} missing query field")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

