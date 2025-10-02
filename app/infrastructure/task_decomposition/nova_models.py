"""Amazon Nova orchestrator model configuration for query analysis."""

import json
import os
import logging
from typing import Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)


class NovaConfig:
    """Configuration for Amazon Nova orchestrator models."""

    MODEL_ID = settings.orchestrator_model_id
    MAX_TOKENS = settings.orchestrator_max_tokens
    TEMPERATURE = settings.orchestrator_temperature
    TOP_P = settings.orchestrator_top_p

    @staticmethod
    def _load_json_structure() -> str:
        """Load JSON structure from external file."""
        current_dir = os.path.dirname(__file__)
        json_file = os.path.join(current_dir, "json_structure.json")

        try:
            with open(json_file, 'r') as f:
                structure = json.load(f)
            return json.dumps(structure, indent=2)
        except FileNotFoundError:
            return """{"needs_context": "boolean", "context_messages": "number", "needs_system_data": "boolean", "system_calls": [{"entity": "string", "endpoint": "string", "method": "string", "params": "object", "missing_required_params": "array"}], "needs_external_knowledge": "boolean", "semantic_query": "string", "format": "table|list|null", "query_clean": "string"}"""

    @staticmethod
    def build_step1_prompt(available_apis: Dict[str, Any]) -> str:
        """Build Step 1 prompt: Endpoint and Parameter Detection"""

        system_prompt = """Task Summary - Step 1: Endpoint and Parameter Detection
You are an API endpoint detector. Analyze user queries and identify which endpoints are needed with their parameters.

Model Instructions
Respond with valid JSON only. No explanations.

JSON Structure
{
  "system_calls": [
    {
      "entity": "string",
      "endpoint": "string",
      "method": "string",
      "params": {},
      "missing_required_params": []
    }
  ]
}

Rules
- ONLY include endpoints that directly match the query's subject matter
- If no endpoints match → system_calls = []
- params = {} (empty object for now)
- missing_required_params = list required parameter names that are missing from query

API Endpoints
""" + json.dumps(available_apis, indent=2) + """

Response Protocol  
JSON only. No additional text."""

        return system_prompt

    @staticmethod
    def build_step2_prompt(step1_result: Dict[str, Any], needs_system_data: bool, available_apis: Dict[str, Any]) -> str:
        """Build Step 2 prompt: Complete Workflow Analysis."""

        system_calls = step1_result.get('system_calls', [])

        system_prompt = f"""Task Summary - Step 2: Complete Workflow Analysis
You are a workflow analyzer. Complete the workflow JSON based on the original query.

Context from Step 1:
- needs_system_data = {str(needs_system_data).lower()}
- system_calls detected: {json.dumps(system_calls)}

Available API Endpoints:
{json.dumps(available_apis, indent=2)}

Model Instructions
Respond with valid JSON only. No explanations.

JSON Structure
{{
  "needs_context": "boolean",
  "context_messages": "number",
  "needs_external_knowledge": "boolean",
  "semantic_query": "string",
  "format": "table|list|null",
  "query_clean": "string"
}}

Rules
- needs_context: false (unless query references previous conversation)
- context_messages: 0
- needs_external_knowledge: true if system_calls array is empty OR if query needs additional knowledge beyond API data, false if API data is sufficient
- semantic_query: extract core search criteria without operations, preserve original language
- format: "table" if query contains "table"/"tabla", "list" if contains ordering words. If needs_system_data = true and no explicit "list"/"lista" or ordering, set format = "table". Otherwise null
- query_clean: original user query

Response Protocol
JSON only. No additional text."""

        return system_prompt


    @staticmethod
    def extract_response(response_body: Dict[str, Any]) -> str:
        """Extract and clean text from Amazon Nova response."""
        content = ""

        # Amazon Nova format: output.message.content[0].text
        if "output" in response_body and "message" in response_body["output"]:
            content_array = response_body["output"]["message"].get("content", [])
            content = content_array[0].get("text", "") if content_array else ""
        # Fallback to generic content format
        elif "content" in response_body:
            content_data = response_body["content"]
            if isinstance(content_data, list) and len(content_data) > 0:
                content = content_data[0].get("text", "")
            else:
                content = str(content_data)

        if not content:
            return ""

        # Clean smart quotes and special characters
        content = content.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")

        # Remove markdown code blocks if present
        if content.strip().startswith("```"):
            lines = content.strip().split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = '\n'.join(lines).strip()

        return content

    @staticmethod
    def analyze(bedrock_client, model_id: str, user_query: str, available_apis: Dict[str, Any]) -> str:
        """
        Analyze query using Nova model with TWO invocations.

        Step 1: Detect needs and identify APIs (no params)
        Step 2: Build parameters for identified APIs

        Args:
            bedrock_client: AWS Bedrock client instance
            model_id: Model identifier
            user_query: User's query text
            available_apis: Dictionary of available API endpoints

        Returns:
            Cleaned JSON string ready for parsing
        """
        # STEP 1: Detection - identify what's needed without params
        logger.info("Nova Step 1: Detecting needs...")
        step1_prompt = NovaConfig.build_step1_prompt(available_apis)

        step1_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": f"Process this query: \"{user_query}\""}]
                }
            ],
            "system": [{"text": step1_prompt}],
            "inferenceConfig": {
                "maxTokens": NovaConfig.MAX_TOKENS,
                "temperature": NovaConfig.TEMPERATURE,
                "topP": NovaConfig.TOP_P
            }
        }

        step1_response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(step1_body),
            contentType="application/json",
            accept="application/json"
        )

        step1_response_body = json.loads(step1_response["body"].read())
        step1_json = NovaConfig.extract_response(step1_response_body)
        step1_result = json.loads(step1_json)

        # Calculate needs_system_data by counting system_calls array
        system_calls = step1_result.get('system_calls', [])
        needs_system_data = len(system_calls) >= 1

        logger.info(f"Nova Step 1 complete - system_calls count: {len(system_calls)}, needs_system_data: {needs_system_data}")

        # STEP 2: Always run to get other fields (needs_context, semantic_query, etc.)
        logger.info("Nova Step 2: Complete workflow analysis...")
        step2_prompt = NovaConfig.build_step2_prompt(step1_result, needs_system_data, available_apis)

        step2_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": f"Process this query: \"{user_query}\""}]
                }
            ],
            "system": [{"text": step2_prompt}],
            "inferenceConfig": {
                "maxTokens": NovaConfig.MAX_TOKENS,
                "temperature": NovaConfig.TEMPERATURE,
                "topP": NovaConfig.TOP_P
            }
        }

        step2_response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(step2_body),
            contentType="application/json",
            accept="application/json"
        )

        step2_response_body = json.loads(step2_response["body"].read())
        step2_json = NovaConfig.extract_response(step2_response_body)
        step2_result = json.loads(step2_json)

        # Merge step1 + step2 + calculated needs_system_data into final structure
        final_result = {
            "needs_context": step2_result.get("needs_context", False),
            "context_messages": step2_result.get("context_messages", 0),
            "needs_system_data": needs_system_data,
            "system_calls": system_calls,
            "needs_external_knowledge": step2_result.get("needs_external_knowledge", True),
            "semantic_query": step2_result.get("semantic_query", user_query),
            "format": step2_result.get("format"),
            "query_clean": step2_result.get("query_clean", user_query)
        }

        final_json = json.dumps(final_result)
        logger.info(f"Nova Step 2 complete - Final merged result")
        return final_json