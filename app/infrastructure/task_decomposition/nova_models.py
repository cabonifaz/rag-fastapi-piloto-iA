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
        """Build Step 1 prompt: Detect API needs without building params."""

        system_prompt = """Task Summary - Step 1: Detection Only
You are a workflow action planner. Analyze user queries and generate a structured JSON object.
IMPORTANT: In this step, DO NOT fill params - leave params as empty objects {}. Just identify which endpoints are needed.

Model Instructions
You MUST respond in valid JSON format only. DO NOT provide any preamble, explanation, or markdown. Always include all fields in the JSON object. Use empty arrays or null values when appropriate.

Required JSON Structure
{
  "needs_context": "boolean",
  "context_messages": "number",
  "needs_system_data": "boolean",
  "system_calls": [
    {
      "entity": "string",
      "endpoint": "string",
      "method": "string"
    }
  ],
  "needs_external_knowledge": "boolean",
  "semantic_query": "string",
  "format": "table|list|null",
  "query_clean": "string"
}


Context Detection Rules
needs_context → true if query references previous interaction or conversation history
context_messages Calculation Rules:
- Single Message Reference → 1
- Multiple Messages Reference → 2-4
- Summary/Overview Request → 10-12
- Specific Topic Continuation → 5-7
- General Conversation Reference → 5

System Data Rules
needs_system_data → true only if the query requires internal data and mentions one or more columns that exist in an endpoint's "Columns" list from the API list.

system_calls → include endpoints ONLY when needed:
ENTITY = descriptive name of the entity from endpoint description
endpoint = API endpoint path (ONLY use relative paths from API list, NEVER full URLs)
method = GET or POST
params = EMPTY OBJECT {} (Step 1 does not build params)
missing_required_params = list all required parameters as missing

External Knowledge Rules
needs_external_knowledge → true if query requires external knowledge retrieval
semantic_query → preserve the original language and wording

Format Detection Rules
format → "table" if query contains "table" or "tabla"
format → "list" if query contains "list", "lista" OR ordering words
format → null in other cases

Other Rules
query_clean → always include original query text
JSON must be valid and complete

API Endpoints
""" + json.dumps(available_apis, indent=2) + """

Validation Rules
Ensure JSON is valid and complete. Verify endpoints exist in available list. For params, always use empty object {}. List ALL required parameters in missing_required_params.

Response Protocol
Respond with JSON object only. No additional text or explanations."""

        return system_prompt

    @staticmethod
    def build_step2_prompt(step1_result: Dict[str, Any], available_apis: Dict[str, Any]) -> str:
        """Build Step 2 prompt: Build params for identified API calls."""

        system_prompt = f"""Task Summary - Step 2: Parameter Building
You received an analysis that identified API calls needed. Now build the EXACT parameters for each API call.

Step 1 Result:
{json.dumps(step1_result, indent=2)}

Your task: For each API call in system_calls, fill in the params object with actual values from the query.

PARAMETER RESOLUTION RULES:
1. Explicit Parameters: Use values explicitly provided in the query
2. User Context Parameters: Use "my_user_id" for possessive user references
3. Missing Parameters: List required parameters not provided in missing_required_params
4. NO INFERENCE: Do not guess or infer parameter values

Required system_calls Structure
[
  {{
    "entity": "string",
    "endpoint": "string",
    "method": "string",
    "params": "object",
    "missing_required_params": "array"
  }}
]

API Endpoints Reference:
{json.dumps(available_apis, indent=2)}

Return the COMPLETE JSON with params filled in. Keep everything else the same from Step 1.

Response Protocol
Respond with JSON object only. No additional text or explanations."""

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

        print(step1_result)

        logger.info(f"Nova Step 1 complete - needs_system_data: {step1_result.get('needs_system_data')}, API calls: {len(step1_result.get('system_calls', []))}")

        # STEP 2: Only if system calls are needed, build params
        if step1_result.get('needs_system_data') and step1_result.get('system_calls'):
            logger.info("Nova Step 2: Building parameters...")
            step2_prompt = NovaConfig.build_step2_prompt(step1_result, available_apis)

            step2_body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": f"Build params for query: \"{user_query}\""}]
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
            print(step2_response_body)
            final_json = NovaConfig.extract_response(step2_response_body)

            logger.info(f"Nova Step 2 complete - Final response length: {len(final_json)}")
            return final_json
        else:
            # No API calls needed, return step 1 result
            logger.info("Nova: No API calls needed, returning Step 1 result")
            return step1_json