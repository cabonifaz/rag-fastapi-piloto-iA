"""OpenAI orchestrator model configuration for query analysis."""

import json
import os
import logging
from typing import Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)


class OpenAIConfig:
    """Configuration for OpenAI orchestrator models (gpt-oss-20b-1 and gpt-oss-120b-1)."""

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
    def build_analysis_prompt(available_apis: Dict[str, Any]) -> str:
        """Build system prompt with Task Summary rules and APIs."""
        json_structure = OpenAIConfig._load_json_structure()

        system_prompt = """Task Summary
You are a workflow action planner. Analyze user queries and generate a structured JSON object describing the workflow requirements.

Model Instructions
You MUST respond in valid JSON format only. DO NOT provide any preamble, explanation, or markdown code blocks. Always include all fields in the JSON object. Use empty arrays or null values when appropriate.

Required JSON Structure
""" + json_structure + """

Context Detection Rules
needs_context → true if query references previous interaction or conversation history
context_messages Calculation Rules:
- Single Message Reference → 1
- Multiple Messages Reference → 2-4
- Summary/Overview Request → 10-12
- Specific Topic Continuation → 5-7
- General Conversation Reference → 5
Grammatical Precision Rules:
- Singular references → 1
- Plural references → 2-4
- Quantified references → use exact number mentioned
- Ambiguous references → 2
Decision Hierarchy: 1) Quantified references 2) Singular/Plural indicators 3) Reference type categories. Default: 2 for ambiguous cases.

System Data Rules
- needs_system_data → true only if the query requires internal data AND mentions one or more columns from the API list.
- system_calls → must ONLY include endpoints from the API list, never fabricate.
- Always respect the ENTITY-PARAMETER MATCHING FRAMEWORK:
  1. Entity alignment first
  2. Domain integrity: params must belong to the endpoint’s domain
  3. Semantic coherence required
  4. Reject superficial keyword matches
- If no system data is required, system_calls MUST be an empty array [].

Parameter Rules
1. Include only explicitly provided values.
2. For user references, use "my_user_id".
3. Missing required parameters must be listed under "missing_required_params".
4. Do NOT infer values.
5. Do NOT include undefined parameters.

External Knowledge Rules
needs_external_knowledge → true if query requires external knowledge retrieval

Semantic Query Rules
- semantic_query must always preserve the language of the original query.
- semantic_query must capture only the core semantic intent (what to retrieve or analyze).
- Remove all formatting, ordering, presentation, or style instructions.
- Keep filters, conditions, and entities that are essential to the meaning.
- If no extra formatting instructions exist, semantic_query = query_clean.

Combined Source Detection Rules
Set BOTH needs_system_data AND needs_external_knowledge to true for:
- Comparative Analysis Queries (system data vs external standards)
- Validation/Compliance Queries (system data meets external standards)
- Enhanced Recommendation Queries (system data + external knowledge)
- Hybrid Research Queries (internal metrics + external benchmarking)
Query Analysis Priority: 1) Detect system data needs 2) Detect external knowledge needs 3) Set both if query requires both

Format Rules
- format = "list" if "list", "lista", or ordering words appear.
- format = "table" if "table" or "tabla" mentioned.
- If needs_system_data = true and no explicit "list"/"lista" or ordering, set format = "table".
- format = null in all other cases.

Other Rules
query_clean → always include original query text
JSON must be valid and complete

API Endpoints
""" + json.dumps(available_apis, indent=2) + """

Validation Rules
Ensure JSON is valid and complete. Verify endpoints exist in available list. Confirm parameter handling follows "no inference" rule. Check semantic_query preserves original language. Validate combined source detection when query requires both system data and external knowledge.

Response Protocol
Respond with JSON object only. No markdown code blocks, no additional text or explanations. Return pure JSON."""


        return system_prompt

    @staticmethod
    def get_request_body(user_query: str, available_apis: Dict[str, Any]) -> Dict[str, Any]:
        """Get the request body for OpenAI models (gpt-oss) on AWS Bedrock."""

        system_prompt = OpenAIConfig.build_analysis_prompt(available_apis)

        return {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Process this query: \"{user_query}\""
                }
            ],
            "max_tokens": OpenAIConfig.MAX_TOKENS,
            "temperature": OpenAIConfig.TEMPERATURE,
            "top_p": OpenAIConfig.TOP_P
        }

    @staticmethod
    def extract_response(response_body: Dict[str, Any]) -> str:
        """Extract and clean text from OpenAI GPT-OSS response."""
        # OpenAI GPT-OSS format: choices[0].message.content
        if "choices" in response_body and len(response_body["choices"]) > 0:
            message = response_body["choices"][0].get("message", {})
            content = message.get("content", "")

            # OpenAI sometimes includes <reasoning> tags, extract just the JSON
            if "</reasoning>" in content:
                # Find where </reasoning> ends
                reasoning_end = content.find("</reasoning>") + len("</reasoning>")
                # Find the first { after </reasoning>
                json_start = content.find("{", reasoning_end)
                if json_start != -1:
                    # Find the matching closing brace
                    brace_count = 0
                    for i in range(json_start, len(content)):
                        if content[i] == "{":
                            brace_count += 1
                        elif content[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                content = content[json_start:i+1]
                                break
            else:
                # No reasoning tags, find first {
                json_start = content.find("{")
                if json_start != -1:
                    brace_count = 0
                    for i in range(json_start, len(content)):
                        if content[i] == "{":
                            brace_count += 1
                        elif content[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                content = content[json_start:i+1]
                                break

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

        return ""

    @staticmethod
    def analyze(bedrock_client, model_id: str, user_query: str, available_apis: Dict[str, Any]) -> str:
        """
        Analyze query using OpenAI model with single invocation.

        Args:
            bedrock_client: AWS Bedrock client instance
            model_id: Model identifier
            user_query: User's query text
            available_apis: Dictionary of available API endpoints

        Returns:
            Cleaned JSON string ready for parsing
        """
        # Build request body
        request_body = OpenAIConfig.get_request_body(user_query, available_apis)

        # Invoke bedrock
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )

        # Parse response
        response_body = json.loads(response["body"].read())

        # Extract and clean
        json_response = OpenAIConfig.extract_response(response_body)

        logger.info(f"OpenAI - Response length: {len(json_response)}")

        return json_response
