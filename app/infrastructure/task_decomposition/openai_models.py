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
needs_system_data → true ONLY if the query explicitly mentions an entity that exactly matches an endpoint description.
needs_system_data → false for all other queries.

Entity Matching Rules:

1. Cross-Language Normalization
   - Use explicit translation dictionary for Spanish/English equivalents
   - Apply lemmatization only for plural/singular within same language
   - CRITICAL: Different concepts must never match

2. Exact Normalized Match
   - After normalization, entities must match endpoint components EXACTLY
   - Match against: table names, endpoint path segments, column names
   - Do NOT use substring matching
   - Case-insensitive comparison after normalization

3. No Semantic Expansion
   - Do NOT infer matches based on conceptual similarity
   - Only use explicitly defined translations from the normalization dictionary
   - Reject any matches not in the approved translation list

4. No Partial or Substring Matching
   - Column names must match the normalized entity EXACTLY
   - Substring overlaps are always rejected
   - Full normalized word comparison only

5. Contextual Validation
   - Evaluate the entire sentence/query context before matching
   - Ensure that the entity reference is domain-consistent and not accidental overlap

System Calls
- system_calls must ONLY include endpoints from the API list
- Endpoints are included only after exact normalized entity matching is confirmed
- If no entity match, system_calls = []

Safeguards Against False Positives
- Use explicit translation dictionary instead of automatic stemming
- Apply lemmatization only for plural/singular, not for word transformation
- Strict normalization: only approved translations allowed
- Reject substring matches entirely

Parameter Rules
1. Include only explicitly provided values from the query.
2. For user references, always use "my_user_id".
3. Any missing required parameters must be listed in missing_required_params.
4. Do NOT infer values under any circumstances.
5. Do NOT include undefined parameters or parameters outside the endpoint’s domain.

External Knowledge Rules
- needs_external_knowledge → true if the query requires information that cannot be answered exclusively from the available system data
- Always set to true when the query requests comparisons, definitions, explanations, or general knowledge unrelated to explicit API columns
- If system_calls = [] (no API match found), set to true
- Default to false only when the query can be fully answered with system data retrieved from the defined API endpoints
- For ambiguous cases where no API columns are mentioned but the query still expects an answer, set to true

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
    def get_converse_request(user_query: str, available_apis: Dict[str, Any]) -> Dict[str, Any]:
        """Get the request parameters for Converse API."""

        system_prompt = OpenAIConfig.build_analysis_prompt(available_apis)

        return {
            "modelId": OpenAIConfig.MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": f"Process this query: \"{user_query}\""}]
                }
            ],
            "system": [{"text": system_prompt}],
            "inferenceConfig": {
                "maxTokens": OpenAIConfig.MAX_TOKENS,
                "temperature": OpenAIConfig.TEMPERATURE,
                "topP": OpenAIConfig.TOP_P
            }
        }

    @staticmethod
    def extract_response(response_body: Dict[str, Any]) -> str:
        """Extract text from Converse API response."""
        # Converse API format: output.message.content
        output_message = response_body.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])

        # OpenAI models may return multiple content blocks (reasoning + text)
        # Find the block with "text" field (not "reasoningContent")
        for block in content_blocks:
            if "text" in block:
                return block.get("text", "").strip()

        return ""

    @staticmethod
    def analyze(bedrock_client, model_id: str, user_query: str, available_apis: Dict[str, Any]) -> str:
        """
        Analyze query using OpenAI model with Converse API.

        Args:
            bedrock_client: AWS Bedrock client instance
            model_id: Model identifier
            user_query: User's query text
            available_apis: Dictionary of available API endpoints

        Returns:
            Cleaned JSON string ready for parsing
        """
        # Build converse request parameters
        request_params = OpenAIConfig.get_converse_request(user_query, available_apis)

        # Call Converse API
        response = bedrock_client.converse(**request_params)

        # Extract and clean
        json_response = OpenAIConfig.extract_response(response)

        logger.info(f"OpenAI - Response length: {len(json_response)}")

        return json_response
