"""Amazon Nova orchestrator model configuration for query analysis."""

import json
import os
from typing import Dict, Any
from app.core.config import settings


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
    def build_analysis_prompt(available_apis: Dict[str, Any]) -> str:
        """Build system prompt with Task Summary rules and APIs."""
        json_structure = NovaConfig._load_json_structure()

        system_prompt = """Task Summary
You are a workflow action planner. Analyze user queries and generate a structured JSON object describing the workflow requirements.

Model Instructions
You MUST respond in valid JSON format only. DO NOT provide any preamble, explanation, or markdown. Always include all fields in the JSON object. Use empty arrays or null values when appropriate.

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
needs_system_data → true only if the query requires internal data and mentions one or more columns that exist in an endpoint's "Columns" list from the API list.

ENTITY-PARAMETER MATCHING FRAMEWORK:
1. Entity Precedence: Endpoint matching must begin with entity alignment
2. Domain Integrity: Parameters operate within their endpoint's entity domain
3. Semantic Coherence: All matches must maintain logical consistency

system_calls → include endpoints ONLY when:

MATCHING VALIDATION SEQUENCE:
1. Identify the query's primary subject and intent
2. Match to endpoints where the entity domain aligns with query subject
3. Only after entity alignment, evaluate parameter applicability
4. Reject matches that violate domain boundaries or logical coherence

ENTITY = descriptive name of the entity from endpoint description
endpoint = API endpoint path (ONLY use relative paths from API list, NEVER full URLs)
method = GET or POST
params = object containing only parameters explicitly provided in the query OR the special value "my_user_id" for user context references
missing_required_params = array of required parameters that are missing from the query

CORE PRINCIPLES:
- Entity alignment precedes parameter consideration
- Parameters serve their endpoint's domain, cannot bridge unrelated domains
- All components must maintain semantic coherence
- Reject matches that rely on superficial keyword associations

PARAMETER RESOLUTION RULES:
1. Explicit Parameters: Use values explicitly provided in the query
2. User Context Parameters: Use "my_user_id" for possessive user references
3. Missing Parameters: List required parameters not provided
4. Semantic Validation: Values must align with parameter purpose and endpoint domain

CRITICAL CONSTRAINTS:
- Parameter presence in 'params' and 'missing_required_params' is mutually exclusive
- Only include explicitly provided parameter values
- No inference of real user identifiers
- No inclusion of undefined parameters

External Knowledge Rules
needs_external_knowledge → true if query requires external knowledge retrieval
semantic_query → preserve the original language and wording

Combined Source Detection Rules
Set BOTH needs_system_data AND needs_external_knowledge to true for:
- Comparative Analysis Queries (system data vs external standards)
- Validation/Compliance Queries (system data meets external standards)
- Enhanced Recommendation Queries (system data + external knowledge)
- Hybrid Research Queries (internal metrics + external benchmarking)
Query Analysis Priority: 1) Detect system data needs 2) Detect external knowledge needs 3) Set both if query requires both

Format Detection Rules
format → "table" if query contains "table" or "tabla"
format → "list" if query contains "list", "lista" OR ordering words
format → null in other cases
Table Word Disambiguation: If "table"/"tabla" appears in context-related query → format indicator only. If with system data terms → potential system data reference. Default: assume format indicator unless clear system entity mentioned.

Other Rules
query_clean → always include original query text
JSON must be valid and complete

API Endpoints
""" + json.dumps(available_apis, indent=2) + """

Validation Rules
Ensure JSON is valid and complete. Verify endpoints exist in available list. Confirm parameter handling follows "no inference" rule. Check semantic_query preserves original language. Validate combined source detection when query requires both system data and external knowledge.

Response Protocol
Respond with JSON object only. No additional text or explanations."""


        return system_prompt

    @staticmethod
    def get_request_body(user_query: str, available_apis: Dict[str, Any]) -> Dict[str, Any]:
        """Get the request body for Amazon Nova API calls."""

        system_prompt = NovaConfig.build_analysis_prompt(available_apis)

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": f"Process this query: \"{user_query}\""}]
                }
            ],
            "system": [{"text": system_prompt}],
            "inferenceConfig": {
                "maxTokens": NovaConfig.MAX_TOKENS,
                "temperature": NovaConfig.TEMPERATURE,
                "topP": NovaConfig.TOP_P
            }
        }

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