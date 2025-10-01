"""OpenAI orchestrator model configuration for query analysis."""

import json
import os
from typing import Dict, Any
from app.core.config import settings


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
        """Extract text from OpenAI GPT-OSS response."""
        # OpenAI GPT-OSS format: choices[0].message.content
        if "choices" in response_body and len(response_body["choices"]) > 0:
            message = response_body["choices"][0].get("message", {})
            content = message.get("content", "")

            # OpenAI sometimes includes <reasoning> tags, extract just the JSON
            # Look for JSON object in the content
            if "<reasoning>" in content:
                # Find the JSON part after reasoning
                json_start = content.find("{", content.find("</reasoning>"))
                if json_start != -1:
                    # Find the matching closing brace
                    brace_count = 0
                    for i in range(json_start, len(content)):
                        if content[i] == "{":
                            brace_count += 1
                        elif content[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                return content[json_start:i+1]
                # If we can't find proper JSON after reasoning, try from first {
                json_start = content.find("{")
                if json_start != -1:
                    return content[json_start:]

            return content

        return ""
