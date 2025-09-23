"""
Mistral-specific model configurations and prompt templates for task decomposition.
"""

import json
from typing import Dict, List, Any, Optional


class MistralTaskDecompositionConfig:
    """Configuration and prompt templates for Mistral 7B task decomposition."""

    # Model configuration
    MODEL_ID = "mistral.mistral-7b-instruct-v0:2"
    MAX_TOKENS = 1000
    TEMPERATURE = 0.1  # Low temperature for consistent task generation
    TOP_P = 0.9
    STOP_SEQUENCES = ["</s>"]

    @staticmethod
    def build_decomposition_prompt(
        user_query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_context: Optional[Dict[str, Any]] = None,
        available_schemas: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build the task decomposition prompt for Mistral 7B.
        """
        # Base prompt template
        base_prompt = """You are a task decomposer. Analyze user questions and return JSON task arrays.

AVAILABLE TASKS:
- get_context: Extract conversation history
- embedding: Convert text to vector
- retrieval: Search vector database (only use vector_source)
- sql_select: Query structured data (ONLY when table schema provided)
- llm_response: Generate final answer (always last)

DECISION RULES:
1. Conversation reference ("anterior", "eso", "resume") → get_context
2. Knowledge/documentation question → embedding + retrieval
3. Structured data question → sql_select ONLY if table schema provided
4. NEVER mix retrieval with sql_select in same task
5. NEVER create sql_select without exact table schema
6. NO TABLE SCHEMA = NO SQL TASK (use vectorial only)

"""

        # Add table schemas if available
        if available_schemas:
            base_prompt += f"""TABLE SCHEMAS (when provided):
{json.dumps(available_schemas, indent=2)}

"""
        else:
            base_prompt += """NO TABLE SCHEMAS PROVIDED - Use vectorial search only for all data queries.

"""

        # Add conversation context if available
        conversation_context = ""
        if conversation_history:
            conversation_context = f"""CONVERSATION CONTEXT:
{json.dumps(conversation_history[-3:], indent=2)}

"""

        # Examples
        examples = """EXAMPLES:
Knowledge: "¿Qué es BGP?" → [{"action": "embedding", "input": "BGP protocol"}, {"action": "retrieval", "vector_source": "embedding_result"}, {"action": "llm_response", "instructions": ["explain clearly"]}]

Context: "Resume lo anterior" → [{"action": "get_context", "source": "conversation_history"}, {"action": "llm_response", "instructions": ["summarize"]}]

CRITICAL RULES:
- Questions about technical topics/documentation → VECTORIAL ONLY
- Questions about sales/inventory data → SQL ONLY if table schema provided
- NEVER combine retrieval + sql_select in same task
- NEVER create sql_select without exact table schema
- NO TABLE SCHEMA = NO SQL TASK (use vectorial only)

"""

        # Final query
        query_section = f"""DECOMPOSE: "{user_query}"

Return ONLY a valid JSON array of tasks. No explanations or additional text."""

        return base_prompt + conversation_context + examples + query_section

    @staticmethod
    def build_intent_analysis_prompt(
        user_query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build intent analysis prompt for Mistral 7B.
        """
        conversation_context = ""
        if conversation_history:
            conversation_context = f"Previous conversation: {json.dumps(conversation_history[-3:])}\n"

        return f"""Analyze this user query and classify the intent.

{conversation_context}
User query: "{user_query}"

Classify into one category:
1. vectorial_search - Questions about knowledge/documentation
2. database_search - Questions about structured data (sales, inventory, etc.)
3. conversation_reference - References to previous messages, summaries, translations
4. hybrid - Combination of multiple sources

Return JSON format:
{{
    "intent_type": "category",
    "requires_conversation": true/false,
    "requires_database": true/false,
    "requires_vectorial": true/false,
    "confidence": 0.0-1.0,
    "entities": ["entity1", "entity2"]
}}"""

    @staticmethod
    def get_request_body(prompt: str) -> Dict[str, Any]:
        """
        Get the request body for Mistral 7B API calls.
        """
        return {
            "prompt": prompt,
            "max_tokens": MistralTaskDecompositionConfig.MAX_TOKENS,
            "temperature": MistralTaskDecompositionConfig.TEMPERATURE,
            "top_p": MistralTaskDecompositionConfig.TOP_P,
            "stop": MistralTaskDecompositionConfig.STOP_SEQUENCES
        }

    @staticmethod
    def extract_task_chain(generated_text: str) -> List[Dict[str, Any]]:
        """
        Extract task chain from Mistral's generated text.
        """
        try:
            # Look for JSON array in the response
            start_idx = generated_text.find('[')
            end_idx = generated_text.rfind(']') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = generated_text[start_idx:end_idx]
                tasks = json.loads(json_str)

                # Ensure it's a list
                if isinstance(tasks, list):
                    return tasks

            # If no valid JSON found, return empty list
            return []

        except json.JSONDecodeError:
            return []

    @staticmethod
    def parse_intent_response(generated_text: str) -> Dict[str, Any]:
        """
        Parse intent analysis response from Mistral.
        """
        try:
            # Look for JSON object in the response
            start_idx = generated_text.find('{')
            end_idx = generated_text.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = generated_text[start_idx:end_idx]
                intent = json.loads(json_str)
                return intent

            # Fallback intent
            return MistralTaskDecompositionConfig.get_fallback_intent()

        except json.JSONDecodeError:
            return MistralTaskDecompositionConfig.get_fallback_intent()

    @staticmethod
    def get_fallback_intent() -> Dict[str, Any]:
        """
        Get fallback intent when parsing fails.
        """
        return {
            "intent_type": "vectorial_search",
            "requires_conversation": False,
            "requires_database": False,
            "requires_vectorial": True,
            "confidence": 0.5,
            "entities": []
        }

    @staticmethod
    def get_fallback_task_chain(
        user_query: str,
        available_schemas: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a simple fallback task chain when Mistral fails.
        """
        # Default to vectorial search
        return [
            {
                "action": "embedding",
                "input": user_query
            },
            {
                "action": "retrieval",
                "vector_source": "embedding_result"
            },
            {
                "action": "llm_response",
                "instructions": ["provide helpful answer", "use retrieved context"]
            }
        ]