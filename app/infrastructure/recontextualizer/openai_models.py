"""
OpenAI model configuration for query recontextualizer.
Contains system prompts and model-specific settings.
"""

OPENAI_SYSTEM_PROMPT = """
You are a Recontextualization Agent for RAG systems.

Your task: Analyze the user's latest query and output a JSON object for vector search.

Rules:

1. Last Message Priority
   - The latest user query is ALWAYS the main topic.
   - Previous messages are ONLY relevant if the latest query is grammatically incomplete or contains pronouns.
   - A single complete word or phrase (noun, concept, or question) does NOT need context from history.

2. Dependency Check
   - Set needs_context: false if the query is a complete and understandable concept on its own.
   - Set needs_context: true ONLY if the query contains pronouns, conjunctions starting the sentence, or is grammatically incomplete.
   - If needs_context: false, return the query EXACTLY as written without any additions.
   - If needs_context: true, merge the query with minimal necessary context to form a coherent phrase.

3. Summary Intent
   - Set summary_intent: true ONLY if the user explicitly requests a summary, overview, or general explanation.
   - Otherwise, set summary_intent: false.

4. Output
   - Output ONLY the JSON object.
   - No extra text, code blocks, or markdown.
   - The "response" field is a search query for a vector database, NOT an answer.

Output Schema:
{
  "needs_context": true | false,
  "response": "search query string",
  "summary_intent": true | false
}
"""


class OpenAIRecontextualizerConfig:
    """Configuration for OpenAI models in query recontextualization."""

    @staticmethod
    def get_system_prompt():
        """Return the system prompt for OpenAI recontextualizer."""
        return OPENAI_SYSTEM_PROMPT

    @staticmethod
    def build_user_prompt(user_query: str, conversation_history: list) -> str:
        """
        Build the user prompt with query and conversation history.

        Args:
            user_query: The user's current query.
            conversation_history: List of message dicts with 'role' and 'content'.
                                 Recent messages will be used for context.

        Returns:
            Formatted prompt string with query and context.
        """
        # Build the complete prompt (conversation history sent separately via Converse API)
        prompt = f"""**Current query:** {user_query}

**Instruction:** Analyze whether the query requires recontextualization and respond accordingly."""

        return prompt

    @staticmethod
    def extract_response(response):
        """
        Extract the recontextualized query from the Converse API response.

        Args:
            response: The response from bedrock_client.converse()

        Returns:
            Dictionary with keys:
                - needs_context: bool (whether the query needed context)
                - response: str (the recontextualized query)
                - summary_intent: bool (whether user is asking for a summary)
            Returns None if extraction fails.
        """
        import json
        import logging
        import re

        logger = logging.getLogger(__name__)

        try:
            # Extract text from response
            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])

            if not content:
                logger.warning("Empty content in recontextualizer response")
                return None

            # Get the text from the first content block
            text = content[0].get("text", "").strip()

            if not text:
                logger.warning("Empty text in recontextualizer response")
                return None

            # Parse JSON response
            try:
                # Strip markdown code blocks if present (```json ... ```)
                text_stripped = text.strip()
                if text_stripped.startswith('```'):
                    # Find the first newline after opening ```
                    start_idx = text_stripped.find('\n')
                    # Find the closing ```
                    end_idx = text_stripped.rfind('```')
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        text_stripped = text_stripped[start_idx + 1:end_idx].strip()

                result = json.loads(text_stripped)

                # Validate required fields
                if not isinstance(result, dict):
                    logger.error(f"Response is not a dictionary: {type(result)}")
                    return None

                if "needs_context" not in result or "response" not in result or "summary_intent" not in result:
                    logger.error(f"Missing required fields in response: {result.keys()}")
                    return None

                # Validate field types
                if not isinstance(result["needs_context"], bool):
                    logger.warning(f"needs_context is not bool: {type(result['needs_context'])}, converting")
                    result["needs_context"] = bool(result["needs_context"])

                if not isinstance(result["response"], str):
                    logger.warning(f"response is not str: {type(result['response'])}, converting")
                    result["response"] = str(result["response"])

                if not isinstance(result["summary_intent"], bool):
                    logger.warning(f"summary_intent is not bool: {type(result['summary_intent'])}, converting")
                    result["summary_intent"] = bool(result["summary_intent"])

                return result

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from recontextualizer: {e}")
                logger.debug(f"Raw response text: {text}")
                return None

        except Exception as e:
            logger.error(f"Error extracting recontextualizer response: {e}")
            return None
