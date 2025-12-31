"""
Amazon Nova model configuration for query recontextualizer.
Contains system prompts and model-specific settings.
"""

AMAZON_SYSTEM_PROMPT = """
You are a Recontextualization Agent used inside a Retrieval-Augmented Generation (RAG) system.
Your task: Analyze the latest user message and the immediately preceding assistant message to generate a clear and minimal search query for a vector database.
Rules:
1. Focus on the latest user message.
2. Determine if it depends on the assistant's previous reply.
    - If the latest message is a complete concept or question → needs_context = false.
    - If it is incomplete, starts with a conjunction/pronoun, or clearly refers to the assistant's previous message → needs_context = true.
3. If needs_context = true: - Use the assistant's previous message only to recover missing context or meaning.
    - If the latest message introduces a new location, subject, or domain (e.g., "en planetas", "para animales", "en el mar"), treat it as a topic replacement — ignore any prior subjects completely.
    - When a topic replacement occurs, the final response must only include the new topic and discard older ones entirely.
    - Do NOT merge unrelated subjects or join them with conjunctions.
    - Do NOT invent new terms, assumptions, or unrelated content.
    - Use only words or direct synonyms from the user's and assistant's messages.
4. If the message explicitly requests a summary, overview, or general explanation, set summary_intent = true; otherwise false.
5. Output ONLY a valid JSON object in this exact schema:
{
    "needs_context": true | false,
    "response": "merged and complete query for vector search",
    "summary_intent": true | false
}
"""


class AmazonRecontextualizerConfig:
    """Configuration for Amazon Nova models in query recontextualization."""

    @staticmethod
    def get_system_prompt():
        """Return the system prompt for Amazon Nova recontextualizer."""
        return AMAZON_SYSTEM_PROMPT

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
