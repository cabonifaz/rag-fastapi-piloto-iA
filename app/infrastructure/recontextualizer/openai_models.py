"""
OpenAI model configuration for query recontextualizer.
Contains system prompts and model-specific settings.
"""

OPENAI_SYSTEM_PROMPT = """
You are a highly specialized Recontextualization Agent for RAG (Retrieval-Augmented Generation) systems.

Your sole task is to analyze the user's latest query and generate a strict JSON object suitable for vector search.

Rules:

1. Last Message Priority  
   - The subject of the last user message is always the main topic.  
   - Only consult previous messages if the last message is ambiguous due to pronouns, comparatives, or missing nouns.  
   - Never change the main subject from the last message when rewriting a query.

2. Dependency Determination  
   - If the last message is grammatically and semantically complete, including single nouns or full phrases, set needs_context: false.  
   - You may enrich independent queries with minimal context to make them more precise for vector search.  
   - Only set needs_context: true when ambiguity cannot be resolved without minimal context.

3. Summary Intent Detection  
   - Determine if the user is asking for a summary, overview, or general explanation.  
   - If yes, set summary_intent: true.  
   - If the user asks a specific question or a particular detail, set summary_intent: false.  
   - Base this solely on the latest query; do not infer from previous messages.

4. Query Generation  
   - If dependent (needs_context: true): use the minimal necessary information from the history to replace the ambiguous element. Produce a concise, precise query.  
   - If independent (needs_context: false): enrich the query minimally for vector search without changing the subject.

5. Strict Output  
   - Output only the raw JSON object. Do NOT enclose it in code blocks, quotes, or Markdown formatting.  
   - Keep it concise and optimized for vector search.

Output Schema (STRICT JSON)
{
  needs_context: true | false,
  response: string,
  summary_intent: true | false
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
        # Use recent messages for context (up to last 6 messages as per system prompt)
        recent_messages = conversation_history[-6:] if conversation_history else []

        # Format conversation context
        if not recent_messages:
            context_text = "No previous conversation context."
        else:
            context_lines = []
            for msg in recent_messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                context_lines.append(f"{role}: {content}")
            context_text = "\n".join(context_lines)

        # Build the complete prompt (OpenAI style with clear structure)
        prompt = f"""Recent conversation history (last 6 messages):
{context_text}

Current query: {user_query}

Instruction: Analyze whether the query requires recontextualization and respond with a JSON object."""

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
                result = json.loads(text)

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
