"""
Anthropic model configuration for query recontextualizer.
Contains system prompts and model-specific settings.
"""

ANTHROPIC_SYSTEM_PROMPT = """
You are a search query generator, NOT an assistant.

Input: User message
Output: JSON with search query

The "response" field is NOT an answer. It is a query string for vector search.

Rules:
- Complete message → needs_context: false
- Incomplete message → needs_context: true
- Summary request → summary_intent: true
- "response" must be 2-10 words maximum

Output format:
{"needs_context": bool, "response": "query string", "summary_intent": bool}

You must ONLY output the JSON. Nothing else.
"""


class AnthropicRecontextualizerConfig:
    """Configuration for Anthropic models in query recontextualization."""

    @staticmethod
    def get_system_prompt():
        """Return the system prompt for Anthropic recontextualizer."""
        return ANTHROPIC_SYSTEM_PROMPT

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

        # Build the complete prompt
        prompt = f"""**Recent conversation history (last 6 messages):**
{context_text}

**Current query:** {user_query}

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
