"""
Anthropic model configuration for query recontextualizer.
Contains system prompts and model-specific settings.
"""

ANTHROPIC_SYSTEM_PROMPT = """
Role: Query Generator for Vector Search

You receive the last 7 messages of a conversation between a user and an assistant (3 previous turns + the current user message).

Task:
Generate a single standalone search query based on the user's last message, using the conversation history to understand what they are referring to.

Rules:
- The query must be optimized for vector similarity search against a knowledge base.
- Resolve pronouns, references, and ambiguities using the conversation context.
- Do not introduce information that is not supported by the conversation.
- Prefer precision over broadness.
- Same language as the user's last message.

Output:
- Only the final query.
- No reasoning. No analysis. No extra text.
- Format: |||query|||
"""


class AnthropicRecontextualizerConfig:
    """Configuration for Anthropic models in query recontextualization."""

    @staticmethod
    def get_system_prompt():
        """Return the system prompt for Anthropic recontextualizer."""
        return ANTHROPIC_SYSTEM_PROMPT

    @staticmethod
    def extract_response(response):
        """
        Extract the recontextualized query from the Converse API response.

        The prompt instructs the model to return: |||final standalone query|||
        This method parses that format and returns the rewritten query string.
        Falls back to returning raw text if no pipe delimiters are found.

        Args:
            response: The response from bedrock_client.converse()

        Returns:
            The rewritten query string, or None if extraction fails.
        """
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

            # Find the text block — skip thinking/reasoning blocks (reasoningContent key)
            text = next(
                (block.get("text", "").strip() for block in content if "text" in block),
                ""
            )

            if not text:
                logger.warning("Empty text in recontextualizer response")
                return None

            # Parse |||query||| format (model may output |, ||, or |||)
            match = re.search(r"\|+([^|]+)\|+", text)

            if not match:
                logger.warning(f"Missing pipe delimiters in response, using raw text: {text}")
                return text

            query = match.group(1).strip()

            if not query:
                logger.warning("Empty query between pipe delimiters")
                return None

            return query

        except Exception as e:
            logger.error(f"Error extracting recontextualizer response: {e}")
            return None
