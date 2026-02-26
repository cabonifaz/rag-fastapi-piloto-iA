"""
Amazon Nova model configuration for query comparator.
Contains system prompts and model-specific settings.
"""

AMAZON_SYSTEM_PROMPT = """
**Role:** Search Readiness Classifier.
**Task:** Analyze the "Current Query" to determine if it is a standalone search term or if it requires context/summarization.

**Instructions:**
1. **needs_context**: 
   - `false`: The query is a complete concept, a technical term, or a noun phrase with independent value. It does not need previous history to be useful as a search term.
   - `true`: The query is a dependent fragment, a single attribute, or a question about a "hidden" subject. It cannot be searched effectively on its own because it lacks the core entity.
2. **is_summary**: 
   - `true`: The user is explicitly asking to recap, list, or summarize the conversation history.
   - `false`: The user is asking for specific information or new data.

**Constraint:** Return ONLY a JSON object. No explanations.
**Output Format:** {"needs_context": boolean, "is_summary": boolean}
"""


class AmazonGatekeeperConfig:
    """Configuration for Amazon Nova models in query comparison."""

    @staticmethod
    def get_system_prompt() -> str:
        """Return the system prompt for the query comparator."""
        return AMAZON_SYSTEM_PROMPT

    @staticmethod
    def build_user_prompt(original_query: str) -> str:
        """
        Build the user message containing the query to classify.

        Args:
            original_query: The user's original query.

        Returns:
            The raw query string as the user message.
        """
        return original_query

    @staticmethod
    def extract_response(response) -> dict | None:
        """
        Extract the comparison result from the Converse API response.

        Args:
            response: The response from bedrock_client.converse()

        Returns:
            Dictionary with keys:
                - needs_context: bool (True if the query requires prior context to be searched)
                - is_summary: bool (True if user is asking for a conversation recap)
            Returns None if extraction fails.
        """
        import json
        import logging

        logger = logging.getLogger(__name__)

        try:
            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])

            if not content:
                logger.warning("Empty content in comparator response")
                return None

            text = content[0].get("text", "").strip()

            if not text:
                logger.warning("Empty text in comparator response")
                return None

            # Strip markdown code fences if present
            text_stripped = text
            if text_stripped.startswith("```"):
                start_idx = text_stripped.find("\n")
                end_idx = text_stripped.rfind("```")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    text_stripped = text_stripped[start_idx + 1:end_idx].strip()

            result = json.loads(text_stripped)

            if not isinstance(result, dict):
                logger.error(f"Comparator response is not a dict: {type(result)}")
                return None

            if "needs_context" not in result or "is_summary" not in result:
                logger.error(f"Missing required fields in gatekeeper response: {list(result.keys())}")
                return None

            if not isinstance(result["needs_context"], bool):
                logger.warning(f"needs_context is not bool ({type(result['needs_context'])}), converting")
                result["needs_context"] = bool(result["needs_context"])

            if not isinstance(result["is_summary"], bool):
                logger.warning(f"is_summary is not bool ({type(result['is_summary'])}), converting")
                result["is_summary"] = bool(result["is_summary"])

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from comparator response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting comparator response: {e}")
            return None
