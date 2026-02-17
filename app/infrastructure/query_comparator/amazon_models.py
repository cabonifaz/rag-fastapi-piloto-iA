"""
Amazon Nova model configuration for query comparator.
Contains system prompts and model-specific settings.
"""

AMAZON_SYSTEM_PROMPT = """
Compare two queries and output ONLY valid JSON with two booleans. No explanations, no extra text.

Rules:
1. same_info:
   - TRUE if the Original Query alone is sufficient for an accurate vector search,
     even if the Recontextualized Query has slightly different wording or small clarifications.
   - FALSE if the Recontextualized Query adds important context or details that make the Original Query insufficient for vector search
     (for example, clarifying ambiguous phrases, specifying scope, or adding missing context).

2. asks_for_summary:
   - TRUE only if the Original Query explicitly or implicitly requests a summary of the PREVIOUS CONVERSATION
     (phrases like "what did we discuss", "earlier", "before", "recap").
   - FALSE for summaries of a topic, document, or single query.

Output (strict JSON):
{
  "same_info": bool,
  "asks_for_summary": bool
}
"""


class AmazonComparatorConfig:
    """Configuration for Amazon Nova models in query comparison."""

    @staticmethod
    def get_system_prompt() -> str:
        """Return the system prompt for the query comparator."""
        return AMAZON_SYSTEM_PROMPT

    @staticmethod
    def build_user_prompt(original_query: str, recontextualized_query: str) -> str:
        """
        Build the user message containing both queries for comparison.

        Args:
            original_query: The user's original query.
            recontextualized_query: The recontextualized version of the query.

        Returns:
            Formatted prompt string with both queries tagged for comparison.
        """
        return (
            f"Original Query: <o>{original_query}</o>\n"
            f"Recontextualized Query: <r>{recontextualized_query}</r>"
        )

    @staticmethod
    def extract_response(response) -> dict | None:
        """
        Extract the comparison result from the Converse API response.

        Args:
            response: The response from bedrock_client.converse()

        Returns:
            Dictionary with keys:
                - same_info: bool (True if original query is sufficient for vector search)
                - asks_for_summary: bool (True if user asks for a conversation recap)
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

            if "same_info" not in result or "asks_for_summary" not in result:
                logger.error(f"Missing required fields in comparator response: {list(result.keys())}")
                return None

            if not isinstance(result["same_info"], bool):
                logger.warning(f"same_info is not bool ({type(result['same_info'])}), converting")
                result["same_info"] = bool(result["same_info"])

            if not isinstance(result["asks_for_summary"], bool):
                logger.warning(f"asks_for_summary is not bool ({type(result['asks_for_summary'])}), converting")
                result["asks_for_summary"] = bool(result["asks_for_summary"])

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from comparator response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting comparator response: {e}")
            return None
