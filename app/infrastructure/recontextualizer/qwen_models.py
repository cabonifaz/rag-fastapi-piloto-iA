"""
Qwen model configuration for query recontextualizer.
Contains system prompts and model-specific settings.
"""

QWEN_SYSTEM_PROMPT = """
Role: Query Recontextualization Module (RAG)

Task:
Rewrite Turn 0 as a standalone, search-ready query using Turns -1 to -3.

Input: [Turn 0][Turn -1][Turn -2][Turn -3]
(Reverse chronological order)

Rules:
0. Ambiguity Gate (APPLY FIRST):
- Before applying any other rule, check if Turn 0 contains an indeterminate reference: ordinal expressions ("the first one", "the second"), vague demonstratives ("that", "this one", "the other one"), or any expression that could plausibly point to more than one referent in the conversation.
- If yes: STOP. Do not reason about what the user might mean. Do not correlate ordinals with the order of prior turns. Do not attempt resolution.
- Go directly to Query Reconstruction and replace only pronouns or implicit subjects with the Active Subject (from Rule 1). Preserve the rest of Turn 0 as literally as possible.
- If no: proceed to Rules 1-4 normally.

1. Active Subject Selection (Recency Rule):
- Starting from Turn -1 and moving backward, select the most recent explicit named entity that defines the subject.
- If a newer named entity appears, it overrides older ones.
- Ignore fragments (locations, attributes, short follow-ups) that do not redefine the subject.

2. Contextual Disambiguation:
- If the Active Subject is ambiguous (e.g., common first name, generic title), and the surrounding context clearly indicates a specific well-known entity, replace it with the precise canonical entity being referred to.
- If the Active Subject is a sub-event, phase, or component of a broader entity established earlier in the conversation, append that broader entity as a disambiguating modifier.
- Only disambiguate when the reference is unambiguous from context.
- Do not guess when ambiguity remains.
- Do not introduce specificity that is not explicitly supported by the conversation turns.

3. Query Reconstruction:
- Replace pronouns or fragments in Turn 0 with the fully disambiguated Active Subject.
- Expand minimal expressions into explicit search queries.
- Reconstruct the query in the most natural grammatical form that preserves the user's intent. Use common sense to infer the appropriate question structure.
- Do NOT introduce unrelated modifiers or speculative details.

4. Intent Preservation:
- Keep the exact informational intent of Turn 0 (definition, cause, impact, timeline, comparison, etc.).
- Do not broaden or reinterpret the question.

Output Requirements:
- Output ONLY the final standalone query.
- The query should sound human-like.
- No reasoning.
- No analysis.
- No extra text.
- Same language as Turn 0.
- Format strictly as:
|||final standalone query|||
"""


class QwenRecontextualizerConfig:
    """Configuration for Qwen models in query recontextualization."""

    @staticmethod
    def get_system_prompt():
        """Return the system prompt for Qwen recontextualizer."""
        return QWEN_SYSTEM_PROMPT

    @staticmethod
    def build_user_prompt(user_query: str, conversation_history: list) -> str:
        """
        Build the user prompt with 1-3 previous messages plus the current query.

        Args:
            user_query: The user's current query — always [Turn  0].
            conversation_history: 1-3 strings ordered oldest to newest,
                                  labeled [Turn -3] to [Turn -1] relative to Turn 0.

        Returns:
            Formatted prompt string with turns.
        """
        n = len(conversation_history)
        lines = [f'[Turn  0] "{user_query}"']
        lines += [f'[Turn {i - n:>2}] "{conversation_history[i]}"' for i in range(n - 1, -1, -1)]
        return "\n".join(lines)

    @staticmethod
    def extract_response(response):
        """
        Extract the recontextualized query from the Converse API response.

        The prompt instructs the model to return: |||final standalone query|||
        This method parses that format and returns the rewritten query string.

        Args:
            response: The response from bedrock_client.converse()

        Returns:
            The rewritten query string, or None if extraction fails.
        """
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

            # Find the text block — skip thinking/reasoning blocks (reasoningContent key)
            text = next(
                (block.get("text", "").strip() for block in content if "text" in block),
                ""
            )

            if not text:
                logger.warning("Empty text in recontextualizer response")
                return None

            # Parse |+query|+ format (model may output |, ||, or |||)
            import re
            match = re.search(r"\|+([^|]+)\|+", text)

            if not match:
                logger.error(f"Missing pipe delimiters in response: {text}")
                return None

            query = match.group(1).strip()

            if not query:
                logger.warning("Empty query between pipe delimiters")
                return None

            return query

        except Exception as e:
            logger.error(f"Error extracting recontextualizer response: {e}")
            return None
