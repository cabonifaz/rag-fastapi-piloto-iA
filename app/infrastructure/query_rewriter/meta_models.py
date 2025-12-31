"""
Meta (Llama) model configuration for query rewriter.
Contains system prompts and model-specific settings.
"""

META_SYSTEM_PROMPT = """
ROLE
You are a Query Rewriter in a RAG pipeline. Use the LAST USER QUERY to resolve ambiguities ONLY when necessary.

GOAL
Make queries self-contained ONLY when they cannot be understood without the immediate previous context.
Be extremely conservative - preserve original wording 95% of the time.

INPUT
1) STATE (context object)
{
  "topic": "string", 
  "entities": ["string"],
  "goal": "string"
}

2) USER QUERY (original language)

OUTPUT (JSON ONLY)
{
  "needs_rewrite": true | false,
  "rewritten_query": "string",
  "is_summary_request": true | false
}

REWRITE DECISION CRITERIA

✅ SET needs_rewrite = true ONLY IF:
• Query contains pronouns ("el", "eso", "esta") that clearly refer to elements in last_user_query
• Query starts with continuation words ("y", "también", "además") AND lacks standalone meaning
• Query is grammatically incomplete AND meaning depends on last_user_query
• Ambiguous terms in query map unambiguously to entities mentioned in last_user_query

❌ SET needs_rewrite = false IF:
• Query is a complete technical term or standard reference
• Query can be understood with basic domain knowledge alone
• Pronouns/continuations can be resolved by the retriever without rewriting
• Removing context words would change the query's relational meaning

REWRITE PRINCIPLES - CONTEXT-AWARE
When rewriting:
• ONLY use last_user_query to resolve SPECIFIC ambiguities (pronouns, implied subjects)
• Add MAX 3-5 words of context - never remove words indicating relationship
• Preserve ALL technical terminology exactly as written
• If ambiguity exists in last_user_query, do NOT rewrite
• rewritten_query must maintain the RELATIONSHIP to previous query when present

STRICT CONSTRAINTS
• If needs_rewrite = false, rewritten_query MUST be identical to original
• Max rewritten_query length = 130% of original
• NEVER remove continuation words ("y", "pero", "también") - only resolve what they refer to
• Output JSON ONLY
"""


class MetaRecontextualizerConfig:
    """Configuration for Meta (Llama) models in query recontextualization."""

    @staticmethod
    def get_system_prompt():
        """Return the system prompt for Meta/Llama recontextualizer."""
        return META_SYSTEM_PROMPT

    @staticmethod
    def build_user_prompt(user_query: str, state: dict = None) -> str:
        """
        Build the user prompt with query and state.

        Args:
            user_query: The user's current query.
            state: Conversation state from state builder with {topic, entities, goal}.

        Returns:
            Formatted prompt string with state and query.
        """
        import json

        # Build the prompt with state and query
        if state:
            state_json = json.dumps(state, indent=2, ensure_ascii=False)
            prompt = f"""STATE:
{state_json}

USER'S CURRENT QUERY:
{user_query}

Analyze and respond with the required JSON format."""
        else:
            # No state provided, use empty state
            empty_state = {"topic": "", "entities": [], "goal": ""}
            state_json = json.dumps(empty_state, indent=2)
            prompt = f"""STATE:
{state_json}

USER'S CURRENT QUERY:
{user_query}

Analyze and respond with the required JSON format."""

        return prompt

    @staticmethod
    def extract_response(response):
        """
        Extract the rewritten query from the Converse API response.

        Args:
            response: The response from bedrock_client.converse()

        Returns:
            Dictionary with keys:
                - needs_rewrite: bool (whether the query needed rewriting)
                - rewritten_query: str (the rewritten query)
                - is_summary_request: bool (whether user is requesting a summary)
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
                logger.warning("Empty content in query rewriter response")
                return None

            # Get the text from the first content block
            text = content[0].get("text", "").strip()

            if not text:
                logger.warning("Empty text in query rewriter response")
                return None

            logger.info(f"[Query Rewriter] Raw LLM response text: {text}")

            # Parse JSON response
            try:
                # Remove markdown code blocks if present
                text_stripped = text.strip()
                if text_stripped.startswith('```json'):
                    text_stripped = text_stripped[7:]
                elif text_stripped.startswith('```'):
                    text_stripped = text_stripped[3:]

                if text_stripped.endswith('```'):
                    text_stripped = text_stripped[:-3]

                text_stripped = text_stripped.strip()

                # Find JSON object
                json_start = text_stripped.find('{')
                if json_start != -1:
                    text_stripped = text_stripped[json_start:]

                    # Find matching closing brace
                    brace_count = 0
                    json_end = -1
                    for i, char in enumerate(text_stripped):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break

                    if json_end != -1:
                        text_stripped = text_stripped[:json_end]

                result = json.loads(text_stripped)

                # Validate required fields
                if not isinstance(result, dict):
                    logger.error(f"Response is not a dictionary: {type(result)}")
                    return None

                if "needs_rewrite" not in result or "rewritten_query" not in result or "is_summary_request" not in result:
                    logger.error(f"Missing required fields in response. Expected: needs_rewrite, rewritten_query, is_summary_request. Got: {result.keys()}")
                    return None

                # Validate field types
                if not isinstance(result["needs_rewrite"], bool):
                    logger.warning(f"needs_rewrite is not bool: {type(result['needs_rewrite'])}, converting")
                    result["needs_rewrite"] = bool(result["needs_rewrite"])

                if not isinstance(result["rewritten_query"], str):
                    logger.warning(f"rewritten_query is not str: {type(result['rewritten_query'])}, converting")
                    result["rewritten_query"] = str(result["rewritten_query"])

                if not isinstance(result["is_summary_request"], bool):
                    logger.warning(f"is_summary_request is not bool: {type(result['is_summary_request'])}, converting")
                    result["is_summary_request"] = bool(result["is_summary_request"])

                return result

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from query rewriter: {e}")
                logger.debug(f"Raw response text: {text}")
                return None

        except Exception as e:
            logger.error(f"Error extracting query rewriter response: {e}")
            return None
