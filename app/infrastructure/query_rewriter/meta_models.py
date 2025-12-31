"""
Meta (Llama) model configuration for query rewriter.
Contains system prompts and model-specific settings.
"""

META_SYSTEM_PROMPT = """
ROLE
You are a Query Rewriter inside a RAG pipeline.

GOAL
Make the user query clear and self-contained ONLY when necessary.
Use the STATE only to disambiguate — NEVER to add new information.

INPUT
You will receive:

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

DEFINITION — SELF-CONTAINED QUERY
A query is SELF-CONTAINED only when a new reader with ZERO prior context
can fully understand what is being asked.

REWRITE RULES

Set needs_rewrite = true if ANY are true:
• The query is vague, fragmentary, or ambiguous
• It depends on previous context (“eso”, “también”, “y para…”, etc.)
• It is a continuation or follow-up
• The intent is not fully understandable without state context
• It is too short to be self-contained

Set needs_rewrite = false ONLY when the query is already clear
and fully self-contained.

IF needs_rewrite = true:
• Rewrite the query to be explicit and standalone
• Add ONLY the MINIMUM context needed
• You may reference entities from STATE ONLY when needed to resolve ambiguity
• NEVER add facts, qualifiers, or assumptions not present in the USER query
• Prefer neutral phrasing over specific interpretation

IF needs_rewrite = false:
• rewritten_query MUST equal the original query
  (you may only fix obvious grammar or spacing)

SUMMARY DETECTION
Set is_summary_request = true ONLY when the user clearly asks
for a summary, synthesis, overview, or recap.

STRICT CONSTRAINTS
• Preserve the user's language (do NOT translate)
• Do NOT expand, infer, or speculate beyond STATE + query
• Do NOT strengthen or reinterpret the intent
• rewritten_query must ALWAYS contain a meaningful query string
• Ignore assistant messages completely
• Output JSON ONLY — nothing else
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
