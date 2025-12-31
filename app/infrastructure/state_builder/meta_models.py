"""
Meta (Llama) model configuration for query recontextualizer.
Contains system prompts and model-specific settings.
"""

META_SYSTEM_PROMPT = """
# State Builder — Strict JSON Output Only

ROLE
You are a State Builder inside a RAG pipeline. Build a FOCUSED conversation state using ONLY USER messages. Ignore all assistant messages.

OUTPUT
Return ONLY valid JSON. No markdown, no comments, no extra text.
{
  "topic": "string", 
  "entities": ["string"],
  "goal": "string"
}

topic
- One broad technical category in ENGLISH, lowercase (1-2 words)
- Determined SOLELY by the MOST RECENT USER MESSAGE
- Default: "general query"

entities
- ALL explicit technical terms, standards, methodologies, or specific concepts from the MOST RECENT USER MESSAGE
- Copy EXACTLY as written (preserve case, accents, symbols, spacing)
- Do NOT translate
- Include compound terms that represent distinct technical concepts
- Exclude: generic verbs, prepositions, articles, pronouns, greetings
- Default: []

goal
- User intent from MOST RECENT USER MESSAGE as: verb + object
- Default: "seek information"

ENTITY EXTRACTION RULES - PRECISE DEFINITION
An "explicit technical term" is any phrase that:
✓ Is a specific technical methodology, standard, test, or procedure
✓ Represents a distinct engineering/geotechnical concept
✓ Appears as a noun phrase with technical meaning
✓ Is NOT a generic descriptive word ("high", "low", "important")
✓ Is NOT a grammatical connector ("del", "de la", "y", "con")

CONTEXT RULES - STRICT SINGLE-MESSAGE FOCUS
1. **EXCLUSIVE FOCUS**: Use ONLY the most recent user message for ALL fields
2. **COMPLETE ENTITY EXTRACTION**: Extract ALL qualifying technical terms from the message
3. **NO SEMANTIC MERGING**: Do NOT combine terms or infer relationships between entities
4. **VERBATIM PRESERVATION**: Copy terms exactly as written, including all modifiers
5. **HARD RESET**: Discard ALL previous state when processing new message

VALIDATION
- topic and goal must be non-empty strings
- entities must be a JSON array (may be empty)
- Every entity must appear as a contiguous substring in the most recent user message
- No entity may be a substring of another entity in the same array
- topic must be inferable from the most recent user message alone
"""


class MetaRecontextualizerConfig:
    """Configuration for Meta (Llama) models in query recontextualization."""

    @staticmethod
    def get_system_prompt():
        """Return the system prompt for Meta/Llama recontextualizer."""
        return META_SYSTEM_PROMPT

    @staticmethod
    def build_user_prompt(user_query: str, conversation_history: list) -> str:
        """
        Build the user prompt instruction.
        Note: conversation_history is passed as separate Converse API messagess.

        Args:
            user_query: The user's current query (not used in state building).
            conversation_history: List of message dicts (not used here, passed as Converse messages).

        Returns:
            Instruction string for the final user message.
        """
        return "Analyze the user messages in the conversation history above and extract the conversation state as JSON."

    @staticmethod
    def extract_response(response):
        """
        Extract the state from the Converse API response.

        Args:
            response: The response from bedrock_client.converse()

        Returns:
            Dictionary with keys:
                - topic: str (the conversation topic)
                - entities: list[str] (entities mentioned)
                - goal: str (user's goal)
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
                logger.warning("Empty content in state builder response")
                return None

            # Get the text from the first content block
            text = content[0].get("text", "").strip()

            if not text:
                logger.warning("Empty text in state builder response")
                return None

            logger.info(f"Raw LLM response text: {text}")

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

                if "topic" not in result or "entities" not in result or "goal" not in result:
                    logger.error(f"Missing required fields in response. Expected: topic, entities, goal. Got: {result.keys()}")
                    return None

                # Validate field types
                if not isinstance(result["topic"], str):
                    logger.warning(f"topic is not str: {type(result['topic'])}, converting")
                    result["topic"] = str(result["topic"])

                if not isinstance(result["entities"], list):
                    logger.warning(f"entities is not list: {type(result['entities'])}, converting")
                    result["entities"] = list(result["entities"]) if result["entities"] else []

                if not isinstance(result["goal"], str):
                    logger.warning(f"goal is not str: {type(result['goal'])}, converting")
                    result["goal"] = str(result["goal"])

                return result

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from state builder: {e}")
                logger.debug(f"Raw response text: {text}")
                return None

        except Exception as e:
            logger.error(f"Error extracting state builder response: {e}")
            return None
