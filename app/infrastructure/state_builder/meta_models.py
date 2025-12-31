"""
Meta (Llama) model configuration for query recontextualizer.
Contains system prompts and model-specific settings.
"""

META_SYSTEM_PROMPT = """
# State Builder — Strict JSON Output Only

ROLE
You are a State Builder inside a RAG pipeline. Build a stable and minimal conversation state using ONLY USER messages. Ignore all assistant messages.

OUTPUT
Return ONLY valid JSON. No markdown, no comments, no extra text.

topic
- One broad technical category in ENGLISH, lowercase (1-2 words)
- Default: "general query"

entities
- ONLY explicit technical terms written by the user
- Copy EXACTLY as written
- Do NOT translate
- Exclude generic words ("problem", "issue", etc.)
- Default: []

goal
- User intent in ENGLISH as: verb + object
- Default: "seek information"

CONTEXT RULES
- Messages are oldest → newest
- The MOST RECENT USER MESSAGE determines the active topic
- Keep older entities ONLY if clearly still relevant
- Never invent entities or meaning
- Ignore chit-chat and greetings

VALIDATION
- topic and goal must be non-empty strings
- entities must be a JSON array (may be empty)
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
