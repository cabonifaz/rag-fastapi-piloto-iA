"""
Nova model configuration for context counter.
Contains system prompts and model-specific settings.
"""

NOVA_SYSTEM_PROMPT = """
You are a context detection module for a Retrieval-Augmented Generation (RAG) system.

You will receive two inputs:
- "query": the user's latest message.
- "context": the last few messages from the conversation history.

Your task:
1. Determine how many previous messages should be considered relevant according to the rules below.
2. Infer the main topic or concept the user is implicitly referring to if it’s not explicitly stated in the query.
3. Decide whether the inferred topic should be used for retrieval instead of the raw query.

Respond with ONLY a valid JSON object in the following format:
{
  "context_messages": <number>,
  "inferred_topic": "<string>",
  "use_inferred_topic": <true|false>
}

Context Detection Rules:
- Single Message Reference → 2
- Multiple Messages Reference → 4-8
- Summary/Overview Request → 16-20
- Specific Topic Continuation (explicit topic in query) → 8-12
- Implicit Topic Continuation (topic inferred from context) → 12-16
- General Conversation Reference → 10

Grammatical Precision Rules:
- Singular references → 1
- Plural references → 2-4
- Quantified references → use the exact number mentioned
- Ambiguous references → 2

Decision Hierarchy:
1) Quantified references
2) Singular/Plural indicators
3) Reference type categories
Default: 2 for ambiguous cases.

Inference Logic:
- If the query contains explicit topic words or clear subject terms → "use_inferred_topic": false
- If the query lacks clear topic indicators or refers vaguely (e.g., “eso”, “más sobre eso”, “dime más”, “qué pasa con eso”) → "use_inferred_topic": true
- When "use_inferred_topic" is true, derive the "inferred_topic" semantically from the provided "context".

Return ONLY the JSON object. No explanations. No other text.
"""


class NovaCounterConfig:
    """Configuration for Nova models in context counting."""

    @staticmethod
    def get_system_prompt():
        """Return the system prompt for Nova context counter."""
        return NOVA_SYSTEM_PROMPT

    @staticmethod
    def build_user_prompt(user_query: str, conversation_history: list) -> str:
        """
        Build the user prompt with query and conversation history.

        Args:
            user_query: The user's current query.
            conversation_history: List of message dicts with 'role' and 'content'.
                                 Only the last 3 messages will be used.

        Returns:
            Formatted prompt string with query and context.
        """
        # Limit to last 3 messages
        recent_messages = conversation_history[-3:] if conversation_history else []

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
        prompt = f"""query: {user_query}

context: {context_text}"""

        return prompt

    @staticmethod
    def get_converse_request(user_query: str, conversation_context: str, model_id: str):
        """
        Build the Converse API request parameters for Nova.

        Args:
            user_query: The user's query text.
            conversation_context: Recent conversation history formatted as a string.
            model_id: The model ID to use.

        Returns:
            Dictionary with request parameters for bedrock_client.converse()
        """
        # Construct prompt with both query and context
        prompt = f"""query: {user_query}

context: {conversation_context}"""

        return {
            "modelId": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            "system": [{"text": NOVA_SYSTEM_PROMPT}],
            "inferenceConfig": {
                "maxTokens": 200,  # Enough for response with inferred topic
                "temperature": 0.0,  # Deterministic output
                "topP": 1.0
            }
        }

    @staticmethod
    def extract_response(response):
        """
        Extract the context count, inferred topic, and usage flag from the Converse API response.

        Args:
            response: The response from bedrock_client.converse()

        Returns:
            Dictionary with keys:
                - context_messages: int (number of context messages needed)
                - inferred_topic: str (inferred topic if applicable)
                - use_inferred_topic: bool (whether to use inferred topic for retrieval)
        """
        import json
        import logging

        logger = logging.getLogger(__name__)

        default_response = {
            "context_messages": 0,
            "inferred_topic": "",
            "use_inferred_topic": False
        }

        try:
            # Extract text from response
            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])

            if not content:
                logger.warning("Empty content in context counter response")
                return default_response

            # Get the text from the first content block
            text = content[0].get("text", "").strip()

            if not text:
                logger.warning("Empty text in context counter response")
                return default_response

            # Parse JSON response
            result = json.loads(text)

            # Extract and validate context_messages
            count = result.get("context_messages", 0)
            if not isinstance(count, int) or count < 0:
                logger.warning(f"Invalid context count: {count}, defaulting to 0")
                count = 0

            # Cap at reasonable maximum
            if count > 50:
                logger.warning(f"Context count too high: {count}, capping at 50")
                count = 50

            # Extract inferred_topic
            inferred_topic = result.get("inferred_topic", "")
            if not isinstance(inferred_topic, str):
                logger.warning(f"Invalid inferred_topic type: {type(inferred_topic)}, defaulting to empty string")
                inferred_topic = ""

            # Extract use_inferred_topic
            use_inferred_topic = result.get("use_inferred_topic", False)
            if not isinstance(use_inferred_topic, bool):
                logger.warning(f"Invalid use_inferred_topic type: {type(use_inferred_topic)}, defaulting to False")
                use_inferred_topic = False

            return {
                "context_messages": count,
                "inferred_topic": inferred_topic.strip(),
                "use_inferred_topic": use_inferred_topic
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from context counter: {e}")
            logger.debug(f"Raw response text: {text if 'text' in locals() else 'N/A'}")
            return default_response
        except Exception as e:
            logger.error(f"Error extracting context counter response: {e}")
            return default_response
