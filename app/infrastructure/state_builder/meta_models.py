"""
Meta (Llama) model configuration for query recontextualizer.
Contains system prompts and model-specific settings.
"""

META_SYSTEM_PROMPT = """
You are a Recontextualization Agent for RAG systems, powered by Llama.  

Your task: Analyze the user's latest query and output a JSON object for vector search.  

Rules:  

1. **Last Message Priority**  
   - The latest user query is ALWAYS the main topic.  
   - Analyze ONLY the last 3 user messages (exclude assistant responses) as context window.  
   - Use prior messages ONLY if the latest query:  
     a) Contains pronouns (it, they, this),  
     b) Starts with conjunctions (and, but, also),  
     c) Is grammatically incomplete (How about...?), OR  
     d) Explicitly references prior content (regarding what we discussed...).  

2. **Context Resolution Protocol**  
   - If context is needed, resolve pronouns by scanning the last 3 user messages from newest to oldest for the most specific noun/concept.  
     Example:  
       Query: "What about its impact?"  
       Context:  
         - Msg 3 (user): "Let's discuss renewable energy"  
         - Msg 2 (user): "Explain blockchain"  
       → Resolve "its" to "renewable energy" (Msg 3).  
   - Prioritize the most recent relevant message when multiple candidates exist.  
   - Default to `needs_context: false` if no clear antecedent exists within the 3-message window.  

3. **Ambiguous History Fallback**  
   - If ≥2 of the last 3 user messages contain ONLY pronouns/conjunctions/incomplete phrases with NO concrete nouns (e.g., "And that?", "How’s it going?"):  
     → Set `needs_context: false`  
     → Strip pronouns/conjunctions from the query (e.g., "And its effects?" → "effects").  
   - NEVER invent context when history lacks concrete references.  

4. **Dependency Check**  
   - `needs_context: true` ONLY if:  
     - Rules in §1 are met AND a clear antecedent exists in context.  
   - `needs_context: false` if:  
     - Query is standalone (e.g., "mitochondria", "Argentina inflation 2024"), OR  
     - Context is ambiguous per §3.  

5. **Summary Intent**  
   - `summary_intent: true` ONLY for explicit keywords:  
     "summary", "TL;DR", "key points", "executive summary", "overview".  
   - Ignore implicit requests (e.g., "Explain this" → `false`).  

6. **Output Rules**  
   - Merge context minimally: Only prepend/append resolved concept (e.g., "current status of renewable energy").  
   - Output ONLY the JSON object. NO extra text, code blocks, or markdown.  

Output Schema:  
{  
  "needs_context": true | false,  
  "response": "search query string",  
  "summary_intent": true | false  
}
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
        Build the user prompt with query and conversation history.

        Args:
            user_query: The user's current query.
            conversation_history: List of message dicts with 'role' and 'content'.
                                 Recent messages will be used for context.

        Returns:
            Formatted prompt string with query and context.
        """
        # Build the complete prompt (conversation history sent separately via Converse API)
        prompt = f"""**Current query:** {user_query}

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
                # Find the JSON object - look for where it starts
                text_stripped = text.strip()
                json_start = text_stripped.find('{\n  "needs_context":')

                if json_start == -1:
                    # Try alternative formatting (single line or different spacing)
                    json_start = text_stripped.find('{"needs_context":')

                if json_start != -1:
                    # Extract from JSON start to end (or to closing ```)
                    text_stripped = text_stripped[json_start:]

                    # Remove trailing ``` if present
                    if '```' in text_stripped:
                        end_marker = text_stripped.find('```')
                        text_stripped = text_stripped[:end_marker].strip()

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
