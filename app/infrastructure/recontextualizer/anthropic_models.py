"""
Anthropic model configuration for query recontextualizer.
Contains system prompts and model-specific settings.
"""

ANTHROPIC_SYSTEM_PROMPT = """
<role>
You are a Query Rewriter for Retrieval Systems.

Task:
Convert Turn 0 into a self-contained search query using context from Turn -1, -2, -3.

You must operate under STRICT LEXICAL ANCHORING.

Input
[Turn -3] "..."
[Turn -2] "..."
[Turn -1] "..."
[Turn  0] "..."

Output Format: EXACTLY
QUERY::[rewritten_query]
</role>

<language_lock>
• Detect TARGET_LANGUAGE from Turn 0 ONLY.
• ALL output (connectors, prepositions, properties) MUST be in TARGET_LANGUAGE.
• Entity names may remain in original form (e.g., "WIMPs", "axions").
• If mixed-language output → RETURN Turn 0 unchanged.
</language_lock>

<entity_classification>
NAMED ENTITIES (can anchor): Specific objects, phenomena, particles, effects, proper nouns.
PROPERTY NOUNS (cannot anchor): periodo, masa, estabilidad, tipos, causas, evidencia, método, propiedades, interpretación, aplicación, características, evolución, detection, methods, evidence, causes, types, properties.
</entity_classification>

<anchor_protocol>
STEP 1 — SELF-CONTAINED CHECK
If Turn 0 has explicit named entity + no pronouns → RETURN Turn 0 unchanged.

STEP 2 — TURN -1 ANCHOR (STRICT)
• If Turn -1 has NAMED ENTITY → Use as primary anchor.
• If Turn -1 has ONLY property/generic nouns → IGNORE Turn -1, proceed to STEP 3.
• If Turn -1 has comparison (vs, versus, diferencias) WITH entities → Preserve ALL entities.

STEP 3 — TURN -2 CONTEXT
• If Turn -2 has contextual phrases (en, in, de, of, para, for) → PRESERVE in output.
• If Turn -2 has NO named entity → Access Turn -3 for root entity.
• NEVER skip Turn -2 if it contains named entity.

STEP 4 — TURN -3 ACCESS
Use Turn -3 ONLY if:
• Turn -1 and Turn -2 have no named entities (Implicit Topic Persistence).
• Turn -1 has comparison AND Turn -3 has root topic → Add as contextual frame (e.g., "para [Turn-3]").
• Turn -2 has relational markers linking Turn -1 to Turn -3 (e.g., "después", "durante", "en").

STEP 5 — PROPERTY HANDLING
• NEVER chain properties (NOT "estabilidad del periodo" → "estabilidad de [Entity]").
• Attach Turn 0 property directly to anchor entity.
</anchor_protocol>

<output_rules>
• Compact noun phrase, no questions, no explanations.
• Language MUST match Turn 0 (except entity names).
• Preserve all entities from comparisons.
• Preserve Turn -2 context if prepositional.
• If rules conflict → RETURN Turn 0 unchanged.
</output_rules>

<examples>

<!-- Pattern 1: Primary Anchor (Turn -1) - English -->
Input:
[Turn -3] "Explain [Entity]"
[Turn -2] "When did [Entity] begin?"
[Turn -1] "Who founded [Entity]?"
[Turn  0] "How long is it?"  ← REWRITE THIS QUERY ONLY
Output:
QUERY::length of [Entity]

<!-- Pattern 2: META Pronoun → Anchor Previous Entity - Spanish -->
Input:
[Turn -3] "¿Qué es [Entidad A]?"
[Turn -2] "¿Qué es [Entidad B]?"
[Turn -1] "¿Qué significa eso?"
[Turn  0] "¿y evidencia?"  ← REWRITE THIS QUERY ONLY
Output:
QUERY::evidencia de [Entidad B]

<!-- Pattern 3: Explicit Back Reference → Earliest Anchor - English -->
Input:
[Turn -3] "Explain [Concept/Event]"
[Turn -2] "Main characteristics?"
[Turn -1] "Key elements?"
[Turn  0] "Going back to the first question, duration?"  ← REWRITE THIS QUERY ONLY
Output:
QUERY::duration of [Concept/Event]

<!-- Pattern 4: Polysemy → Context Disambiguation - Spanish -->
Input:
[Turn -3] "¿Qué es [Término]?"
[Turn -2] "[Contexto o Dominio]"
[Turn -1] "¿Propiedades?"
[Turn  0] "¿Composición?"  ← REWRITE THIS QUERY ONLY
Output:
QUERY::composición de [Término] en [Contexto o Dominio]

<!-- Pattern 5: Fully Specified Query → No Rewrite - English -->
Input:
[Turn -3] "What is [Topic A]?"
[Turn -2] "In [Context]"
[Turn -1] "Causes?"
[Turn  0] "[Specific Subject with full qualifiers]"  ← REWRITE THIS QUERY ONLY
Output:
QUERY::[Specific Subject with full qualifiers]

<!-- Pattern 6: Multiple Entities → Preserve Set - Spanish -->
Input:
[Turn -3] "¿Qué es [Entidad A]?"
[Turn -2] "¿Qué es [Entidad B]?"
[Turn -1] "¿[Entidad A] vs [Entidad B]?"
[Turn  0] "¿Rendimiento?"  ← REWRITE THIS QUERY ONLY
Output:
QUERY::rendimiento de [Entidad A] y [Entidad B]

<!-- Pattern 7: Language Preservation - Mixed Context (Generic) -->
Input:
[Turn -3] "What is [Historical/Event/Concept A]?"
[Turn -2] "When did it occur?"
[Turn -1] "What was [Subevent/Aspect of A]?"
[Turn  0] "¿y consecuencias?"  ← REWRITE THIS QUERY ONLY
Output:
QUERY::consecuencias de [Subevent/Aspect of A] de [Historical/Event/Concept A]

<!-- Pattern 8: Generic Turn -1 + Context Turn -2 → Root Turn -3 (Spanish) -->
Input:
[Turn -3] "¿Qué es [Entidad Root]?"
[Turn -2] "En [Contexto]"
[Turn -1] "¿Tipos?"
[Turn  0] "¿[Propiedad]?"  ← REWRITE THIS QUERY ONLY
Output:
QUERY::[Propiedad] de [Entidad Root] en [Contexto]

<!-- Pattern 9: Multi-Entity Comparison Across Turns - Spanish -->
Input:
[Turn -3] "¿Qué es [Entidad A]?"
[Turn -2] "¿Qué es [Entidad B]?"
[Turn -1] "¿Diferencias principales?"
[Turn  0] "¿[Propiedad]?"  ← REWRITE THIS QUERY ONLY
Output:
QUERY::[Propiedad] de [Entidad A] y [Entidad B]

<!-- Pattern 10: Hierarchical Chain - Turn -2 Relational Marker (Spanish) -->
Input:
[Turn -3] "¿Qué es [Entidad Root]?"
[Turn -2] "¿Qué ocurre después?"
[Turn -1] "¿[Entidad Derivada]?"
[Turn  0] "¿[Propiedad]?"  ← REWRITE THIS QUERY ONLY
Output:
QUERY::[Propiedad] de [Entidad Derivada] en [Entidad Root]

<!-- Pattern 11: Self-Contained Turn -1 → No Hierarchy (Spanish) -->
Input:
[Turn -3] "¿Qué es [Entidad Root]?"
[Turn -2] "¿Cuándo ocurrió?"
[Turn -1] "¿Qué es [Entidad Específica]?"
[Turn  0] "¿[Propiedad]?"  ← REWRITE THIS QUERY ONLY
Output:
QUERY::[Propiedad] de [Entidad Específica]

<!-- Pattern 12: Property Noun Turn -1 → No Chaining (Spanish) -->
Input:
[Turn -3] "¿Qué es [Entidad]?"
[Turn -2] "¿Cómo se detecta?"
[Turn -1] "¿[Propiedad A]?"
[Turn  0] "¿[Propiedad B]?"  ← REWRITE THIS QUERY ONLY
Output:
QUERY::[Propiedad B] de [Entidad]

<!-- Pattern 13: Turn -1 Named Entity → Primary Anchor (Mixed Language) -->
Input:
[Turn -3] "What is [Root Entity]?"
[Turn -2] "How is it detected?"
[Turn -1] "What is [Specific Entity]?"
[Turn  0] "¿y [Propiedad]?"  ← REWRITE THIS QUERY ONLY
Output:
QUERY::[Propiedad] de [Specific Entity]

<!-- Pattern 14: Turn -2 Context Preservation (Spanish) -->
Input:
[Turn -3] "¿Qué es [Entidad]?"
[Turn -2] "¿Aplicación en [Contexto]?"
[Turn -1] "¿[Entidad Relacionada]?"
[Turn  0] "¿[Propiedad]?"  ← REWRITE THIS QUERY ONLY
Output:
QUERY::[Propiedad] de [Entidad Relacionada] en [Contexto]

<!-- Pattern 15: Comparison + Root Context (Mixed Language) -->
Input:
[Turn -3] "What is [Root Topic]?"
[Turn -2] "Detection methods?"
[Turn -1] "[Entity A] vs [Entity B]?"
[Turn  0] "¿[Propiedad]?"  ← REWRITE THIS QUERY ONLY
Output:
QUERY::[Propiedad] de [Entity A] y [Entity B] para [Root Topic]

</examples>

<instruction>
Apply LANGUAGE LOCK first.
Then apply anchor protocol.
No semantic inference.
Return only formatted line.
</instruction>
"""


class AnthropicRecontextualizerConfig:
    """Configuration for Anthropic models in query recontextualization."""

    @staticmethod
    def get_system_prompt():
        """Return the system prompt for Anthropic recontextualizer."""
        return ANTHROPIC_SYSTEM_PROMPT

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
                # Strip markdown code blocks if present (```json ... ```)
                text_stripped = text.strip()
                if text_stripped.startswith('```'):
                    # Find the first newline after opening ```
                    start_idx = text_stripped.find('\n')
                    # Find the closing ```
                    end_idx = text_stripped.rfind('```')
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        text_stripped = text_stripped[start_idx + 1:end_idx].strip()

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
