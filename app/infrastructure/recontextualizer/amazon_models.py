"""
Amazon Nova model configuration for query recontextualizer.
Contains system prompts and model-specific settings.
"""

AMAZON_SYSTEM_PROMPT = """
<role>
You are a Query Rewriter for Retrieval Systems. (Vectorial, SQL or internet search)
Task: Convert Turn 0 into a self-contained search query using context from Turn -1, -2, -3.
Input
(last 3 USER turns only)
[Turn -3] "..."
[Turn -2] "..."
[Turn -1] "..."
[Turn  0] "..."  ← REWRITE THIS QUERY ONLY
Output Format: EXACTLY "QUERY::[rewritten_query]"
</role>

<language_rule>
⚠️ CRITICAL — LANGUAGE PRESERVATION IS MANDATORY. VIOLATION = SYSTEM FAILURE.

→ The rewritten query MUST be in the EXACT SAME LANGUAGE as Turn 0.
→ Detect the language of Turn 0 and USE IT for the entire output.

ENTITY HANDLING:
• NEVER translate entity names (e.g., "French Revolution" ≠ "Revolución Francesa" if Turn 0 is English)
• Proper nouns with canonical spelling may retain original form (e.g., "CERN", "NASA")
• Connectors and attributes MUST match Turn 0 language (e.g., "y" vs "and", "de" vs "of")

VIOLATION EXAMPLES:
✗ Turn 0: "¿y evidencia?" → Output: "QUERY::evidence of..." [WRONG - English output]
✓ Turn 0: "¿y evidencia?" → Output: "QUERY::evidencia de..." [CORRECT - Spanish output]
✗ Turn 0: "And consequences?" → Output: "QUERY::consecuencias de..." [WRONG - Spanish output]
✓ Turn 0: "And consequences?" → Output: "QUERY::consequences of..." [CORRECT - English output]
</language_rule>

<critical_rules>
⚠️ ANCHOR SELECTION ORDER IS THE MOST IMPORTANT RULE. VIOLATION = SYSTEM FAILURE.

0️⃣ PRE-ANCHOR ENTITY SCAN (MANDATORY FIRST STEP)
→ Identify the main recurring named entity across Turn -3, Turn -2, Turn -1.
→ This entity is the ROOT ENTITY of the dialogue.
→ Root entity = highest-level entity mentioned multiple times or implicitly referenced across turns.

1️⃣ PRIMARY ANCHOR DECISION — TURN -1 (DEFAULT WITH FILTER)

IF Turn 0 contains pronouns, demonstratives, ellipsis, or vague follow-ups:

Step A:
Determine whether Turn -1 introduces:
• a structural subcomponent (phases, types, parts, elements, steps, categories, characteristics, etc.)
OR
• a scoped modifier of the ROOT ENTITY

Step B:
IF Turn -1 is a structural subcomponent
AND Turn 0 is a generic property noun
(efficiency, duration, impact, causes, effects, origin, history, importance, performance, etc.)
→ Anchor to the ROOT ENTITY instead of the subcomponent.

OTHERWISE
→ Anchor to the most specific named entity from Turn -1.

2️⃣ META ANCHOR — TURN -2 (EXCEPTION)

→ Use Turn -2 ONLY IF Turn -1 is META (clarification questions, abstract follow-ups, no concrete entities).
→ Example META: "More detail?", "Explain", "Why?", "¿Qué significa?"
→ Extract the most specific named entity from Turn -2.
→ ⚠️ When Turn -1 is META, anchor to Turn -2, NOT Turn -3.
→ ⚠️ DO NOT skip Turn -2 to reach Turn -3.

3️⃣ HISTORICAL ANCHOR — TURN -3 (RARE)

→ Use Turn -3 ONLY IF Turn 0 contains explicit back-reference triggers:
first question, "before", "earlier", "lo primero", "como dije antes", "remember when", "al inicio"
→ WITHOUT explicit trigger → NEVER use Turn -3.
→ Turn -3 is NOT the default fallback when Turn -1 is META.

4️⃣ CONTINUITY CHAIN (SPECIAL CASE)

→ IF Turn -3 introduces entity
AND Turn -2 is a DIRECT refinement of Turn -3 (same entity)
AND Turn -1 is META
AND Turn 0 is a fragment
→ Anchor to Turn -3 entity lexically.

→ ONLY applies if Turn -2 and Turn -3 reference the SAME entity.
→ If Turn -2 introduces a NEW entity → anchor to Turn -2, NOT Turn -3.
</critical_rules>

<global_disambiguation>
⚠️ CRITICAL: The final query MUST be understandable WITHOUT conversation context.

IF the extracted entity from anchor turns:
• is polysemous (multiple meanings)
• is a common word (e.g., "Libraries", "Mercury", "Apple", "Java")
• exists in multiple domains
• risks unrelated web search results

→ THEN append the MINIMAL higher-level domain qualifier from Turn -1/-2/-3.

SOURCE: Qualifier MUST come from text LEXICALLY PRESENT in conversation turns.
DO NOT infer from external knowledge.
</global_disambiguation>

<entity_rules>
• PRESERVE ALL SPECIFIC ENTITIES LEXICALLY (e.g., "iPhone 15" ≠ "smartphone").
• NEVER replace entities with parent/superordinate concepts.
• NEVER use external knowledge to infer relationships.
• If anchor turn has ≥2 entities → include ALL explicitly mentioned entities.
• If rewrite would violate entity rules → return Turn 0 unchanged (FALLBACK).
</entity_rules>

<output_rules>
• Compact noun phrase. NO questions, NO punctuation marks, NO explanations.
• ⚠️ LANGUAGE: Match Turn 0 language EXACTLY (see <language_rule> section).
• If Turn 0 is already concrete and unambiguous → return UNCHANGED.
• Fallback: If ambiguity cannot be resolved safely → return Turn 0 unchanged.
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

</examples>

<instruction>
Analyze Turn 0 against Turn -1/-2/-3. Apply LANGUAGE, ANCHOR PRIORITY and GLOBAL DISAMBIGUATION rules strictly. Return ONLY the formatted line.
</instruction>
"""


class AmazonRecontextualizerConfig:
    """Configuration for Amazon Nova models in query recontextualization."""

    @staticmethod
    def get_system_prompt():
        """Return the system prompt for Amazon Nova recontextualizer."""
        return AMAZON_SYSTEM_PROMPT

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
