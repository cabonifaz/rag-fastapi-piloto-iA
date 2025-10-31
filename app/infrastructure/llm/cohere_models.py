"""
Cohere model configurations for AWS Bedrock LLM provider.
"""

import json
from typing import Dict, Any


class CohereModelConfig:
    """Configuration for Cohere models on AWS Bedrock."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def get_converse_additional_fields(self) -> Dict[str, Any]:
        """
        Get additionalModelRequestFields for AWS Bedrock Converse API.

        Returns Cohere-specific parameters if needed.
        Currently returns empty dict as Cohere uses standard parameters.
        """
        return {}

    def extract_response(self, response_body: Dict[str, Any]) -> str:
        """Extract text from Cohere response."""
        # Cohere uses standard Bedrock response format
        content = response_body.get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", "").strip()
        return ""

    def extract_stream_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """Extract text from Cohere streaming chunk."""
        # Cohere streaming format uses standard Bedrock delta
        if "delta" in chunk_data:
            return chunk_data["delta"].get("text", "")
        elif "text" in chunk_data:
            return chunk_data["text"]
        return ""

    def build_rag_prompt(self, message: str, context_text: str) -> str:
        """Build RAG prompt optimized for Cohere models."""
        return f"""Answer the user's question based on the following:

- If the question is about a specific context provided, answer only using that context.
- If the user asks for a summary or review, use the conversation history to generate the summary.
- Do not search online or make assumptions beyond what is provided in the context or conversation history.

# Output rules
- Give a **clear and informative answer**, focused directly on the question.
- Include the **main details or explanations** from the context, but avoid unnecessary length.
- Keep a **balanced tone**: neither too short nor overly elaborate.
- If the context includes document excerpts, **cite titles or page numbers briefly** when relevant.
- Do **not invent** or add information not present in the context.

Question:
{message}

Context:
{context_text}

Return only the final answer that addresses the question clearly and completely."""


class CohereCommandR7BConfig(CohereModelConfig):
    """Specific configuration for Cohere Command R 7B."""

    def __init__(self):
        super().__init__("cohere.command-r-7b-12-2024-v1:0")


class CohereCommandR35LConfig(CohereModelConfig):
    """Specific configuration for Cohere Command R 35B."""

    def __init__(self):
        super().__init__("cohere.command-r-35-12-2024-v1:0")


class CohereCommandRPlus4BConfig(CohereModelConfig):
    """Specific configuration for Cohere Command R+ 4B."""

    def __init__(self):
        super().__init__("cohere.command-r-plus-4b-v1:0")


class CohereCommandRPlusConfig(CohereModelConfig):
    """Specific configuration for Cohere Command R+."""

    def __init__(self):
        super().__init__("cohere.command-r-plus-v1:0")


class CohereCommandLightConfig(CohereModelConfig):
    """Specific configuration for Cohere Command Light."""

    def __init__(self):
        super().__init__("cohere.command-light-text-v14:7:4k")


class CohereCommandConfig(CohereModelConfig):
    """Specific configuration for Cohere Command."""

    def __init__(self):
        super().__init__("cohere.command-text-v14:7:4k")


def get_cohere_config(model_id: str) -> CohereModelConfig:
    """Factory function to get the appropriate Cohere model configuration."""
    model_id_lower = model_id.lower()

    # Cohere Command R models
    if "command-r-7b" in model_id_lower:
        return CohereCommandR7BConfig()
    elif "command-r-35" in model_id_lower or "command-r-35b" in model_id_lower:
        return CohereCommandR35LConfig()

    # Cohere Command R+ models
    elif "command-r-plus-4b" in model_id_lower:
        return CohereCommandRPlus4BConfig()
    elif "command-r-plus" in model_id_lower or "command-r+" in model_id_lower:
        return CohereCommandRPlusConfig()

    # Cohere Command models
    elif "command-light" in model_id_lower:
        return CohereCommandLightConfig()
    elif "command-text" in model_id_lower or "command" in model_id_lower:
        return CohereCommandConfig()

    else:
        # Default to generic Cohere config for unknown models
        return CohereModelConfig(model_id)
