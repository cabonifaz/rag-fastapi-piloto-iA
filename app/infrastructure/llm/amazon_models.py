"""
Amazon Nova model configurations for AWS Bedrock LLM provider.
"""

import json
from typing import Dict, Any


class AmazonNovaModelConfig:
    """Configuration for Amazon Nova models on AWS Bedrock."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def get_converse_additional_fields(self) -> Dict[str, Any]:
        """
        Get additionalModelRequestFields for AWS Bedrock Converse API.

        Returns Amazon Nova-specific parameters if needed.
        Currently returns empty dict as Nova uses standard parameters.
        """
        return {}

    def extract_response(self, response_body: Dict[str, Any]) -> str:
        """Extract text from Amazon Nova response."""
        # Nova uses standard Bedrock response format
        content = response_body.get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", "").strip()
        return ""

    def extract_stream_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """Extract text from Amazon Nova streaming chunk."""
        # Nova streaming format uses standard Bedrock delta
        if "delta" in chunk_data:
            return chunk_data["delta"].get("text", "")
        elif "text" in chunk_data:
            return chunk_data["text"]
        return ""

    def build_rag_prompt(self, message: str, context_text: str) -> str:
        """Build RAG prompt optimized for Amazon Nova models."""
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


class NovaPremierConfig(AmazonNovaModelConfig):
    """Specific configuration for Amazon Nova Premier."""

    def __init__(self):
        super().__init__("amazon.nova-premier-v1:0")


class NovaProConfig(AmazonNovaModelConfig):
    """Specific configuration for Amazon Nova Pro."""

    def __init__(self):
        super().__init__("amazon.nova-pro-v1:0")


class NovaLiteConfig(AmazonNovaModelConfig):
    """Specific configuration for Amazon Nova Lite."""

    def __init__(self):
        super().__init__("amazon.nova-lite-v1:0")


class NovaMicroConfig(AmazonNovaModelConfig):
    """Specific configuration for Amazon Nova Micro."""

    def __init__(self):
        super().__init__("amazon.nova-micro-v1:0")


def get_amazon_config(model_id: str) -> AmazonNovaModelConfig:
    """Factory function to get the appropriate Amazon Nova model configuration."""
    model_id_lower = model_id.lower()

    if "nova-premier" in model_id_lower:
        return NovaPremierConfig()
    elif "nova-pro" in model_id_lower:
        return NovaProConfig()
    elif "nova-lite" in model_id_lower:
        return NovaLiteConfig()
    elif "nova-micro" in model_id_lower:
        return NovaMicroConfig()
    else:
        # Default to generic Amazon Nova config for unknown models
        return AmazonNovaModelConfig(model_id)
