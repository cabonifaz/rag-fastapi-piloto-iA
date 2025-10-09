"""
OpenAI model configurations for AWS Bedrock LLM provider.
"""

import json
from typing import Dict, Any


class OpenAIModelConfig:
    """Configuration for OpenAI models on AWS Bedrock."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        # OpenAI-specific configuration to reduce token consumption
        self.reasoning_effort = "medium"           # reduce internal reasoning tokens

    def get_converse_additional_fields(self) -> Dict[str, Any]:
        """
        Get additionalModelRequestFields for AWS Bedrock Converse API.

        Returns OpenAI-specific parameters:
        - reasoning_effort: "medium" to reduce internal token consumption
        """
        return {
            "reasoning_effort": self.reasoning_effort
        }

    def extract_response(self, response_body: Dict[str, Any]) -> str:
        """Extract text from OpenAI response."""
        # OpenAI format on Bedrock uses 'choices' array
        choices = response_body.get("choices", [])
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            return message.get("content", "").strip()
        return ""

    def extract_stream_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """Extract text from OpenAI streaming chunk."""
        # OpenAI streaming format uses 'choices' with 'delta'
        choices = chunk_data.get("choices", [])
        if choices and len(choices) > 0:
            delta = choices[0].get("delta", {})
            return delta.get("content", "")

        # Fallback for alternative streaming formats
        if "delta" in chunk_data:
            return chunk_data["delta"].get("content", "")
        elif "text" in chunk_data:
            return chunk_data["text"]

        return ""

    def build_rag_prompt(self, message: str, context_text: str) -> str:
        """Build RAG prompt optimized for OpenAI models."""
        return f"""Answer the user's question based **only** on the following context.
Do not search online or make assumptions beyond it.

# Output rules
- Answer **directly and briefly**, focusing only on the question.
- If the context includes document excerpts, cite the document title and page numbers concisely.
- **Do not repeat content** or restate the reasoning.

Question:
{message}

Context:
{context_text}

Return only the final answer that directly addresses the question."""


class GPTOss20BConfig(OpenAIModelConfig):
    """Specific configuration for GPT OSS 20B."""

    def __init__(self):
        super().__init__("openai.gpt-oss-20b-1:0")


class GPTOss120BConfig(OpenAIModelConfig):
    """Specific configuration for GPT OSS 120B."""

    def __init__(self):
        super().__init__("openai.gpt-oss-120b-1:0")


def get_openai_config(model_id: str) -> OpenAIModelConfig:
    """Factory function to get the appropriate OpenAI model configuration."""
    model_id_lower = model_id.lower()

    if "gpt-oss-120b" in model_id_lower:
        return GPTOss120BConfig()
    elif "gpt-oss-20b" in model_id_lower:
        return GPTOss20BConfig()
    else:
        # Default to generic OpenAI config for unknown models
        return OpenAIModelConfig(model_id)
