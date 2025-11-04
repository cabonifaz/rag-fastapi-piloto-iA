"""
Alibaba Qwen model configurations for AWS Bedrock LLM provider.
"""

import json
from typing import Dict, Any


class QwenModelConfig:
    """Configuration for Alibaba Qwen models on AWS Bedrock."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def get_converse_additional_fields(self) -> Dict[str, Any]:
        """
        Get additionalModelRequestFields for AWS Bedrock Converse API.

        Returns Qwen-specific parameters if needed.
        Currently returns empty dict as Qwen uses standard parameters.
        """
        return {}

    def extract_response(self, response_body: Dict[str, Any]) -> str:
        """Extract text from Qwen response."""
        # Qwen uses standard Bedrock response format
        content = response_body.get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", "").strip()
        return ""

    def extract_stream_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """Extract text from Qwen streaming chunk."""
        # Qwen streaming format uses standard Bedrock delta
        if "delta" in chunk_data:
            return chunk_data["delta"].get("text", "")
        elif "text" in chunk_data:
            return chunk_data["text"]
        return ""

    def build_rag_prompt(self, message: str, context_text: str) -> str:
        """Build RAG prompt optimized for Qwen models."""
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


class QwenPlusConfig(QwenModelConfig):
    """Specific configuration for Qwen Plus."""

    def __init__(self):
        super().__init__("alibaba.qwen1-5-plus-v1:0")


class QwenTurboConfig(QwenModelConfig):
    """Specific configuration for Qwen Turbo."""

    def __init__(self):
        super().__init__("alibaba.qwen1-5-turbo-v1:0")


class QwenMaxConfig(QwenModelConfig):
    """Specific configuration for Qwen Max."""

    def __init__(self):
        super().__init__("alibaba.qwen2-72b-instruct-v1:0")


class QwenLongConfig(QwenModelConfig):
    """Specific configuration for Qwen Long Context."""

    def __init__(self):
        super().__init__("alibaba.qwen2-1b-instruct-v1:0")


def get_qwen_config(model_id: str) -> QwenModelConfig:
    """Factory function to get the appropriate Qwen model configuration."""
    model_id_lower = model_id.lower()

    # Qwen 2 models
    if "qwen2-72b" in model_id_lower or "qwen-max" in model_id_lower:
        return QwenMaxConfig()
    elif "qwen2-1b" in model_id_lower or "qwen-long" in model_id_lower:
        return QwenLongConfig()

    # Qwen 1.5 models
    elif "qwen1-5-plus" in model_id_lower or "qwen-plus" in model_id_lower:
        return QwenPlusConfig()
    elif "qwen1-5-turbo" in model_id_lower or "qwen-turbo" in model_id_lower:
        return QwenTurboConfig()

    else:
        # Default to generic Qwen config for unknown models
        return QwenModelConfig(model_id)