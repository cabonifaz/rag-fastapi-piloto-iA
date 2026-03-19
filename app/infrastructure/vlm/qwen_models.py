"""
Alibaba Qwen model configurations for AWS Bedrock VLM provider.
"""

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
        content = response_body.get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", "").strip()
        return ""

    def extract_stream_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """Extract text from Qwen streaming chunk."""
        if "delta" in chunk_data:
            return chunk_data["delta"].get("text", "")
        elif "text" in chunk_data:
            return chunk_data["text"]
        return ""


class Qwen3VLConfig(QwenModelConfig):
    """Specific configuration for Qwen 3 VL 235B (vision-language)."""

    def __init__(self):
        super().__init__("qwen.qwen3-vl-235b-a22b")

    def get_converse_additional_fields(self) -> Dict[str, Any]:
        return {
            "enable_thinking": False,
            "stop": [
                "[ILEGIBLE][ILEGIBLE][ILEGIBLE]",
                "0000000000000000000000000",
            ],
        }


def get_qwen_config(model_id: str) -> QwenModelConfig:
    """Factory function to get the appropriate Qwen model configuration."""
    model_id_lower = model_id.lower()

    if "qwen3-vl-235b" in model_id_lower:
        return Qwen3VLConfig()

    else:
        raise ValueError(f"Unsupported VLM model: {model_id}. Supported models: qwen.qwen3-vl-235b-a22b")
