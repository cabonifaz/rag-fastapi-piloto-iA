"""
Factory for creating model-specific configurations for AWS Bedrock VLM provider.
Qwen models are validated to ensure only VL (vision-language) variants are used.
"""

from typing import Union
from app.infrastructure.vlm.qwen_models import QwenModelConfig, get_qwen_config


class ModelConfigFactory:
    """Factory to get the appropriate model configuration for a given model ID."""

    @staticmethod
    def get_model_config(model_id: str) -> Union[QwenModelConfig]:
        model_id_lower = model_id.lower()

        if "qwen" in model_id_lower or "alibaba." in model_id:
            return get_qwen_config(model_id)

        else:
            return get_qwen_config(model_id)

    @staticmethod
    def is_qwen_model(model_id: str) -> bool:
        """Check if the model is a Qwen (Alibaba) model."""
        model_id_lower = model_id.lower()
        return "qwen" in model_id_lower or "alibaba." in model_id

    @staticmethod
    def get_model_provider(model_id: str) -> str:
        if ModelConfigFactory.is_qwen_model(model_id):
            return "qwen"
        else:
            return "unknown"