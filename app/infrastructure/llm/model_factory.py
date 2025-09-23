"""
Factory for creating model-specific configurations for AWS Bedrock LLM provider.
"""

from typing import Union
from app.infrastructure.llm.anthropic_models import AnthropicModelConfig, get_anthropic_config
from app.infrastructure.llm.meta_models import MetaModelConfig, get_meta_config


class ModelConfigFactory:
    """Factory to get the appropriate model configuration for a given model ID."""

    @staticmethod
    def get_model_config(model_id: str) -> Union[AnthropicModelConfig, MetaModelConfig]:
        """
        Return the appropriate model configuration based on model ID.

        Args:
            model_id: The AWS Bedrock model identifier

        Returns:
            Model configuration object with format methods
        """
        model_id_lower = model_id.lower()

        # Check for Anthropic Claude models
        if "claude" in model_id_lower or "anthropic." in model_id:
            return get_anthropic_config(model_id)

        # Check for Meta Llama models
        elif "llama" in model_id_lower or "meta." in model_id:
            return get_meta_config(model_id)

        else:
            # Default to Claude format for unknown models
            return get_anthropic_config(model_id)

    @staticmethod
    def is_anthropic_model(model_id: str) -> bool:
        """Check if the model is an Anthropic Claude model."""
        model_id_lower = model_id.lower()
        return "claude" in model_id_lower or "anthropic." in model_id

    @staticmethod
    def is_meta_model(model_id: str) -> bool:
        """Check if the model is a Meta Llama model."""
        model_id_lower = model_id.lower()
        return "llama" in model_id_lower or "meta." in model_id

    @staticmethod
    def get_model_provider(model_id: str) -> str:
        """Get the provider name for a given model ID."""
        if ModelConfigFactory.is_anthropic_model(model_id):
            return "anthropic"
        elif ModelConfigFactory.is_meta_model(model_id):
            return "meta"
        else:
            return "unknown"