"""
Factory for creating model-specific configurations for AWS Bedrock LLM provider.
"""

from typing import Union
from app.infrastructure.llm.anthropic_models import AnthropicModelConfig, get_anthropic_config
from app.infrastructure.llm.meta_models import MetaModelConfig, get_meta_config
from app.infrastructure.llm.openai_models import OpenAIModelConfig, get_openai_config
from app.infrastructure.llm.qwen_models import QwenModelConfig, get_qwen_config
from app.infrastructure.llm.amazon_models import AmazonNovaModelConfig, get_amazon_config


class ModelConfigFactory:
    """Factory to get the appropriate model configuration for a given model ID."""

    @staticmethod
    def get_model_config(model_id: str) -> Union[AnthropicModelConfig, MetaModelConfig, OpenAIModelConfig, QwenModelConfig, AmazonNovaModelConfig]:
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

        # Check for OpenAI models
        elif "gpt" in model_id_lower or "openai." in model_id:
            return get_openai_config(model_id)

        # Check for Qwen models (Alibaba)
        elif "qwen" in model_id_lower or "alibaba." in model_id:
            return get_qwen_config(model_id)

        # Check for Amazon Nova models
        elif "nova" in model_id_lower or "amazon." in model_id:
            return get_amazon_config(model_id)

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
    def is_openai_model(model_id: str) -> bool:
        """Check if the model is an OpenAI model."""
        model_id_lower = model_id.lower()
        return "gpt" in model_id_lower or "openai." in model_id

    @staticmethod
    def is_qwen_model(model_id: str) -> bool:
        """Check if the model is a Qwen (Alibaba) model."""
        model_id_lower = model_id.lower()
        return "qwen" in model_id_lower or "alibaba." in model_id

    @staticmethod
    def is_amazon_model(model_id: str) -> bool:
        """Check if the model is an Amazon Nova model."""
        model_id_lower = model_id.lower()
        return "nova" in model_id_lower or "amazon." in model_id

    @staticmethod
    def get_model_provider(model_id: str) -> str:
        """Get the provider name for a given model ID."""
        if ModelConfigFactory.is_anthropic_model(model_id):
            return "anthropic"
        elif ModelConfigFactory.is_meta_model(model_id):
            return "meta"
        elif ModelConfigFactory.is_openai_model(model_id):
            return "openai"
        elif ModelConfigFactory.is_qwen_model(model_id):
            return "qwen"
        elif ModelConfigFactory.is_amazon_model(model_id):
            return "amazon"
        else:
            return "unknown"