"""
Factory for creating orchestrator model-specific configurations.
"""

from typing import Union
from app.infrastructure.task_decomposition.nova_models import NovaConfig
from app.infrastructure.task_decomposition.openai_models import OpenAIConfig


class OrchestratorConfigFactory:
    """Factory to get the appropriate orchestrator configuration for a given model ID."""

    @staticmethod
    def get_config(model_id: str) -> Union[NovaConfig, OpenAIConfig]:
        """
        Return the appropriate orchestrator configuration based on model ID.

        Args:
            model_id: The AWS Bedrock model identifier

        Returns:
            Orchestrator configuration class with build methods
        """
        model_id_lower = model_id.lower()

        # Check for OpenAI GPT-OSS models
        if "openai.gpt-oss" in model_id_lower:
            return OpenAIConfig

        # Check for Amazon Nova models
        elif "nova" in model_id_lower or "amazon.nova" in model_id_lower:
            return NovaConfig

        else:
            # Default to Nova format for unknown models
            return NovaConfig

    @staticmethod
    def is_openai_model(model_id: str) -> bool:
        """Check if the model is an OpenAI GPT-OSS model."""
        model_id_lower = model_id.lower()
        return "openai.gpt-oss" in model_id_lower

    @staticmethod
    def is_nova_model(model_id: str) -> bool:
        """Check if the model is an Amazon Nova model."""
        model_id_lower = model_id.lower()
        return "nova" in model_id_lower or "amazon.nova" in model_id_lower

    @staticmethod
    def get_model_provider(model_id: str) -> str:
        """Get the provider name for a given model ID."""
        if OrchestratorConfigFactory.is_openai_model(model_id):
            return "openai"
        elif OrchestratorConfigFactory.is_nova_model(model_id):
            return "amazon-nova"
        else:
            return "unknown"
