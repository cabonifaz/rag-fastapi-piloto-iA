from typing import Union
from app.infrastructure.recontextualizer.nova_models import NovaRecontextualizerConfig
from app.infrastructure.recontextualizer.anthropic_models import AnthropicRecontextualizerConfig
from app.infrastructure.recontextualizer.openai_models import OpenAIRecontextualizerConfig

class ModelFactory:
    """Factory to get the appropriate model-specific configuration for query recontextualization."""

    @staticmethod
    def get_model_config(model_id: str) -> Union[NovaRecontextualizerConfig, AnthropicRecontextualizerConfig, OpenAIRecontextualizerConfig]:
        """
        Returns the appropriate configuration class based on the model ID.

        Args:
            model_id: The AWS Bedrock model identifier.

        Returns:
            A configuration class with methods for the Converse API.
        """
        model_id_lower = model_id.lower()

        # Check for Anthropic models (Claude)
        if "anthropic" in model_id_lower or "claude" in model_id_lower:
            return AnthropicRecontextualizerConfig
        # Check for OpenAI models
        elif "openai" in model_id_lower or "gpt" in model_id_lower:
            return OpenAIRecontextualizerConfig
        # Check for Amazon Nova models
        elif "nova" in model_id_lower or "amazon.nova" in model_id_lower:
            return NovaRecontextualizerConfig
        else:
            # Default to Nova as a fallback
            return NovaRecontextualizerConfig
