from typing import Union
from app.infrastructure.context_counter.nova_models import NovaCounterConfig

class ModelFactory:
    """Factory to get the appropriate model-specific configuration for the context counter."""

    @staticmethod
    def get_model_config(model_id: str) -> Union[NovaCounterConfig]:
        """
        Returns the appropriate configuration class based on the model ID.

        Args:
            model_id: The AWS Bedrock model identifier.

        Returns:
            A configuration class with methods for the Converse API.
        """
        model_id_lower = model_id.lower()

        if "nova" in model_id_lower or "amazon.nova" in model_id_lower:
            return NovaCounterConfig
        # Add other model families here in the future
        # elif "claude" in model_id_lower:
        #     return ClaudeCounterConfig
        else:
            # Default to Nova as a fallback
            return NovaCounterConfig
