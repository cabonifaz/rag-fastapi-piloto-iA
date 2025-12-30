from typing import Union
from app.infrastructure.query_rewriter.amazon_models import AmazonRecontextualizerConfig
from app.infrastructure.query_rewriter.anthropic_models import AnthropicRecontextualizerConfig
from app.infrastructure.query_rewriter.openai_models import OpenAIRecontextualizerConfig
from app.infrastructure.query_rewriter.meta_models import MetaRecontextualizerConfig

class ModelFactory:
    """Factory to get the appropriate model-specific configuration for query recontextualization."""

    @staticmethod
    def get_model_config(model_id: str) -> Union[AmazonRecontextualizerConfig, AnthropicRecontextualizerConfig, OpenAIRecontextualizerConfig, MetaRecontextualizerConfig]:
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
        # Check for Meta Llama models
        elif "llama" in model_id_lower or "meta." in model_id:
            return MetaRecontextualizerConfig
        # Check for OpenAI models
        elif "openai" in model_id_lower or "gpt" in model_id_lower:
            return OpenAIRecontextualizerConfig
        # Check for Meta models (Llama)
        elif "meta" in model_id_lower or "llama" in model_id_lower:
            return MetaRecontextualizerConfig
        # Check for Amazon Nova models
        elif "nova" in model_id_lower or "amazon." in model_id:
            return AmazonRecontextualizerConfig
        else:
            # Default to Amazon Nova as a fallback
            return AmazonRecontextualizerConfig
