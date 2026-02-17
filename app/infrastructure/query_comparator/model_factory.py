from typing import Union
from app.infrastructure.query_comparator.amazon_models import AmazonComparatorConfig


class ModelFactory:
    """Factory to get the appropriate model-specific configuration for query comparison."""

    @staticmethod
    def get_model_config(model_id: str) -> Union[AmazonComparatorConfig]:
        """
        Returns the appropriate configuration class based on the model ID.

        Args:
            model_id: The AWS Bedrock model identifier.

        Returns:
            A configuration class with methods for the Converse API.
        """
        model_id_lower = model_id.lower()

        if "nova" in model_id_lower or "amazon." in model_id:
            return AmazonComparatorConfig
        else:
            # Default to Amazon Nova as a fallback
            return AmazonComparatorConfig
