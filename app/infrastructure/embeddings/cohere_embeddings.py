"""
Cohere embedding model configuration for AWS Bedrock.
"""

import json
from typing import List


class CohereEmbedConfig:
    """Configuration for Cohere embedding models."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.input_type = "search_query"  # or "search_document" for indexing

    def format_request(self, text: str) -> str:
        """Format request body for Cohere embedding models."""
        return json.dumps({
            "texts": [text],
            "input_type": self.input_type
        })

    def extract_embedding(self, response_body: dict) -> List[float]:
        """Extract embedding vector from Cohere response."""
        embeddings = response_body.get("embeddings", [])
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        return []
