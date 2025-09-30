"""
Amazon Titan embedding model configuration for AWS Bedrock.
"""

import json
from typing import List


class TitanEmbedConfig:
    """Configuration for Amazon Titan embedding models."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def format_request(self, text: str) -> str:
        """Format request body for Titan embedding models."""
        return json.dumps({
            "inputText": text
        })

    def extract_embedding(self, response_body: dict) -> List[float]:
        """Extract embedding vector from Titan response."""
        return response_body.get("embedding", [])
