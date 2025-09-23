"""
Meta (Llama) model configurations for AWS Bedrock LLM provider.
"""

import json
from typing import Dict, Any


class MetaModelConfig:
    """Configuration for Meta Llama models on AWS Bedrock."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def format_request(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        """Format request body for Llama models."""
        return json.dumps({
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": [
                "(1)",
                "La respuesta correcta es:",
                "\n\nQ:",
                "La respuesta final",
                "(respuesta directa)",
                "(no repetir texto)",
                "(no repetir respuesta)"
            ]
        })

    def extract_response(self, response_body: Dict[str, Any]) -> str:
        """Extract text from Llama response."""
        return response_body.get("generation", "").strip()

    def extract_stream_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """Extract text from Llama streaming chunk."""
        # For Llama streaming, the response format can vary
        # Check multiple possible fields in the chunk
        if "generation" in chunk_data:
            return chunk_data["generation"]
        elif "outputs" in chunk_data:
            # Sometimes streaming responses use "outputs" array
            outputs = chunk_data["outputs"]
            if outputs and len(outputs) > 0:
                return outputs[0].get("text", "")
        elif "text" in chunk_data:
            return chunk_data["text"]
        return ""


class Llama3_8BConfig(MetaModelConfig):
    """Specific configuration for Llama 3 8B - focused on request/response formatting."""

    def __init__(self):
        super().__init__("meta.llama3-8b-instruct-v1:0")


class Llama3_70BConfig(MetaModelConfig):
    """Specific configuration for Llama 3 70B - focused on request/response formatting."""

    def __init__(self):
        super().__init__("meta.llama3-70b-instruct-v1:0")


class Llama31_8BConfig(MetaModelConfig):
    """Specific configuration for Llama 3.1 8B - focused on request/response formatting."""

    def __init__(self):
        super().__init__("meta.llama3-1-8b-instruct-v1:0")


class Llama31_70BConfig(MetaModelConfig):
    """Specific configuration for Llama 3.1 70B - focused on request/response formatting."""

    def __init__(self):
        super().__init__("meta.llama3-1-70b-instruct-v1:0")


class Llama31_405BConfig(MetaModelConfig):
    """Specific configuration for Llama 3.1 405B - focused on request/response formatting."""

    def __init__(self):
        super().__init__("meta.llama3-1-405b-instruct-v1:0")


def get_meta_config(model_id: str) -> MetaModelConfig:
    """Factory function to get the appropriate Meta model configuration."""
    model_id_lower = model_id.lower()

    if "llama3-1-8b" in model_id_lower:
        return Llama31_8BConfig()
    elif "llama3-1-70b" in model_id_lower:
        return Llama31_70BConfig()
    elif "llama3-1-405b" in model_id_lower:
        return Llama31_405BConfig()
    elif "llama3-8b" in model_id_lower:
        return Llama3_8BConfig()
    elif "llama3-70b" in model_id_lower:
        return Llama3_70BConfig()
    else:
        # Default to generic Meta config for unknown models
        return MetaModelConfig(model_id)