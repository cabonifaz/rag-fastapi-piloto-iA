"""
OpenAI model configurations for AWS Bedrock LLM provider.
"""

import json
from typing import Dict, Any


class OpenAIModelConfig:
    """Configuration for OpenAI models on AWS Bedrock."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def format_request(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        """Format request body for OpenAI models."""
        return json.dumps({
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        })

    def extract_response(self, response_body: Dict[str, Any]) -> str:
        """Extract text from OpenAI response."""
        # OpenAI format on Bedrock uses 'choices' array
        choices = response_body.get("choices", [])
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            return message.get("content", "").strip()
        return ""

    def extract_stream_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """Extract text from OpenAI streaming chunk."""
        # OpenAI streaming format uses 'choices' with 'delta'
        choices = chunk_data.get("choices", [])
        if choices and len(choices) > 0:
            delta = choices[0].get("delta", {})
            return delta.get("content", "")

        # Fallback for alternative streaming formats
        if "delta" in chunk_data:
            return chunk_data["delta"].get("content", "")
        elif "text" in chunk_data:
            return chunk_data["text"]

        return ""

    def build_rag_prompt(self, message: str, context_text: str) -> str:
        """Build RAG prompt optimized for OpenAI models."""
        return f"""Answer the user's question based on the following context. Do not search on the internet. Answer in the same language as the question. Present the answer on markdown format.

If the data contains JSON with "table", "headers", and "rows" keys, interpret and present it as a formatted Markdown table. Include all source references used in the answer with document title and page numbers. 

If the data comes from an API call, the context will include an "API Call". In this case, interpret the response JSON (usually an array of objects) as a table. Present that data in Markdown format, but do not include document or page references for API data.

Question: {message}

Context:
{context_text}

Provide a clear, accurate response based solely on the provided context. Only answer what the user asked."""


class GPTOss20BConfig(OpenAIModelConfig):
    """Specific configuration for GPT OSS 20B."""

    def __init__(self):
        super().__init__("openai.gpt-oss-20b-1:0")


class GPTOss120BConfig(OpenAIModelConfig):
    """Specific configuration for GPT OSS 120B."""

    def __init__(self):
        super().__init__("openai.gpt-oss-120b-1:0")


def get_openai_config(model_id: str) -> OpenAIModelConfig:
    """Factory function to get the appropriate OpenAI model configuration."""
    model_id_lower = model_id.lower()

    if "gpt-oss-120b" in model_id_lower:
        return GPTOss120BConfig()
    elif "gpt-oss-20b" in model_id_lower:
        return GPTOss20BConfig()
    else:
        # Default to generic OpenAI config for unknown models
        return OpenAIModelConfig(model_id)
