"""
Anthropic (Claude) model configurations for AWS Bedrock LLM provider.
"""

import json
from typing import Dict, Any


class AnthropicModelConfig:
    """Configuration for Anthropic Claude models on AWS Bedrock."""

    def __init__(self, model_id: str):
        self.model_id = model_id.lower()
        self.is_claude_3 = "claude-3" in self.model_id or "us.anthropic.claude-3" in self.model_id

    def format_request(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        """Format request body for Claude models."""
        if self.is_claude_3:
            # Claude 3+ uses Messages API format
            return json.dumps({
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "anthropic_version": "bedrock-2023-05-31"
            })
        else:
            # Legacy Claude format
            return json.dumps({
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop_sequences": ["\n\nHuman:"]
            })

    def extract_response(self, response_body: Dict[str, Any]) -> str:
        """Extract text from Claude response."""
        if self.is_claude_3:
            # Claude 3+ response format
            content = response_body.get("content", [])
            if content and len(content) > 0:
                return content[0].get("text", "").strip()
            return ""
        else:
            # Legacy Claude format
            return response_body.get("completion", "").strip()

    def extract_stream_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """Extract text from Claude streaming chunk."""
        if self.is_claude_3:
            # Claude 3+ streaming format
            if "delta" in chunk_data:
                return chunk_data["delta"].get("text", "")
            elif "content_block_delta" in chunk_data:
                delta = chunk_data["content_block_delta"]
                return delta.get("delta", {}).get("text", "")
            return ""
        else:
            # Legacy Claude streaming format
            if "completion" in chunk_data:
                return chunk_data["completion"]
            elif "delta" in chunk_data:
                return chunk_data["delta"].get("text", "")
            elif "text" in chunk_data:
                return chunk_data["text"]
            return ""

    def build_rag_prompt(self, message: str, context_text: str) -> str:
        """Build RAG prompt optimized for Claude models."""
        return f"""Answer the user's question based on the following:

- If the question is about a specific context provided, answer only using that context.
- If the user asks for a summary or review, use the conversation history to generate the summary.
- Do not search online or make assumptions beyond what is provided in the context or conversation history.

# Output rules
- Give a **clear and informative answer**, focused directly on the question.
- Include the **main details or explanations** from the context, but avoid unnecessary length.
- Keep a **balanced tone**: neither too short nor overly elaborate.
- If the context includes document excerpts, **cite titles or page numbers briefly** when relevant.
- Do **not invent** or add information not present in the context.

Question:
{message}

Context:
{context_text}

Return only the final answer that addresses the question clearly and completely."""


class Claude3HaikuConfig(AnthropicModelConfig):
    """Specific configuration for Claude 3 Haiku."""

    def __init__(self):
        super().__init__("us.anthropic.claude-3-haiku-20240307-v1:0")


class Claude35HaikuConfig(AnthropicModelConfig):
    """Specific configuration for Claude 3.5 Haiku."""

    def __init__(self):
        super().__init__("anthropic.claude-3-5-haiku-20241022-v1:0")


class Claude3SonnetConfig(AnthropicModelConfig):
    """Specific configuration for Claude 3 Sonnet."""

    def __init__(self):
        super().__init__("anthropic.claude-3-sonnet-20240229-v1:0")


class Claude35SonnetConfig(AnthropicModelConfig):
    """Specific configuration for Claude 3.5 Sonnet."""

    def __init__(self):
        super().__init__("anthropic.claude-3-5-sonnet-20241022-v2:0")


class Claude37SonnetConfig(AnthropicModelConfig):
    """Specific configuration for Claude 3.7 Sonnet."""

    def __init__(self):
        super().__init__("anthropic.claude-3-7-sonnet-20250219-v1:0")


class Claude4SonnetConfig(AnthropicModelConfig):
    """Specific configuration for Claude 4 Sonnet."""

    def __init__(self):
        super().__init__("anthropic.claude-sonnet-4-20250514-v1:0")


class Claude45SonnetConfig(AnthropicModelConfig):
    """Specific configuration for Claude 4.5 Sonnet."""

    def __init__(self):
        super().__init__("anthropic.claude-sonnet-4-5-20250929-v1:0")


class Claude3OpusConfig(AnthropicModelConfig):
    """Specific configuration for Claude 3 Opus."""

    def __init__(self):
        super().__init__("anthropic.claude-3-opus-20240229-v1:0")


class Claude4OpusConfig(AnthropicModelConfig):
    """Specific configuration for Claude 4 Opus."""

    def __init__(self):
        super().__init__("anthropic.claude-opus-4-20250514-v1:0")


class Claude41OpusConfig(AnthropicModelConfig):
    """Specific configuration for Claude 4.1 Opus."""

    def __init__(self):
        super().__init__("anthropic.claude-opus-4-1-20250805-v1:0")


def get_anthropic_config(model_id: str) -> AnthropicModelConfig:
    """Factory function to get the appropriate Anthropic model configuration."""
    model_id_lower = model_id.lower()

    # Claude 4.5 models
    if "claude-sonnet-4-5" in model_id_lower or "sonnet-4-5" in model_id_lower:
        return Claude45SonnetConfig()

    # Claude 4.1 models
    elif "claude-opus-4-1" in model_id_lower or "opus-4-1" in model_id_lower:
        return Claude41OpusConfig()

    # Claude 4 models
    elif "claude-opus-4" in model_id_lower or "opus-4-20250514" in model_id_lower:
        return Claude4OpusConfig()
    elif "claude-sonnet-4" in model_id_lower or "sonnet-4-20250514" in model_id_lower:
        return Claude4SonnetConfig()

    # Claude 3.7 models
    elif "claude-3-7-sonnet" in model_id_lower:
        return Claude37SonnetConfig()

    # Claude 3.5 models
    elif "claude-3-5-haiku" in model_id_lower:
        return Claude35HaikuConfig()
    elif "claude-3-5-sonnet" in model_id_lower:
        return Claude35SonnetConfig()

    # Claude 3 models
    elif "claude-3-opus" in model_id_lower:
        return Claude3OpusConfig()
    elif "claude-3-haiku" in model_id_lower:
        return Claude3HaikuConfig()
    elif "claude-3-sonnet" in model_id_lower:
        return Claude3SonnetConfig()

    else:
        # Default to generic Claude config for unknown models
        return AnthropicModelConfig(model_id)