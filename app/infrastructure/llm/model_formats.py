import json
from abc import ABC, abstractmethod
from typing import Dict, Any


class ModelFormatStrategy(ABC):
    """Strategy interface for different LLM model request formats."""
    
    @abstractmethod
    def format_request(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        """Format the request body for the specific model."""
        pass
    
    @abstractmethod
    def extract_response(self, response_body: Dict[str, Any]) -> str:
        """Extract the generated text from the response."""
        pass
    
    @abstractmethod
    def extract_stream_chunk(self, chunk_data: Dict[str, Any]) -> str:
        """Extract text from streaming response chunk."""
        pass


class LlamaFormat(ModelFormatStrategy):
    """Format for Meta Llama models (Llama3, etc.)"""
    
    def format_request(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        return json.dumps({
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": ["(1)", "La respuesta correcta es:", "\n\nQ:"]
        })
    
    def extract_response(self, response_body: Dict[str, Any]) -> str:
        return response_body.get("generation", "").strip()
    
    def extract_stream_chunk(self, chunk_data: Dict[str, Any]) -> str:
        return chunk_data.get("generation", "")


class ClaudeFormat(ModelFormatStrategy):
    """Format for Anthropic Claude models"""
    
    def format_request(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        return json.dumps({
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_sequences": ["\n\nHuman:"]
        })
    
    def extract_response(self, response_body: Dict[str, Any]) -> str:
        return response_body.get("completion", "").strip()
    
    def extract_stream_chunk(self, chunk_data: Dict[str, Any]) -> str:
        return chunk_data.get("completion", "")


class ModelFormatFactory:
    """Factory to get the appropriate format strategy for a model."""
    
    @staticmethod
    def get_format_strategy(model_id: str) -> ModelFormatStrategy:
        """Return the appropriate format strategy based on model ID."""
        model_id_lower = model_id.lower()
        
        if "llama" in model_id_lower or "meta." in model_id:
            return LlamaFormat()
        elif "claude" in model_id_lower or "anthropic." in model_id:
            return ClaudeFormat()
        else:
            # Default to Claude format for unknown models
            return ClaudeFormat()