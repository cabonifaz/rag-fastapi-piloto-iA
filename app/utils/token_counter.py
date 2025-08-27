import logging
import tiktoken
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model_id": self.model_id
        }

@dataclass
class CostCalculation:
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "USD"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "currency": self.currency
        }

class TokenCounter:
    """Utility for counting tokens and calculating costs for different AWS Bedrock models"""
    
    # AWS Bedrock pricing per 1M tokens (as of 2024)
    MODEL_COSTS = {
        # Claude 3 Haiku
        "us.anthropic.claude-3-haiku-20240307-v1:0": {
            "input": 0.25,   # $0.25 per 1M input tokens
            "output": 1.25   # $1.25 per 1M output tokens
        },
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "input": 0.25,
            "output": 1.25
        },
        # Llama 3.1 8B Instruct
        "us.meta.llama3-1-8b-instruct-v1:0": {
            "input": 0.22,   # $0.22 per 1M input tokens
            "output": 0.22   # $0.22 per 1M output tokens
        },
        "meta.llama3-1-8b-instruct-v1:0": {
            "input": 0.22,
            "output": 0.22
        },
        # Titan Text Embeddings V2
        "amazon.titan-embed-text-v2:0": {
            "input": 0.02,   # $0.02 per 1M tokens (embeddings don't have output)
            "output": 0.0
        }
    }
    
    @staticmethod
    def estimate_tokens(text: str, model_id: str = "") -> int:
        """
        Estimate token count for given text.
        Uses tiktoken for approximation since exact tokenizers aren't available.
        """
        try:
            # Use cl100k_base (GPT-4) as approximation for most models
            # It's reasonably close for Claude and Llama models
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = len(encoding.encode(text))
            return tokens
        except Exception as e:
            logger.warning(f"Failed to count tokens with tiktoken: {e}")
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return max(1, len(text) // 4)
    
    @staticmethod
    def extract_token_usage_from_response(response_data: Dict[str, Any], model_id: str) -> TokenUsage:
        """Extract token usage from AWS Bedrock response if available"""
        usage = response_data.get("usage", {})
        
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        total_tokens = input_tokens + output_tokens
        
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model_id=model_id
        )
    
    @staticmethod
    def calculate_cost(token_usage: TokenUsage) -> CostCalculation:
        """Calculate cost based on token usage and model pricing"""
        model_id = token_usage.model_id
        
        if model_id not in TokenCounter.MODEL_COSTS:
            logger.warning(f"No pricing data for model {model_id}")
            return CostCalculation(0.0, 0.0, 0.0)
        
        pricing = TokenCounter.MODEL_COSTS[model_id]
        
        # Convert to cost (pricing is per 1M tokens)
        input_cost = (token_usage.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (token_usage.output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return CostCalculation(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )
    
    @staticmethod
    def log_usage_and_cost(token_usage: TokenUsage, cost_calc: CostCalculation, operation: str = ""):
        """Log token usage and cost to console"""
        logger.info("=" * 60)
        logger.info(f"TOKEN USAGE REPORT - {operation}")
        logger.info("=" * 60)
        logger.info(f"Model: {token_usage.model_id}")
        logger.info(f"Input Tokens: {token_usage.input_tokens:,}")
        logger.info(f"Output Tokens: {token_usage.output_tokens:,}")
        logger.info(f"Total Tokens: {token_usage.total_tokens:,}")
        logger.info("-" * 40)
        logger.info(f"Input Cost: ${cost_calc.input_cost:.6f}")
        logger.info(f"Output Cost: ${cost_calc.output_cost:.6f}")
        logger.info(f"Total Cost: ${cost_calc.total_cost:.6f}")
        logger.info("=" * 60)
        
        # Also print to console for immediate visibility
        print("=" * 60)
        print(f"TOKEN USAGE REPORT - {operation}")
        print("=" * 60)
        print(f"Model: {token_usage.model_id}")
        print(f"Input Tokens: {token_usage.input_tokens:,}")
        print(f"Output Tokens: {token_usage.output_tokens:,}")
        print(f"Total Tokens: {token_usage.total_tokens:,}")
        print("-" * 40)
        print(f"Input Cost: ${cost_calc.input_cost:.6f}")
        print(f"Output Cost: ${cost_calc.output_cost:.6f}")
        print(f"Total Cost: ${cost_calc.total_cost:.6f}")
        print("=" * 60)