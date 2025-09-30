import boto3
import json
import logging
import os
from typing import Optional, AsyncGenerator
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from app.domain.ports.llm_port import LLMPort
from app.infrastructure.llm.model_factory import ModelConfigFactory
# from app.utils.token_counter import TokenCounter, TokenUsage

# Configure logging
logger = logging.getLogger(__name__)


class AWSBedrockLLMProvider(LLMPort):
    """
    Generic AWS Bedrock LLM provider that supports multiple model families.
    Uses model-specific configurations for proper request/response formatting.
    """

    def __init__(
        self,
        region: str,
        model_id: str,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        """
        Initialize AWS Bedrock client with model-specific configuration.

        Args:
            region: AWS region (e.g., "us-east-1")
            model_id: Bedrock model ID (e.g., "anthropic.claude-v2")
            profile_name: AWS profile name (optional, for local development)
            aws_access_key_id: AWS access key ID (optional)
            aws_secret_access_key: AWS secret access key (optional)
        """
        session_params = {"region_name": region}

        # Use profile only in development, not in production with IAM roles
        if profile_name and os.getenv('ENVIRONMENT', '').lower() != 'production':
            session_params["profile_name"] = profile_name
        # If no profile, use direct credentials if available
        elif aws_access_key_id and aws_secret_access_key:
            session_params["aws_access_key_id"] = aws_access_key_id
            session_params["aws_secret_access_key"] = aws_secret_access_key

        session = boto3.Session(**session_params)
        self.client = session.client("bedrock-runtime")
        self.model_id = model_id

        # Get model-specific configuration
        self.model_config = ModelConfigFactory.get_model_config(model_id)

        logger.info(f"AWS Bedrock LLM provider initialized with model: {model_id} "
                   f"(Provider: {ModelConfigFactory.get_model_provider(model_id)})")

    def get_model_config(self):
        """Get the model configuration object for accessing model-specific methods."""
        return self.model_config

    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Genera texto usando un modelo de AWS Bedrock.
        Soporta tanto modelos Claude como Llama3.
        Includes token counting and cost calculation.
        """
        try:
            from app.core.config import settings
            
            # Costs calculation (deactivated)
            # Count input tokens
            # input_tokens = TokenCounter.estimate_tokens(prompt, self.model_id)
            
            # Use model-specific configuration for formatting
            body = self.model_config.format_request(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=getattr(settings, 'llm_top_p', 0.4)
            )

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response["body"].read())
            generated_text = self.model_config.extract_response(response_body)
            
            # Costs calculation (deactivated)
            # Count output tokens
            # output_tokens = TokenCounter.estimate_tokens(generated_text, self.model_id)
            # 
            # # Extract token usage from response (if available)
            # token_usage = TokenCounter.extract_token_usage_from_response(response_body, self.model_id)
            # 
            # # If no usage data from AWS, use our estimation
            # if token_usage.input_tokens == 0 and token_usage.output_tokens == 0:
            #     token_usage = TokenUsage(
            #         input_tokens=input_tokens,
            #         output_tokens=output_tokens,
            #         total_tokens=input_tokens + output_tokens,
            #         model_id=self.model_id
            #     )
            # 
            # # Calculate and log costs
            # cost_calc = TokenCounter.calculate_cost(token_usage)
            # TokenCounter.log_usage_and_cost(token_usage, cost_calc, f"LLM GENERATE - {self.model_id}")
            # # Costs calculation (deactivated)

            return generated_text
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"AWS ClientError in generate: {error_code} - {e}")
            if error_code == 'ValidationException':
                raise ValueError(f"Invalid parameters for model {self.model_id}: {str(e)}")
            elif error_code == 'ThrottlingException':
                raise ConnectionError(f"Rate limit exceeded for model {self.model_id}")
            elif error_code == 'ServiceQuotaExceededException':
                raise ConnectionError(f"Service quota exceeded for model {self.model_id}")
            elif error_code == 'ModelNotReadyException':
                raise ValueError(f"Model {self.model_id} is not ready")
            else:
                raise ConnectionError(f"AWS Bedrock error: {error_code}")
                
        except NoCredentialsError as e:
            logger.error(f"AWS credentials error in generate: {e}")
            raise ConnectionError("AWS credentials not configured or invalid")
            
        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error in generate: {e}")
            raise ConnectionError("Unable to connect to AWS Bedrock service")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in generate: {e}")
            raise ValueError("Invalid response format from LLM service")
            
        except KeyError as e:
            logger.error(f"Missing key in response: {e}")
            raise ValueError("Unexpected response format from LLM service")
            
        except Exception as e:
            logger.error(f"Unexpected error in generate: {e}")
            raise ConnectionError(f"LLM service error: {str(e)}")

    async def generate_stream(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> AsyncGenerator[str, None]:
        """
        Genera texto usando streaming con AWS Bedrock invoke_model_with_response_stream.
        Includes token counting and cost calculation for streaming.
        """
        try:
            from app.core.config import settings
            
            # Costs calculation (deactivated)
            # Count input tokens
            # input_tokens = TokenCounter.estimate_tokens(prompt, self.model_id)
            generated_text = ""  # Accumulate for output token counting
            
            # Use model-specific configuration for formatting
            body = self.model_config.format_request(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=getattr(settings, 'llm_top_p', 0.9)
            )

            response = self.client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=body,
                accept="application/json",
                contentType="application/json",
            )

            # Process streaming response
            for event in response.get("body", []):
                try:
                    chunk = event.get("chunk")
                    if chunk:
                        chunk_data = json.loads(chunk.get("bytes").decode())
                        text_chunk = self.model_config.extract_stream_chunk(chunk_data)
                        if text_chunk:
                            generated_text += text_chunk  # Accumulate for token counting
                            yield text_chunk
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in streaming chunk: {e}")
                    continue  # Skip malformed chunks
                except KeyError as e:
                    logger.error(f"Missing key in streaming chunk: {e}")
                    continue  # Skip chunks with missing data
                except Exception as e:
                    logger.error(f"Error processing streaming chunk: {e}")
                    continue  # Skip problematic chunks
            
            # Costs calculation (deactivated)
            # After streaming is complete, calculate and log token usage
            # output_tokens = TokenCounter.estimate_tokens(generated_text, self.model_id)
            # token_usage = TokenUsage(
            #     input_tokens=input_tokens,
            #     output_tokens=output_tokens,
            #     total_tokens=input_tokens + output_tokens,
            #     model_id=self.model_id
            # )
            # 
            # # Calculate and log costs
            # cost_calc = TokenCounter.calculate_cost(token_usage)
            # TokenCounter.log_usage_and_cost(token_usage, cost_calc, f"LLM STREAM - {self.model_id}")
            # # Costs calculation (deactivated)        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"AWS ClientError in generate_stream: {error_code} - {e}")
            if error_code == 'ValidationException':
                raise ValueError(f"Invalid parameters for model {self.model_id}: {str(e)}")
            elif error_code == 'ThrottlingException':
                raise ConnectionError(f"Rate limit exceeded for model {self.model_id}")
            elif error_code == 'ServiceQuotaExceededException':
                raise ConnectionError(f"Service quota exceeded for model {self.model_id}")
            elif error_code == 'ModelNotReadyException':
                raise ValueError(f"Model {self.model_id} is not ready")
            else:
                raise ConnectionError(f"AWS Bedrock error: {error_code}")
                
        except NoCredentialsError as e:
            logger.error(f"AWS credentials error in generate_stream: {e}")
            raise ConnectionError("AWS credentials not configured or invalid")
            
        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error in generate_stream: {e}")
            raise ConnectionError("Unable to connect to AWS Bedrock service")
            
        except Exception as e:
            logger.error(f"Unexpected error in generate_stream: {e}")
            raise ConnectionError(f"LLM streaming service error: {str(e)}")
