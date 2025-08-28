import boto3
import json
import logging
from typing import Optional, AsyncGenerator
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from app.domain.ports.llm_port import LLMPort
from app.infrastructure.llm.model_formats import ModelFormatFactory
from app.utils.token_counter import TokenCounter, TokenUsage

# Configure logging
logger = logging.getLogger(__name__)


class AWSLLMProvider(LLMPort):
    """
    Adaptador para usar un LLM desde AWS (ej. Bedrock).
    Implementa el puerto LLMPort.
    """

    def __init__(
        self,
        region: str,
        model_id: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        """
        Inicializa el cliente de AWS Bedrock.

        :param region: región de AWS (ej. "us-east-1")
        :param model_id: ID del modelo de Bedrock (ej. "anthropic.claude-v2")
        :param access_key: AWS_ACCESS_KEY_ID (opcional si usas IAM Role)
        :param secret_key: AWS_SECRET_ACCESS_KEY (opcional si usas IAM Role)
        """
        session_params = {"region_name": region}
        if access_key and secret_key:
            session_params["aws_access_key_id"] = access_key
            session_params["aws_secret_access_key"] = secret_key

        session = boto3.Session(**session_params)
        self.client = session.client("bedrock-runtime")
        self.model_id = model_id
        self.format_strategy = ModelFormatFactory.get_format_strategy(model_id)

    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Genera texto usando un modelo de AWS Bedrock.
        Soporta tanto modelos Claude como Llama3.
        Includes token counting and cost calculation.
        """
        try:
            from app.core.config import settings
            
            # Costs calculation
            # Count input tokens
            input_tokens = TokenCounter.estimate_tokens(prompt, self.model_id)
            
            # Use strategy pattern for model-specific formatting
            body = self.format_strategy.format_request(
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
            generated_text = self.format_strategy.extract_response(response_body)
            
            # Costs calculation
            # Count output tokens
            output_tokens = TokenCounter.estimate_tokens(generated_text, self.model_id)
            
            # Extract token usage from response (if available)
            token_usage = TokenCounter.extract_token_usage_from_response(response_body, self.model_id)
            
            # If no usage data from AWS, use our estimation
            if token_usage.input_tokens == 0 and token_usage.output_tokens == 0:
                token_usage = TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    model_id=self.model_id
                )
            
            # Calculate and log costs
            cost_calc = TokenCounter.calculate_cost(token_usage)
            TokenCounter.log_usage_and_cost(token_usage, cost_calc, f"LLM GENERATE - {self.model_id}")
            # Costs calculation

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
            
            # Costs calculation
            # Count input tokens
            input_tokens = TokenCounter.estimate_tokens(prompt, self.model_id)
            generated_text = ""  # Accumulate for output token counting
            
            # Use strategy pattern for model-specific formatting
            body = self.format_strategy.format_request(
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
                        text_chunk = self.format_strategy.extract_stream_chunk(chunk_data)
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
            
            # Costs calculation
            # After streaming is complete, calculate and log token usage
            output_tokens = TokenCounter.estimate_tokens(generated_text, self.model_id)
            token_usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                model_id=self.model_id
            )
            
            # Calculate and log costs
            cost_calc = TokenCounter.calculate_cost(token_usage)
            TokenCounter.log_usage_and_cost(token_usage, cost_calc, f"LLM STREAM - {self.model_id}")
            # Costs calculation        
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
