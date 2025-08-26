import boto3
import json
import logging
from typing import Optional, AsyncGenerator
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from app.domain.ports.llm_port import LLMPort
from app.infrastructure.llm.model_formats import ModelFormatFactory

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
        """
        try:
            from app.core.config import settings
            
            # Use strategy pattern for model-specific formatting
            body = self.format_strategy.format_request(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=getattr(settings, 'llm_top_p', 0.9)
            )

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response["body"].read())
            return self.format_strategy.extract_response(response_body)
            
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
        """
        try:
            from app.core.config import settings
            
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
