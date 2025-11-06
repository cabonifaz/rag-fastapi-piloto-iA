import aioboto3
import json
import logging
import os
import asyncio
from typing import List, Optional
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from app.domain.ports.embeddings_port import EmbeddingsPort
from app.infrastructure.embeddings.titan_embeddings import TitanEmbedConfig
from app.infrastructure.embeddings.cohere_embeddings import CohereEmbedConfig
# from app.utils.token_counter import TokenCounter, TokenUsage

# Configure logging
logger = logging.getLogger(__name__)


def get_embedding_config(model_id: str):
    """Factory function to get the appropriate embedding model configuration."""
    model_id_lower = model_id.lower()

    if "titan-embed" in model_id_lower or "amazon.titan" in model_id_lower:
        return TitanEmbedConfig(model_id)
    elif "cohere.embed" in model_id_lower:
        return CohereEmbedConfig(model_id)
    else:
        # Default to Titan format for unknown models
        logger.warning(f"Unknown embedding model {model_id}, using Titan format as default")
        return TitanEmbedConfig(model_id)


class AWSBedrockEmbeddingsProvider(EmbeddingsPort):
    """
    Adaptador para usar embeddings desde AWS Bedrock (ej. Titan Embeddings, Cohere).
    Implementa EmbeddingsPort.
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
        Initialize AWS Bedrock embeddings client using aioboto3 (async).

        :param region: AWS region (ej. "us-east-1")
        :param model_id: Embedding model ID (ej. "amazon.titan-embed-text-v1" or "cohere.embed-english-v3")
        :param profile_name: AWS profile name (optional if using IAM Role)
        :param aws_access_key_id: AWS access key ID (optional, used if no profile)
        :param aws_secret_access_key: AWS secret access key (optional, used if no profile)
        """
        session_params = {"region_name": region}

        # Use profile only in development, not in production with IAM Role
        if profile_name and os.getenv('ENVIRONMENT', '').lower() != 'production':
            session_params["profile_name"] = profile_name
        # If no profile, use direct credentials if available
        elif aws_access_key_id and aws_secret_access_key:
            session_params["aws_access_key_id"] = aws_access_key_id
            session_params["aws_secret_access_key"] = aws_secret_access_key

        # Create aioboto3 session (don't create client yet)
        self.session = aioboto3.Session(**session_params)
        self.region = region
        self.model_id = model_id
        self.model_config = get_embedding_config(model_id)

    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings from AWS Bedrock using aioboto3 (truly async).
        """
        try:
            if not text or not text.strip():
                raise ValueError("Input text cannot be empty")

            # Use model-specific configuration to format request
            body = self.model_config.format_request(text)

            # Use aioboto3 async client for truly non-blocking Bedrock calls
            async with self.session.client("bedrock-runtime") as client:
                response = await client.invoke_model(
                    modelId=self.model_id,
                    body=body,
                    accept="application/json",
                    contentType="application/json"
                )

                # Read and parse response body (aioboto3 returns async StreamingBody)
                response_body_bytes = await response["body"].read()
                response_body = json.loads(response_body_bytes)

            # Use model-specific configuration to extract embedding
            embedding = self.model_config.extract_embedding(response_body)

            if not embedding:
                raise ValueError("Empty embedding returned from service")

            return embedding
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"AWS ClientError in embed: {error_code} - {e}")
            if error_code == 'ValidationException':
                raise ValueError(f"Invalid input for embedding model {self.model_id}: {str(e)}")
            elif error_code == 'ThrottlingException':
                raise ConnectionError(f"Rate limit exceeded for embedding model {self.model_id}")
            elif error_code == 'ServiceQuotaExceededException':
                raise ConnectionError(f"Service quota exceeded for embedding model {self.model_id}")
            elif error_code == 'ModelNotReadyException':
                raise ValueError(f"Embedding model {self.model_id} is not ready")
            else:
                raise ConnectionError(f"AWS Bedrock embedding error: {error_code}")
                
        except NoCredentialsError as e:
            logger.error(f"AWS credentials error in embed: {e}")
            raise ConnectionError("AWS credentials not configured or invalid")
            
        except EndpointConnectionError as e:
            logger.error(f"AWS endpoint connection error in embed: {e}")
            raise ConnectionError("Unable to connect to AWS Bedrock embeddings service")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in embed: {e}")
            raise ValueError("Invalid response format from embedding service")
            
        except KeyError as e:
            logger.error(f"Missing key in embedding response: {e}")
            raise ValueError("Unexpected response format from embedding service")
            
        except ValueError:
            raise  # Re-raise validation errors
            
        except Exception as e:
            logger.error(f"Unexpected error in embed: {e}")
            raise ConnectionError(f"Embedding service error: {str(e)}")
