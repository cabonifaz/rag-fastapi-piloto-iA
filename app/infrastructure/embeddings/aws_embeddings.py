import boto3
import json
import logging
from typing import List, Optional
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from app.domain.ports.embeddings_port import EmbeddingsPort

# Configure logging
logger = logging.getLogger(__name__)


class AWSBedrockEmbeddingsProvider(EmbeddingsPort):
    """
    Adaptador para usar embeddings desde AWS Bedrock (ej. Titan Embeddings).
    Implementa EmbeddingsPort.
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
        :param model_id: ID del modelo de embeddings (ej. "amazon.titan-embed-text-v1")
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

    async def embed(self, text: str) -> List[float]:
        """
        Genera embeddings desde un modelo de AWS Bedrock.
        """
        try:
            if not text or not text.strip():
                raise ValueError("Input text cannot be empty")
                
            body = json.dumps({
                "inputText": text
            })

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response["body"].read())
            # Titan embeddings devuelve "embedding"
            embedding = response_body.get("embedding", [])
            
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
