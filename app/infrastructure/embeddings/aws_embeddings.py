# app/infrastructure/embeddings/aws_embeddings.py

import boto3
import json
from typing import List, Optional
from app.domain.ports.embeddings_port import EmbeddingsPort


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
        return response_body.get("embedding", [])
