import boto3
import json
from typing import Optional
from app.domain.ports.llm_port import LLMPort
from app.infrastructure.llm.model_formats import ModelFormatFactory


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
