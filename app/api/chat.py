# app/api/chat.py

from fastapi import APIRouter, Depends
from app.application.chat_service import ChatService
from app.core.config import settings

# ---- Importamos interfaces y adaptadores ----
from app.domain.ports.llm_port import LLMPort
from app.domain.ports.embeddings_port import EmbeddingsPort

# LLM providers
from app.infrastructure.llm.aws_provider import AWSLLMProvider
# (puedes agregar OpenAIProvider, AnthropicProvider, etc.)

# Embeddings providers
from app.infrastructure.embeddings.openai_embeddings import OpenAIEmbeddingsProvider
from app.infrastructure.embeddings.aws_embeddings import AWSBedrockEmbeddingsProvider

# Vectorstore (ejemplo con Weaviate, pero puede ser Chroma u otro)
from app.infrastructure.vectorstores.weaviate_repository import WeaviateRepository


router = APIRouter()


# ---------- FACTORY PARA LLM ----------
def get_llm_provider() -> LLMPort:
    if settings.LLM_PROVIDER == "aws":
        return AWSLLMProvider(
            region=settings.LLM_REGION,
            model_id=settings.LLM_MODEL_ID,
            access_key=settings.LLM_API_KEY or None,
            secret_key=None  # si se usan roles de IAM no es necesario
        )
    # elif settings.LLM_PROVIDER == "openai":
    #     return OpenAILLMProvider(api_key=settings.LLM_API_KEY, model=settings.LLM_MODEL_ID)
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")


# ---------- FACTORY PARA EMBEDDINGS ----------
def get_embeddings_provider() -> EmbeddingsPort:
    if settings.EMBEDDINGS_PROVIDER == "openai":
        return OpenAIEmbeddingsProvider(
            api_key=settings.EMBEDDINGS_API_KEY,
            model=settings.EMBEDDINGS_MODEL
        )
    elif settings.EMBEDDINGS_PROVIDER == "aws":
        return AWSBedrockEmbeddingsProvider(
            region=settings.EMBEDDINGS_REGION,
            model_id=settings.EMBEDDINGS_MODEL_ID,
            access_key=settings.AWS_ACCESS_KEY_ID,
            secret_key=settings.AWS_SECRET_ACCESS_KEY
        )
    else:
        raise ValueError(f"Unsupported Embeddings provider: {settings.EMBEDDINGS_PROVIDER}")


# ---------- FACTORY PARA CHAT SERVICE ----------
def get_chat_service() -> ChatService:
    llm_provider = get_llm_provider()
    embeddings_provider = get_embeddings_provider()
    vectorstore = WeaviateRepository(
        url=settings.VECTORDB_URL,
        api_key=settings.VECTORDB_API_KEY
    )

    return ChatService(
        llm_provider=llm_provider,
        embeddings_provider=embeddings_provider,
        vectorstore=vectorstore
    )


# ---------- ENDPOINT ----------
@router.post("/chat")
async def chat_endpoint(user_id: str, message: str, chat_service: ChatService = Depends(get_chat_service)):
    """
    Chat endpoint with RAG.
    """
    return await chat_service.chat(user_id=user_id, message=message)
