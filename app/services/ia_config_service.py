"""Service for managing IA area configuration."""

import logging
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from decimal import Decimal
from app.core.config import settings
from app.infrastructure.repositories.ia_config_repository import IaConfigRepository

logger = logging.getLogger(__name__)


class IaConfigService:
    """
    Service for IA area configuration operations.
    Handles business logic for loading and managing IA area-specific settings.
    """

    def __init__(self):
        """Initialize stateless IaConfigService - no db parameter."""
        pass

    async def get_ia_area_config(self, db: Session, id_ia_area: int) -> str:
        """
        Load IA area configuration from database using stored procedure.
        Falls back to LLM_ROLE_BEHAVIOR from env if SP returns no value or fails.

        Args:
            db: Database session
            id_ia_area: ID of the IA area

        Returns:
            Configuration string (max 1000 characters) from SP or LLM_ROLE_BEHAVIOR from env
        """
        try:
            if not db:
                logger.warning("Database session not available in IaConfigService, using llm_role_behavior from env")
                return settings.llm_role_behavior

            # Create repository for this request
            repository = IaConfigRepository(db)

            # Get configuration from database
            config_text = repository.get_ia_area_config(id_ia_area)

            if config_text:
                return config_text

            # No valid config text, use env fallback
            logger.info(f"No valid config found for id_ia_area={id_ia_area}, using llm_role_behavior from env")
            return settings.llm_role_behavior

        except Exception as e:
            logger.error(f"Error in get_ia_area_config service for id_ia_area={id_ia_area}: {e}, using llm_role_behavior from env")
            return settings.llm_role_behavior

    async def update_ia_area_config(
        self,
        db: Session,
        id_usuario: int,
        id_empresa: int,
        id_area: int,
        id_embeddings: int,
        id_llm: int,
        embeddings_dimensions: int,
        llm_max_tokens: int,
        llm_temperature: Decimal,
        llm_top_p: Decimal,
        rag_top_k_results: int,
        rag_similarity_threshold: Decimal,
        rag_alpha: Decimal,
        role_behavior: str
    ) -> Dict[str, Any]:
        """
        Update IA area configuration using stored procedure.

        Args:
            db: Database session
            id_usuario: User ID performing the update
            id_empresa: Company ID
            id_area: Area ID
            id_embeddings: Embeddings model ID
            id_llm: LLM model ID
            embeddings_dimensions: Embedding vector dimensions
            llm_max_tokens: Maximum tokens for LLM
            llm_temperature: Temperature parameter for LLM
            llm_top_p: Top-P parameter for LLM
            rag_top_k_results: Number of top-K results for RAG
            rag_similarity_threshold: Similarity threshold for RAG
            rag_alpha: Alpha parameter for RAG
            role_behavior: Role behavior prompt/instructions

        Returns:
            Dictionary containing ID_TIPO_MENSAJE and message
        """
        try:
            # Create repository for this request
            repository = IaConfigRepository(db)

            # Use repository to update IA area config with SP_UPDATE_IA_AREA_BASE
            result = await repository.update_ia_area_config(
                id_usuario=id_usuario,
                id_empresa=id_empresa,
                id_area=id_area,
                id_embeddings=id_embeddings,
                id_llm=id_llm,
                embeddings_dimensions=embeddings_dimensions,
                llm_max_tokens=llm_max_tokens,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                rag_top_k_results=rag_top_k_results,
                rag_similarity_threshold=rag_similarity_threshold,
                rag_alpha=rag_alpha,
                role_behavior=role_behavior
            )

            if result and result.get('ID_TIPO_MENSAJE'):
                logger.info(f"IA area config updated successfully: User={id_usuario}, Area={id_area}")
            else:
                logger.warning(f"IA area config update returned unexpected result: {result}")

            return result

        except Exception as e:
            logger.error(f"Error in update_ia_area_config service: {e}")
            raise
