"""Service for managing IA area configuration."""

import logging
from typing import Optional
from sqlalchemy.orm import Session
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
