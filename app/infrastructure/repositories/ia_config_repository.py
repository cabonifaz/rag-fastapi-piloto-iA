"""Repository for IA area configuration operations."""

import logging
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)


class IaConfigRepository:
    """
    Repository for managing IA area configuration.
    Handles database operations for loading IA area-specific settings.
    """

    def __init__(self, db: Session):
        """
        Initialize the repository with a database session.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    def get_ia_area_config(self, id_ia_area: int) -> Optional[str]:
        """
        Load IA area configuration from database using stored procedure.

        Calls SP_IA_AREA_CONFIG_LOAD to retrieve configuration text for the specified IA area.

        Args:
            id_ia_area: ID of the IA area

        Returns:
            Configuration string (max 1000 characters) from stored procedure, or None if not found

        Raises:
            Exception: If database operation fails
        """
        try:
            query = text("""
                EXEC SP_IA_AREA_CONFIG_LOAD
                @ID_IA_AREA = :id_ia_area
            """)

            result = self.db.execute(query, {
                'id_ia_area': id_ia_area
            })

            config_data = result.fetchone()
            result.close()

            if not config_data:
                logger.info(f"No config found for id_ia_area={id_ia_area}")
                return None

            # Convert result to dictionary
            config_dict = dict(config_data._mapping) if hasattr(config_data, '_mapping') else dict(zip(result.keys(), config_data))

            # Get the first value from the result (config text)
            config_text = list(config_dict.values())[0] if config_dict else None

            if config_text and str(config_text).strip():
                # Limit to 1000 characters as per business rule
                return str(config_text)[:1000]

            return None

        except Exception as e:
            logger.error(f"Error loading IA area config for id_ia_area={id_ia_area}: {e}")
            raise
