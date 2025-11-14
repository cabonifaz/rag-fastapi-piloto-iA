"""Service for managing area operations following hexagonal architecture."""

from typing import List, Dict, Any
import logging
from sqlalchemy.orm import Session
from app.infrastructure.repositories.area_repository import AreaRepository

logger = logging.getLogger(__name__)


class AreaService:
    """
    Service for area operations.
    Handles business logic for creating areas.
    """

    def __init__(self):
        """Initialize stateless AreaService - no db parameter."""
        pass

    async def create_area(
        self,
        db: Session,
        id_usuario: int,
        id_empresa: int,
        area: str
    ) -> List[Dict[str, Any]]:
        """
        Create a new area base using stored procedure.

        Args:
            db: Database session
            id_usuario: User ID creating the area
            id_empresa: Company ID
            area: Area name (max 100 chars)

        Returns:
            List of dictionaries containing:
            - ID_TIPO_MENSAJE: Message type ID
            - MENSAJE: Status message
            Empty list if creation failed
        """
        try:
            # Create repository for this request
            repository = AreaRepository(db)

            # Validate input
            if not area or len(area.strip()) == 0:
                logger.error("Area cannot be empty")
                return []

            # Trim inputs to match database constraints
            area = area.strip()[:100]

            # Use repository to create area with SP_CREATE_AREA_BASE
            results = repository.create_area(
                id_usuario=id_usuario,
                id_empresa=id_empresa,
                area=area
            )

            if results:
                logger.info(f"Area created successfully: Area={area}, Results count={len(results)}")
            else:
                logger.warning(f"Area creation returned no results: Area={area}")

            return results

        except Exception as e:
            logger.error(f"Error in create_area service: {e}")
            raise
