"""Service for managing phone code operations following hexagonal architecture."""

from typing import List, Dict, Any
import logging
from sqlalchemy.orm import Session
from app.infrastructure.repositories.phone_code_repository import PhoneCodeRepository

logger = logging.getLogger(__name__)


class PhoneCodeService:
    """
    Service for phone code operations.
    Handles business logic for retrieving country phone codes.
    """

    def __init__(self):
        """Initialize stateless PhoneCodeService - no db parameter."""
        pass

    async def get_phone_codes(
        self,
        db: Session
    ) -> List[Dict[str, Any]]:
        """
        Get all phone codes using stored procedure.

        Args:
            db: Database session

        Returns:
            List of dictionaries containing phone code information:
            - CODIGO_NUMERICO: Numeric country code
            - CODIGO_ISO: ISO country code (e.g., PE, ES, MX)
            - NOMBRE_PAIS: Country name
            - PREFIJO_TELEFONICO: Phone prefix (e.g., +51, +34, +52)
            Empty list if query failed
        """
        try:
            # Create repository for this request
            repository = PhoneCodeRepository(db)

            # Use repository to get phone codes with SP_CODIGOS_PAIS_LST
            results = repository.get_phone_codes()

            if results:
                logger.info(f"Phone codes retrieved successfully: Count={len(results)}")
            else:
                logger.info("No phone codes found")

            return results

        except Exception as e:
            logger.error(f"Error in get_phone_codes service: {e}")
            raise
