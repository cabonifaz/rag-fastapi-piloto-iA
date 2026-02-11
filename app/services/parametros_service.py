"""Service wrapper for parameter repository."""

import logging
from typing import Optional
from sqlalchemy.orm import Session
from app.infrastructure.repositories.parametros_repository import ParametrosRepository

logger = logging.getLogger(__name__)


class ParametrosService:
    """Stateless service that fetches parameter values from DB."""

    def __init__(self):
        """Initialize stateless ParametrosService - no db parameter."""
        pass

    async def get_param_by_id_maestro(self, db: Session, grp_id_maestro: str) -> Optional[int]:
        """
        Return numeric value (NUM1) for the given GRP_ID_MAESTRO parameter.

        Args:
            db: Database session
            grp_id_maestro: Group ID (GRP_ID_MAESTRO) to look up

        Returns:
            Integer NUM1 value for the matching ID_MAESTRO, or None if not found
        """
        try:
            # Create repository for this request
            repository = ParametrosRepository(db)

            # Validate input
            if not grp_id_maestro or len(grp_id_maestro.strip()) == 0:
                logger.error("grp_id_maestro cannot be empty")
                return None

            val = repository.get_params_by_id_maestro(grp_id_maestro)

            if val is None:
                logger.warning(f"Parameter not found for grp_id_maestro={grp_id_maestro}")
                return None

            logger.info(f"Parameter fetched successfully: grp_id_maestro={grp_id_maestro}, value={val}")
            return val

        except Exception as e:
            logger.error(f"Error in get_param_by_id_maestro grp_id_maestro={grp_id_maestro}: {e}")
            raise