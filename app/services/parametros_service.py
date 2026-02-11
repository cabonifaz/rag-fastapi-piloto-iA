"""Service wrapper for parameter repository."""

import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.infrastructure.repositories.parametros_repository import ParametrosRepository

logger = logging.getLogger(__name__)


class ParametrosService:
    """Stateless service that fetches parameter values from DB."""

    def __init__(self):
        """Initialize stateless ParametrosService - no db parameter."""
        pass

    async def get_param_by_id_maestro(self, db: Session, grp_id_maestro: str) -> List[Dict[str, Any]]:
        """
        Return all parameter rows for the given GRP_ID_MAESTRO.

        Args:
            db: Database session
            grp_id_maestro: Group ID (GRP_ID_MAESTRO) to look up

        Returns:
            List of dictionaries with all parameter rows for the group.
            Empty list if not found.
        """
        try:
            # Create repository for this request
            repository = ParametrosRepository(db)

            # Validate input
            if not grp_id_maestro or len(grp_id_maestro.strip()) == 0:
                logger.error("grp_id_maestro cannot be empty")
                return []

            results = repository.get_params_by_id_maestro(grp_id_maestro)

            if not results:
                logger.warning(f"No parameters found for grp_id_maestro={grp_id_maestro}")
                return []

            logger.info(f"Parameters fetched successfully: grp_id_maestro={grp_id_maestro}, count={len(results)}")
            return results

        except Exception as e:
            logger.error(f"Error in get_param_by_id_maestro grp_id_maestro={grp_id_maestro}: {e}")
            raise
