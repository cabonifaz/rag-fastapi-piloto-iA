"""Service wrapper for parameter repository."""

import logging
from typing import Optional
from sqlalchemy.orm import Session
from app.infrastructure.repositories.parametros_repository import ParametrosRepository

logger = logging.getLogger(__name__)


class ParametrosService:
    """Stateless service that fetches parameter values from DB."""

    def __init__(self):
        pass

    async def get_numeric_param(self, db: Session, num1: int, default: Optional[int] = None) -> Optional[int]:
        """Return numeric value for the given parameter NUM1. Returns default if not found."""
        try:
            if not db:
                logger.warning("Database session not available in ParametrosService")
                return default

            repo = ParametrosRepository(db)
            val = repo.get_param_by_num1(num1)

            if val is None:
                return default

            return val

        except Exception as e:
            logger.error(f"Error in get_numeric_param num1={num1}: {e}")
            return default