"""Service for managing company operations following hexagonal architecture."""

from typing import List, Dict, Any
import logging
from sqlalchemy.orm import Session
from app.infrastructure.repositories.company_repository import CompanyRepository

logger = logging.getLogger(__name__)


class CompanyService:
    """
    Service for company operations.
    Handles business logic for creating companies and related entities.
    """

    def __init__(self):
        """Initialize stateless CompanyService - no db parameter."""
        pass

    async def create_company(
        self,
        db: Session,
        id_usuario: int,
        ruc: str,
        razon_social: str
    ) -> List[Dict[str, Any]]:
        """
        Create a new company base with associated roles and areas using stored procedure.

        Args:
            db: Database session
            id_usuario: User ID creating the company
            ruc: RUC identifier (max 30 chars)
            razon_social: Company name (max 255 chars)

        Returns:
            List of dictionaries containing:
            - id_rol: Role ID
            - message: Status message
            - id_company: Company ID
            - id_area: Area ID
            Empty list if creation failed
        """
        try:
            # Create repository for this request
            repository = CompanyRepository(db)

            # Validate input
            if not ruc or len(ruc.strip()) == 0:
                logger.error("RUC cannot be empty")
                return []

            if not razon_social or len(razon_social.strip()) == 0:
                logger.error("RAZON_SOCIAL cannot be empty")
                return []

            # Trim inputs to match database constraints
            ruc = ruc.strip()[:30]
            razon_social = razon_social.strip()[:255]

            # Use repository to create company with SP_CREATE_EMPRESA_BASE
            results = repository.create_company(
                id_usuario=id_usuario,
                ruc=ruc,
                razon_social=razon_social
            )

            if results:
                logger.info(f"Company created successfully: RUC={ruc}, Results count={len(results)}")
            else:
                logger.warning(f"Company creation returned no results: RUC={ruc}")

            return results

        except Exception as e:
            logger.error(f"Error in create_company service: {e}")
            raise

    async def get_companies(self, db: Session) -> List[Dict[str, Any]]:
        """
        Get all companies using stored procedure SP_EMPRESAS_LST.

        Args:
            db: Database session

        Returns:
            List of dictionaries containing company data
            Empty list if fetch failed
        """
        try:
            # Create repository for this request
            repository = CompanyRepository(db)

            # Get companies from repository
            companies = repository.get_companies()

            if companies:
                logger.info(f"Retrieved {len(companies)} companies")
            else:
                logger.warning("No companies found")

            return companies

        except Exception as e:
            logger.error(f"Error in get_companies service: {e}")
            raise

    async def update_company_status(
        self,
        db: Session,
        id_usuario: int,
        id_empresa: int,
        status: int
    ) -> List[Dict[str, Any]]:
        """
        Update company status using stored procedure SP_UPDATE_EMPRESA_STATUS.

        Args:
            db: Database session
            id_usuario: User ID performing the update
            id_empresa: Company ID to update
            status: Status value to set (typically 1=active, 0=deleted)

        Returns:
            List of dictionaries containing update result information
            Empty list if update failed
        """
        try:
            # Create repository for this request
            repository = CompanyRepository(db)

            # Validate inputs
            if id_empresa <= 0:
                logger.error("Invalid company ID")
                return []

            if status not in [0, 1]:
                logger.error(f"Invalid status value: {status}. Expected 0 or 1")
                return []

            # Update company status via repository
            results = repository.update_company_status(
                id_usuario=id_usuario,
                id_empresa=id_empresa,
                status=status
            )

            if results:
                logger.info(f"Company status updated successfully: ID_EMPRESA={id_empresa}, STATUS={status}, Results count={len(results)}")
            else:
                logger.warning(f"Company status update returned no results: ID_EMPRESA={id_empresa}")

            return results

        except Exception as e:
            logger.error(f"Error in update_company_status service: {e}")
            raise
