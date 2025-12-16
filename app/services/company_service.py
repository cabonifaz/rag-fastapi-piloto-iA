"""Service for managing company operations following hexagonal architecture."""

from typing import List, Dict, Any
import logging
import time
from sqlalchemy.orm import Session
from app.infrastructure.repositories.company_repository import CompanyRepository
from app.core.config import settings
from app.core.aws_clients import get_s3_client

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

    async def get_companies_login(self, db: Session) -> List[Dict[str, Any]]:
        """
        Get companies with their secret keys using stored procedure SP_EMPRESAS_LST_LOGIN.

        Args:
            db: Database session

        Returns:
            List of dictionaries containing RAZON_SOCIAL and SECRET_KEY
            Empty list if fetch failed
        """
        try:
            # Create repository for this request
            repository = CompanyRepository(db)

            # Get companies login data from repository
            companies = repository.get_companies_login()

            if companies:
                logger.info(f"Retrieved {len(companies)} companies with login credentials")
            else:
                logger.warning("No companies found with login credentials")

            return companies

        except Exception as e:
            logger.error(f"Error in get_companies_login service: {e}")
            return []

    async def generate_logo_presigned_url(
        self,
        db: Session,
        id_usuario: int,
        id_empresa: int,
        logo_filename: str
    ) -> Dict[str, Any]:
        """
        Generate presigned URL for company logo upload and update database.

        This method combines both operations:
        1. Generates presigned S3 URL for upload
        2. Updates company record in database with logo path
        3. Frontend then uploads directly to S3 using presigned URL

        Args:
            db: Database session
            id_usuario: User ID performing the update
            id_empresa: Company ID
            logo_filename: Original filename (e.g., "logo.png")

        Returns:
            Dictionary containing:
            - presigned_url: S3 presigned PUT URL (5 min expiration)
            - s3_key: S3 object key path
            - logo_filename: Original filename
            - results: DB update results (ID_TIPO_MENSAJE, MENSAJE)

        Raises:
            ValueError: If file extension is not allowed or company ID is invalid
        """
        try:
            # Validate company ID
            if id_empresa <= 0:
                raise ValueError("Invalid company ID")

            # Validate file extension
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.svg']
            file_ext = logo_filename.lower()[logo_filename.rfind('.'):]

            if file_ext not in allowed_extensions:
                raise ValueError(f"Invalid file type. Allowed: {', '.join(allowed_extensions)}")

            # Get S3 bucket name (use logos bucket)
            bucket_name = settings.s3_logos_bucket

            # Generate deterministic S3 key: logos/{empresa_id}/logo.{extension}
            # This ensures old logos are overwritten when a new one is uploaded
            s3_key = f"logos/{id_empresa}/logo-{id_empresa}{file_ext}"

            # Step 1: Generate presigned PUT URL using aioboto3 (async)
            try:
                async with get_s3_client() as s3_client:
                    presigned_url = await s3_client.generate_presigned_url(
                        'put_object',
                        Params={
                            'Bucket': bucket_name,
                            'Key': s3_key,
                        },
                        ExpiresIn=300,  # 5 minutes
                        HttpMethod='PUT'
                    )
            except Exception as s3_error:
                logger.error(f"Error generating presigned URL for S3: {s3_error}")
                return {
                    'presigned_url': None,
                    's3_key': s3_key,
                    'logo_filename': logo_filename,
                    'results': [{
                        'ID_TIPO_MENSAJE': 1,
                        'MENSAJE': 'Generacion de URL prefirmada fallo'
                    }]
                }

            # Step 2: Update database with logo path AFTER presigned URL is generated
            # Only updates DB if presigned URL generation was successful
            repository = CompanyRepository(db)
            db_results = repository.update_company_logo(
                id_usuario=id_usuario,
                id_empresa=id_empresa,
                logo_url=s3_key
            )

            logger.info(f"Generated presigned URL and updated DB for logo: {s3_key}")

            return {
                'presigned_url': presigned_url,
                's3_key': s3_key,
                'logo_filename': logo_filename,
                'results': db_results
            }

        except ValueError as ve:
            logger.error(f"Validation error in generate_logo_presigned_url: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error generating logo presigned URL: {e}")
            raise
