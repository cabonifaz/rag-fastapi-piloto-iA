"""Service for handling knowledge/document operations."""

import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.infrastructure.repositories.knowledge_repository import KnowledgeRepository
from app.core.config import settings
from app.core.aws_clients import get_s3_client

logger = logging.getLogger(__name__)


class KnowledgeService:
    """Service for knowledge/document management operations."""

    def __init__(self, db: Session):
        """Initialize service with repository and S3 bucket."""
        self.repository = KnowledgeRepository(db)
        self.bucket_name = settings.s3_pdfs_bucket

    async def get_knowledge_by_company(
        self,
        id_usuario: int,
        id_empresa: int,
        id_area: int = None
    ) -> List[Dict[str, Any]]:
        """
        Get all knowledge/documents for a company and optional area.

        Args:
            id_usuario: User ID requesting the documents
            id_empresa: Company ID
            id_area: Area ID (optional, retrieves all areas if None)

        Returns:
            List of knowledge/document records
        """
        try:
            knowledge_list = self.repository.get_knowledge_by_company(
                id_usuario=id_usuario,
                id_empresa=id_empresa,
                id_area=id_area
            )

            logger.info(f"Retrieved {len(knowledge_list)} knowledge records for company {id_empresa}, user {id_usuario}")
            return knowledge_list

        except Exception as e:
            logger.error(f"Error getting knowledge for company {id_empresa}: {e}")
            raise

    async def generate_presigned_urls(
        self,
        id_usuario: int,
        id_empresa: int,
        id_area: int,
        pdf_keys: List[str],
        id_modelo_embedding: str = "4"
    ) -> List[Dict[str, Any]]:
        """
        Generate presigned URLs for multiple PDF uploads and create knowledge records in batch.

        Phase 1: Prepare all records and S3 keys (no DB writes yet)
        Phase 2: Batch create all knowledge records in database
        Phase 3: Generate presigned URLs for all records (truly async with aioboto3)

        Args:
            id_usuario: User ID who is uploading
            id_empresa: Company ID
            id_area: Area ID
            pdf_keys: List of PDF filenames/keys
            id_modelo_embedding: Embedding model ID (default: "4")

        Returns:
            List of objects containing:
            - presigned_url: S3 presigned PUT URL (5 min expiration)
            - s3_key: S3 object key path
            - document_name: Original filename
            - results: DB response (ID_TIPO_MENSAJE, MENSAJE)
        """
        try:
            # Phase 1: Prepare all records and S3 keys (no DB writes yet)
            batch_records = []
            s3_keys_map = {}  # Map filename to S3 key for later use

            for pdf_filename in pdf_keys:
                # Construct S3 key: <empresa_id>/<area_id>/<filename>
                s3_key = f"{id_empresa}/{id_area}/{pdf_filename}"

                # Store for later use
                s3_keys_map[pdf_filename] = s3_key

                # Add to batch records (not written to DB yet)
                batch_records.append({
                    'ruta_documento': s3_key,
                    'nombre_documento': pdf_filename
                })

            # Phase 2: Batch create all knowledge records in database (single SP call)
            db_results = self.repository.batch_create_knowledge(
                id_usuario=id_usuario,
                id_empresa=id_empresa,
                id_area=id_area,
                records=batch_records,
                id_modelo_embedding=id_modelo_embedding
            )

            # Phase 3: Generate presigned URLs and build responses (truly async with aioboto3)
            response_objects = await self._generate_presigned_urls_async(
                pdf_keys,
                s3_keys_map,
                db_results
            )

            logger.info(f"Generated {len(response_objects)} presigned URLs for company {id_empresa}, area {id_area}")
            return response_objects

        except Exception as e:
            logger.error(f"Error generating presigned URLs: {e}")
            raise

    async def _generate_presigned_urls_async(
        self,
        pdf_keys: List[str],
        s3_keys_map: Dict[str, str],
        db_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate presigned URLs for all records using aioboto3 (truly async).

        Args:
            pdf_keys: List of PDF filenames
            s3_keys_map: Map of filename to S3 key
            db_results: Database response results

        Returns:
            List of response objects with presigned URLs
        """
        response_objects = []

        # Use aioboto3 S3 client for async presigned URL generation
        async with get_s3_client() as s3_client:
            for pdf_filename in pdf_keys:
                s3_key = s3_keys_map[pdf_filename]

                # Generate presigned PUT URL (5 min expiration)
                presigned_url = await s3_client.generate_presigned_url(
                    'put_object',
                    Params={
                        'Bucket': self.bucket_name,
                        'Key': s3_key,
                        'ContentType': 'application/pdf'
                    },
                    ExpiresIn=300  # 5 minutes
                )

                # Build response object
                response_obj = {
                    'presigned_url': presigned_url,
                    's3_key': s3_key,
                    'document_name': pdf_filename,
                    'results': db_results
                }

                response_objects.append(response_obj)

        return response_objects

    async def update_knowledge_process_state(
        self,
        id_carga: int,
        id_estado_proceso: int,
        usumod: str = "System"
    ) -> List[Dict[str, Any]]:
        """
        Update the process state for a knowledge/document record.

        Args:
            id_carga: Knowledge/document ID to update
            id_estado_proceso: New process state ID
            usumod: User who modified the record (max 200 chars, default: "System")

        Returns:
            List of dictionaries with: ID_TIPO_MENSAJE, MENSAJE
        """
        try:
            results = self.repository.update_knowledge_process_state(
                id_carga=id_carga,
                id_estado_proceso=id_estado_proceso,
                usumod=usumod
            )

            logger.info(f"Updated knowledge process state: id_carga={id_carga}, new state={id_estado_proceso}")
            return results

        except Exception as e:
            logger.error(f"Error updating knowledge process state: {e}")
            raise
