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
                    },
                    ExpiresIn=300,  # 5 minutes
                    HttpMethod='PUT'
                )

                # Build upload object (without results)
                upload_obj = {
                    'presigned_url': presigned_url,
                    's3_key': s3_key,
                    'document_name': pdf_filename,
                }

                response_objects.append(upload_obj)

        return {
            'uploads': response_objects,
            'results': db_results
        }

    async def batch_update_knowledge_state(
        self,
        id_usuario: int,
        id_cargas: List[int],
        id_estado_proceso: int,
        usumod: str = "System"
    ) -> Dict[str, Any]:
        """
        Update the process state for multiple knowledge/document records in batch.

        Args:
            id_usuario: User ID who is updating (for audit purposes)
            id_cargas: List of knowledge/document IDs to update
            id_estado_proceso: New process state ID
            usumod: User who modified the records (max 200 chars, default: "System")

        Returns:
            Dictionary with ID_TIPO_MENSAJE and MENSAJE from the stored procedure
        """
        try:
            message_result = self.repository.batch_update_knowledge_state(
                id_usuario=id_usuario,
                id_cargas=id_cargas,
                id_estado_proceso=id_estado_proceso,
                usumod=usumod
            )

            logger.info(f"Batch updated {len(id_cargas)} knowledge records to state {id_estado_proceso}")
            return message_result

        except Exception as e:
            logger.error(f"Error batch updating knowledge process state: {e}")
            raise

    async def batch_delete_knowledge(
        self,
        id_usuario: int,
        id_cargas: List[int],
        usumod: str = "System"
    ) -> Dict[str, Any]:
        """
        Delete multiple knowledge/document records in batch.
        - Logical deletion in SQL Server (ID_ESTADO_REGISTRO = 0)
        - Physical deletion in Weaviate (complete removal of all related objects)

        Args:
            id_usuario: User ID who is deleting (for audit purposes)
            id_cargas: List of knowledge/document IDs to delete
            usumod: User who deleted the records (max 200 chars, default: "System")

        Returns:
            Dictionary with:
            - message_result: Status message from the stored procedure
            - deleted_count: Number of records deleted from SQL Server
            - weaviate_result: Result of Weaviate deletion (success, deleted_count, errors)
        """
        try:
            # Phase 1: Logical deletion in SQL Server
            logger.info(f"Starting batch deletion of {len(id_cargas)} knowledge records")
            db_result = self.repository.batch_delete_knowledge(
                id_usuario=id_usuario,
                id_cargas=id_cargas,
                usumod=usumod
            )

            if not db_result:
                logger.error("Failed to delete knowledge records from SQL Server")
                raise ValueError("Database deletion failed")

            message_result = db_result.get('message_result')
            deleted_records = db_result.get('deleted_records', [])

            logger.info(f"Logically deleted {len(deleted_records)} records from SQL Server")

            # Phase 2: Physical deletion from Weaviate
            # Group deleted records by company to delete from appropriate collections
            weaviate_results = []

            if deleted_records:
                # Import here to avoid circular dependency
                from app.core.container import container
                vectorstore = container.get_vectorstore()

                # Group records by company
                records_by_company = {}
                for record in deleted_records:
                    company_id = record.get('ID_EMPRESA')
                    id_carga = record.get('ID_CARGA')

                    if company_id and id_carga:
                        company_key = f"EMPR{company_id}"
                        if company_key not in records_by_company:
                            records_by_company[company_key] = []
                        records_by_company[company_key].append(str(id_carga))

                # Delete from Weaviate for each company collection
                for collection_name, doc_ids in records_by_company.items():
                    try:
                        # Check if collection exists first
                        collection_exists = await vectorstore.collection_exists(collection_name)

                        if not collection_exists:
                            logger.warning(f"Collection {collection_name} does not exist in Weaviate, skipping physical deletion")
                            weaviate_results.append({
                                "collection": collection_name,
                                "success": True,
                                "deleted_count": 0,
                                "message": "Collection does not exist (no action needed)"
                            })
                            continue

                        # Delete from Weaviate
                        logger.info(f"Deleting {len(doc_ids)} documents from Weaviate collection {collection_name}")
                        delete_result = await vectorstore.delete_by_doc_ids(
                            class_name=collection_name,
                            doc_ids=doc_ids,
                            company_id=collection_name.replace("EMPR", "")  # Extract company_id from collection name
                        )

                        weaviate_results.append({
                            "collection": collection_name,
                            **delete_result
                        })

                        logger.info(f"Weaviate deletion result for {collection_name}: {delete_result}")

                    except Exception as weaviate_error:
                        error_msg = f"Error deleting from Weaviate collection {collection_name}: {str(weaviate_error)}"
                        logger.error(error_msg)
                        weaviate_results.append({
                            "collection": collection_name,
                            "success": False,
                            "deleted_count": 0,
                            "errors": [error_msg]
                        })

            return {
                "message_result": message_result,
                "deleted_count": len(deleted_records),
                "deleted_records": deleted_records,
                "weaviate_result": weaviate_results
            }

        except Exception as e:
            logger.error(f"Error batch deleting knowledge: {e}")
            raise
