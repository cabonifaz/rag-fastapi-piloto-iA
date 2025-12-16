"""Service for handling knowledge/document operations."""

import logging
import asyncio
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
        - First queries SQL Server to verify existence and get company_id
        - Validates existence in Weaviate (throws error if not found)
        - Logical deletion in SQL Server (ID_ESTADO_REGISTRO = 0)
        - Physical deletion in Weaviate (complete removal of all related objects)

        Args:
            id_usuario: User ID who is deleting (for audit and permission validation)
            id_cargas: List of knowledge/document IDs to delete
            usumod: User who deleted the records (max 200 chars, default: "System")

        Returns:
            Dictionary with:
            - message_result: Status message from the stored procedure
            - deleted_count: Number of records deleted from SQL Server
            - weaviate_result: Result of Weaviate deletion (success, deleted_count, errors)

        Raises:
            ValueError: If documents are not found in SQL Server or Weaviate, or if user has insufficient permissions
        """
        try:
            logger.info(f"Starting batch deletion of {len(id_cargas)} knowledge records")

            # PHASE 1: Query SQL Server to verify existence and get company_id
            logger.info(f"Querying SQL Server to verify existence and get company information")

            sql_result = self.repository.get_knowledge_by_ids(
                id_usuario=id_usuario,
                id_cargas=id_cargas
            )

            if not sql_result:
                error_msg = "Error al consultar documentos en la base de datos"
                logger.error(error_msg)
                raise ValueError(error_msg)

            message_result_check = sql_result.get('message_result')
            records = sql_result.get('records', [])

            # Check if the SP returned an error message (permission denied, user not found, etc.)
            if message_result_check and message_result_check.get('ID_TIPO_MENSAJE') == 1:
                error_msg = message_result_check.get('MENSAJE', 'Error de permisos')
                logger.error(f"Permission or validation error: {error_msg}")
                raise ValueError(error_msg)

            # Check if we got any records
            if not records:
                error_msg = "Los documentos especificados no existen o ya fueron eliminados"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Extract company_id from the first record (all records should have the same company)
            company_id = records[0].get('ID_EMPRESA')
            if not company_id:
                error_msg = "No se pudo obtener el ID de empresa de los documentos"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"Found {len(records)} records in SQL Server for company {company_id}")

            # PHASE 2: Validate existence in Weaviate BEFORE deleting from SQL Server
            logger.info(f"Validating existence of documents in Weaviate (collection: EMPR{company_id})")

            # Import here to avoid circular dependency
            from app.core.container import container
            vectorstore = container.get_vectorstore()

            collection_name = f"EMPR{company_id}"

            # Check if collection exists
            collection_exists = await vectorstore.collection_exists(collection_name)

            if not collection_exists:
                error_msg = f"La colección {collection_name} no existe en la base de datos vectorial"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Check if each document exists in Weaviate
            not_found_in_weaviate = []
            from weaviate.classes.query import Filter

            try:
                collection = vectorstore._client.collections.get(collection_name)

                for id_carga in id_cargas:
                    doc_id = f"CONOC-{id_carga}"

                    # Query to check if exists
                    check_query = collection.query.fetch_objects(
                        filters=Filter.by_property("doc_id").equal(doc_id),
                        limit=1
                    )

                    if len(check_query.objects) == 0:
                        logger.warning(f"Document {doc_id} not found in {collection_name}")
                        not_found_in_weaviate.append(f"ID: {id_carga}")

            except Exception as check_error:
                logger.error(f"Error checking existence in {collection_name}: {check_error}")
                raise ValueError(f"Error al verificar existencia de documentos en la base de datos vectorial: {str(check_error)}")

            # If any documents not found, raise error and don't proceed with deletion
            if not_found_in_weaviate:
                error_msg = f"Los siguientes documentos no se encontraron en la base de datos vectorial: {', '.join(not_found_in_weaviate)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"All {len(id_cargas)} documents found in Weaviate. Proceeding with deletion.")

            # PHASE 3: Logical deletion in SQL Server (only if validation passed)
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

            # PHASE 4: Physical deletion from Weaviate with retry and compensation
            # Group deleted records by company to delete from appropriate collections
            weaviate_results = []
            any_weaviate_failure = False

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

                # Delete from Weaviate for each company collection (with retry)
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

                        # RETRY MECHANISM: Attempt deletion with exponential backoff (max 3 attempts)
                        max_retries = 3
                        delete_result = None
                        last_error = None

                        for attempt in range(max_retries):
                            try:
                                logger.info(f"Deleting {len(doc_ids)} documents from Weaviate collection {collection_name} (attempt {attempt + 1}/{max_retries})")

                                delete_result = await vectorstore.delete_by_doc_ids(
                                    class_name=collection_name,
                                    doc_ids=doc_ids,
                                    company_id=None  # Collection already scoped by company
                                )

                                # Check if deletion was successful
                                if delete_result.get('success', False):
                                    logger.info(f"Weaviate deletion successful for {collection_name}: {delete_result['deleted_count']} objects deleted")
                                    break  # Success - exit retry loop
                                else:
                                    last_error = f"Deletion failed: {delete_result.get('errors', [])}"
                                    logger.warning(f"Attempt {attempt + 1} failed for {collection_name}: {last_error}")

                                    # If not last attempt, wait before retry (exponential backoff)
                                    if attempt < max_retries - 1:
                                        wait_time = 2 ** attempt  # 1s, 2s, 4s
                                        logger.info(f"Retrying in {wait_time} seconds...")
                                        await asyncio.sleep(wait_time)

                            except TimeoutError as timeout_error:
                                last_error = f"Timeout on attempt {attempt + 1}: {str(timeout_error)}"
                                logger.error(last_error)

                                if attempt < max_retries - 1:
                                    wait_time = 2 ** attempt
                                    logger.info(f"Retrying after timeout in {wait_time} seconds...")
                                    await asyncio.sleep(wait_time)
                                else:
                                    delete_result = {
                                        "success": False,
                                        "deleted_count": 0,
                                        "errors": [last_error]
                                    }

                            except Exception as retry_error:
                                last_error = f"Error on attempt {attempt + 1}: {str(retry_error)}"
                                logger.error(last_error)

                                if attempt < max_retries - 1:
                                    wait_time = 2 ** attempt
                                    logger.info(f"Retrying after error in {wait_time} seconds...")
                                    await asyncio.sleep(wait_time)
                                else:
                                    delete_result = {
                                        "success": False,
                                        "deleted_count": 0,
                                        "errors": [last_error]
                                    }

                        # Record result
                        weaviate_results.append({
                            "collection": collection_name,
                            **delete_result
                        })

                        # Check for failures
                        if not delete_result.get('success', False):
                            any_weaviate_failure = True
                            logger.error(f"All retry attempts failed for {collection_name}: {delete_result.get('errors', [])}")

                        if delete_result.get('not_found'):
                            logger.warning(f"Documents not found in {collection_name}: {delete_result['not_found']}")

                    except Exception as weaviate_error:
                        error_msg = f"Unexpected error deleting from Weaviate collection {collection_name}: {str(weaviate_error)}"
                        logger.error(error_msg)
                        any_weaviate_failure = True
                        weaviate_results.append({
                            "collection": collection_name,
                            "success": False,
                            "deleted_count": 0,
                            "errors": [error_msg]
                        })

            # COMPENSATION LOGIC: If any Weaviate deletion failed, rollback SQL changes
            if any_weaviate_failure and deleted_records:
                logger.warning(f"Weaviate deletion failed - initiating SQL rollback for {len(id_cargas)} records")

                try:
                    # Restore SQL records (undo soft delete)
                    restore_result = self.repository.batch_restore_knowledge(
                        id_usuario=id_usuario,
                        id_cargas=id_cargas,
                        usumod=f"{usumod} (auto-rollback)"
                    )

                    if restore_result and restore_result.get('restored_records'):
                        logger.info(f"Successfully restored {len(restore_result['restored_records'])} records in SQL (compensation)")

                        # Add rollback info to response
                        return {
                            "message_result": {
                                "ID_TIPO_MENSAJE": 1,  # Error
                                "MENSAJE": "Eliminación de Weaviate falló. Los registros fueron restaurados automáticamente en SQL."
                            },
                            "deleted_count": 0,  # No net deletion due to rollback
                            "deleted_records": [],
                            "restored_records": restore_result.get('restored_records', []),
                            "weaviate_result": weaviate_results,
                            "rollback_performed": True
                        }
                    else:
                        logger.error("Failed to restore SQL records after Weaviate failure")
                        raise ValueError("Weaviate deletion failed and SQL rollback also failed - data may be inconsistent")

                except Exception as rollback_error:
                    logger.error(f"CRITICAL: Failed to rollback SQL after Weaviate failure: {rollback_error}")
                    raise ValueError(f"Weaviate deletion failed and SQL rollback failed: {str(rollback_error)}. Manual intervention required.")

            return {
                "message_result": message_result,
                "deleted_count": len(deleted_records),
                "deleted_records": deleted_records,
                "weaviate_result": weaviate_results,
                "rollback_performed": False
            }

        except Exception as e:
            logger.error(f"Error batch deleting knowledge: {e}")
            raise
