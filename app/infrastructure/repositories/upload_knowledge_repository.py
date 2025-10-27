"""Repository for DynamoDB upload knowledge operations."""

from boto3.dynamodb.conditions import Key
from typing import Optional, List, Dict, Any
import logging
import asyncio
from app.core.config import settings
from app.core.aws_clients import get_dynamodb_table
from datetime import datetime, UTC

logger = logging.getLogger(__name__)


class UploadKnowledgeRepository:
    """
    Repository for DynamoDB UPLOAD_KNOWLEDGE table operations.

    Table structure:
    - id (Partition Key): STRING (UUID/process_id)
    - uploaded_by_id: NUMBER
    - company_id: NUMBER
    - area_id: NUMBER
    - process_stage: NUMBER (0-7)
    - is_error: BOOLEAN
    - embedding_model: STRING
    - pdf_key: STRING (S3 key: <process_id>/<filename>)
    - created_at: STRING (ISO timestamp)
    """

    def __init__(self):
        """Initialize DynamoDB table using centralized AWS client"""
        self.table = get_dynamodb_table(settings.dynamodb_table_upload_knowledge)
        self.table_name = settings.dynamodb_table_upload_knowledge

    def create_upload_record(
        self,
        process_id: str,
        uploaded_by_id: int,
        company_id: int,
        area_id: int,
        embedding_model: str,
        pdf_key: str,
        process_stage: int = 0,
        is_error: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new upload record in DynamoDB.

        Args:
            process_id: Unique UUID for this upload process
            uploaded_by_id: User ID who uploaded
            company_id: Company identifier
            area_id: Area identifier
            embedding_model: Model to use for embeddings
            pdf_key: S3 key path (<process_id>/<filename>)
            process_stage: Current stage (default: 0 = UPLOAD)
            is_error: Error flag (default: False)

        Returns:
            Dictionary with the created record

        Raises:
            Exception: If record creation fails
        """
        try:
            created_at = datetime.now(UTC).isoformat()

            item = {
                'id': process_id,
                'uploaded_by_id': uploaded_by_id,
                'company_id': company_id,
                'area_id': area_id,
                'process_stage': process_stage,
                'is_error': is_error,
                'embedding_model': embedding_model,
                'pdf_key': pdf_key,
                'created_at': created_at
            }

            self.table.put_item(Item=item)

            logger.info(f"Created upload record for process_id: {process_id}")
            return item

        except Exception as e:
            logger.error(f"Error creating upload record for process_id {process_id}: {e}")
            raise

    def get_upload_record(self, process_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an upload record by process_id.

        Args:
            process_id: The UUID/process_id to look up

        Returns:
            The upload record if found, None otherwise
        """
        try:
            response = self.table.get_item(Key={'id': process_id})
            return response.get('Item')

        except Exception as e:
            logger.error(f"Error getting upload record for process_id {process_id}: {e}")
            return None

    def update_process_stage(
        self,
        process_id: str,
        process_stage: int,
        is_error: bool = False
    ) -> bool:
        """
        Update the process stage for an upload record.

        Args:
            process_id: The UUID/process_id
            process_stage: New stage value (0-7)
            is_error: Error flag

        Returns:
            True if successful, False otherwise
        """
        try:
            self.table.update_item(
                Key={'id': process_id},
                UpdateExpression='SET process_stage = :stage, is_error = :error',
                ExpressionAttributeValues={
                    ':stage': process_stage,
                    ':error': is_error
                }
            )
            logger.info(f"Updated process_id {process_id} to stage {process_stage}")
            return True

        except Exception as e:
            logger.error(f"Error updating process stage for {process_id}: {e}")
            return False

    def get_uploads_by_user(
        self,
        uploaded_by_id: int,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get all uploads for a specific user.

        Args:
            uploaded_by_id: User ID
            limit: Maximum number of records to return

        Returns:
            List of upload records
        """
        try:
            response = self.table.scan(
                FilterExpression=Key('uploaded_by_id').eq(uploaded_by_id),
                Limit=limit
            )

            return response.get('Items', [])

        except Exception as e:
            logger.error(f"Error getting uploads for user {uploaded_by_id}: {e}")
            return []

    def get_uploads_by_company_area(
        self,
        company_id: int,
        area_id: int,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get all uploads for a specific company and area.

        Args:
            company_id: Company identifier
            area_id: Area identifier
            limit: Maximum number of records to return

        Returns:
            List of upload records
        """
        try:
            response = self.table.scan(
                FilterExpression=Key('company_id').eq(company_id) & Key('area_id').eq(area_id),
                Limit=limit
            )

            return response.get('Items', [])

        except Exception as e:
            logger.error(f"Error getting uploads for company {company_id}, area {area_id}: {e}")
            return []

    def batch_create_upload_records(
        self,
        records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create multiple upload records in DynamoDB using batch write.

        DynamoDB batch_write_item has a limit of 25 items per call,
        so this method handles batching automatically for larger lists.

        Args:
            records: List of record dictionaries, each containing:
                - process_id: Unique UUID for this upload process
                - uploaded_by_id: User ID who uploaded
                - company_id: Company identifier
                - area_id: Area identifier
                - embedding_model: Model to use for embeddings
                - pdf_key: S3 key path (<process_id>/<filename>)
                - process_stage: Current stage (default: 0 = UPLOAD)
                - is_error: Error flag (default: False)

        Returns:
            List of created records with timestamps

        Raises:
            Exception: If batch write fails
        """
        try:
            if not records:
                return []

            created_at = datetime.now(UTC).isoformat()
            processed_records = []

            # Process records and add timestamps
            for record in records:
                item = {
                    'id': record['process_id'],
                    'uploaded_by_id': record['uploaded_by_id'],
                    'company_id': record['company_id'],
                    'area_id': record['area_id'],
                    'process_stage': record.get('process_stage', 0),
                    'is_error': record.get('is_error', False),
                    'embedding_model': record['embedding_model'],
                    'pdf_key': record['pdf_key'],
                    'created_at': created_at
                }
                processed_records.append(item)

            # DynamoDB batch_write_item has a limit of 25 items per request
            # Split into batches of 25 and write each batch
            batch_size = 25
            for i in range(0, len(processed_records), batch_size):
                batch = processed_records[i:i + batch_size]

                with self.table.batch_writer(
                    overwrite_by_pkeys=['id']
                ) as batch_writer:
                    for item in batch:
                        batch_writer.put_item(Item=item)

            logger.info(f"Batch created {len(processed_records)} upload records")
            return processed_records

        except Exception as e:
            logger.error(f"Error batch creating upload records: {e}")
            raise

    async def async_batch_create_upload_records(
        self,
        records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create multiple upload records in DynamoDB using aioboto3 (truly async).

        Args:
            records: List of record dictionaries, each containing:
                - process_id: Unique UUID for this upload process
                - uploaded_by_id: User ID who uploaded
                - company_id: Company identifier
                - area_id: Area identifier
                - embedding_model: Model to use for embeddings
                - pdf_key: S3 key path (<process_id>/<filename>)
                - process_stage: Current stage (default: 0 = UPLOAD)
                - is_error: Error flag (default: False)

        Returns:
            List of created records with timestamps

        Raises:
            Exception: If batch write fails
        """
        try:
            if not records:
                return []

            from app.core.aws_clients import get_dynamodb_resource

            created_at = datetime.now(UTC).isoformat()
            processed_records = []

            # Process records and add timestamps
            for record in records:
                item = {
                    'id': record['process_id'],
                    'uploaded_by_id': record['uploaded_by_id'],
                    'company_id': record['company_id'],
                    'area_id': record['area_id'],
                    'process_stage': record.get('process_stage', 0),
                    'is_error': record.get('is_error', False),
                    'embedding_model': record['embedding_model'],
                    'pdf_key': record['pdf_key'],
                    'created_at': created_at
                }
                processed_records.append(item)

            # Use aioboto3 for true async DynamoDB operations
            async with get_dynamodb_resource() as dynamodb:
                table = await dynamodb.Table(self.table_name)

                # DynamoDB batch_write_item has a limit of 25 items per request
                # Split into batches of 25 and write each batch
                batch_size = 25
                for i in range(0, len(processed_records), batch_size):
                    batch = processed_records[i:i + batch_size]

                    async with table.batch_writer(
                        overwrite_by_pkeys=['id']
                    ) as batch_writer:
                        for item in batch:
                            await batch_writer.put_item(Item=item)

            logger.info(f"Batch created {len(processed_records)} upload records")
            return processed_records

        except Exception as e:
            logger.error(f"Error batch creating upload records: {e}")
            raise
