"""Service for handling upload knowledge operations."""

import uuid
import logging
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta
from app.core.config import settings
from app.core.aws_clients import get_s3_client
from app.infrastructure.repositories.upload_knowledge_repository import UploadKnowledgeRepository

logger = logging.getLogger(__name__)


class UploadKnowledgeService:
    """Service for generating presigned URLs and managing upload records."""

    def __init__(self):
        """Initialize service with repository and bucket name."""
        self.repository = UploadKnowledgeRepository()
        self.bucket_name = settings.s3_pdfs_bucket

    async def generate_presigned_urls(
        self,
        company_id: int,
        area_id: int,
        user_id: int,
        embedding_model: str,
        pdf_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate presigned URLs for PDF uploads and create DynamoDB records in batch.

        Args:
            company_id: Company identifier
            area_id: Area identifier
            user_id: User ID who is uploading
            embedding_model: Embedding model to use
            pdf_keys: List of PDF filenames

        Returns:
            List of objects containing presigned URLs and metadata

        Raises:
            Exception: If presigned URL generation or DynamoDB batch write fails
        """
        try:
            # Phase 1: Prepare all records and S3 keys (no DB writes yet)
            batch_records = []
            s3_keys_map = {}  # Map process_id to S3 key for later use

            for filename in pdf_keys:
                # Generate unique UUID for this PDF
                process_id = str(uuid.uuid4())

                # Construct S3 key: <process_id>/<filename>
                pdf_key = f"{process_id}/{filename}"

                # Store for later use
                s3_keys_map[process_id] = pdf_key

                # Add to batch records (not written to DB yet)
                batch_records.append({
                    'process_id': process_id,
                    'uploaded_by_id': user_id,
                    'company_id': company_id,
                    'area_id': area_id,
                    'embedding_model': embedding_model,
                    'pdf_key': pdf_key,
                    'process_stage': 0,  # UPLOAD stage
                    'is_error': False
                })

            # Phase 2: Single batch write to DynamoDB (all records at once, non-blocking)
            created_records = await self.repository.async_batch_create_upload_records(batch_records)

            # Phase 3: Generate presigned URLs and build responses (truly async with aioboto3)
            response_objects = await self._generate_presigned_urls_async(
                created_records,
                s3_keys_map
            )

            logger.info(f"Generated {len(response_objects)} presigned URLs in batch")
            return response_objects

        except Exception as e:
            logger.error(f"Error generating presigned URLs: {e}")
            raise

    async def _generate_presigned_urls_async(
        self,
        created_records: List[Dict[str, Any]],
        s3_keys_map: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Generate presigned URLs for all records using aioboto3 (truly async).

        Args:
            created_records: List of created DynamoDB records
            s3_keys_map: Map of process_id to S3 key

        Returns:
            List of response objects with presigned URLs
        """
        from app.core.aws_clients import get_s3_client

        response_objects = []

        # Use aioboto3 S3 client for async presigned URL generation
        async with get_s3_client() as s3_client:
            for record in created_records:
                process_id = record['id']
                pdf_key = s3_keys_map[process_id]

                # Generate presigned PUT URL (5 min expiration)
                # With aioboto3, generate_presigned_url returns a coroutine and must be awaited
                presigned_url = await s3_client.generate_presigned_url(
                    'put_object',
                    Params={
                        'Bucket': self.bucket_name,
                        'Key': pdf_key,
                        'ContentType': 'application/pdf'
                    },
                    ExpiresIn=300  # 5 min
                )

                # Build response object
                response_obj = {
                    'process_id': process_id,
                    'pdf_key': pdf_key,
                    'presigned_url': presigned_url,
                    'process_stage': record['process_stage'],
                    'is_text_based': True,  # Default assumption
                    'uploaded_by_id': record['uploaded_by_id'],
                    'company_id': record['company_id'],
                    'area_id': record['area_id'],
                    'embedding_model': record['embedding_model'],
                    'created_at': record['created_at']
                }

                response_objects.append(response_obj)

        return response_objects

    def get_upload_status(self, process_id: str) -> Dict[str, Any]:
        """
        Get the status of an upload process.

        Args:
            process_id: The UUID/process_id to query

        Returns:
            Upload record with current status
        """
        try:
            record = self.repository.get_upload_record(process_id)
            if not record:
                raise ValueError(f"No record found for process_id: {process_id}")

            return record

        except Exception as e:
            logger.error(f"Error getting upload status for {process_id}: {e}")
            raise

    def update_upload_stage(
        self,
        process_id: str,
        process_stage: int,
        is_error: bool = False
    ) -> bool:
        """
        Update the processing stage of an upload.

        Args:
            process_id: The UUID/process_id
            process_stage: New stage (0-7)
            is_error: Error flag

        Returns:
            True if successful
        """
        try:
            return self.repository.update_process_stage(
                process_id=process_id,
                process_stage=process_stage,
                is_error=is_error
            )

        except Exception as e:
            logger.error(f"Error updating stage for {process_id}: {e}")
            raise

    def get_user_uploads(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Get all uploads for a user.

        Args:
            user_id: User identifier

        Returns:
            List of upload records
        """
        try:
            return self.repository.get_uploads_by_user(user_id)

        except Exception as e:
            logger.error(f"Error getting uploads for user {user_id}: {e}")
            raise

    async def get_company_uploads(
        self,
        company_id: int,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all uploads for a company across all areas (async).

        Args:
            company_id: Company identifier
            limit: Maximum number of records to return (default: 100)

        Returns:
            List of upload records sorted by created_at (most recent first)
        """
        try:
            return await self.repository.async_get_uploads_by_company(company_id, limit)

        except Exception as e:
            logger.error(f"Error getting uploads for company {company_id}: {e}")
            raise

    async def get_company_area_uploads(
        self,
        company_id: int,
        area_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get all uploads for a company/area (async).

        Args:
            company_id: Company identifier
            area_id: Area identifier

        Returns:
            List of upload records
        """
        try:
            return await self.repository.async_get_uploads_by_company_area(company_id, area_id)

        except Exception as e:
            logger.error(f"Error getting uploads for company {company_id}, area {area_id}: {e}")
            raise
