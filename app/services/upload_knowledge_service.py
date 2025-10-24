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
        """Initialize service with repository and S3 client."""
        self.repository = UploadKnowledgeRepository()
        self.s3_client = get_s3_client()
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

            # Phase 3: Generate presigned URLs and build responses (in thread pool to avoid blocking)
            loop = asyncio.get_event_loop()
            response_objects = await loop.run_in_executor(
                None,
                self._generate_presigned_urls_sync,
                created_records,
                s3_keys_map
            )

            logger.info(f"Generated {len(response_objects)} presigned URLs in batch")
            return response_objects

        except Exception as e:
            logger.error(f"Error generating presigned URLs: {e}")
            raise

    def _generate_presigned_urls_sync(
        self,
        created_records: List[Dict[str, Any]],
        s3_keys_map: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Synchronous helper to generate presigned URLs for all records.

        This is run in a thread pool from the async method.

        Args:
            created_records: List of created DynamoDB records
            s3_keys_map: Map of process_id to S3 key

        Returns:
            List of response objects with presigned URLs
        """
        response_objects = []
        for record in created_records:
            process_id = record['id']
            pdf_key = s3_keys_map[process_id]

            # Generate presigned PUT URL (5 min expiration)
            presigned_url = self.s3_client.generate_presigned_url(
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

    def get_company_area_uploads(
        self,
        company_id: int,
        area_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get all uploads for a company/area.

        Args:
            company_id: Company identifier
            area_id: Area identifier

        Returns:
            List of upload records
        """
        try:
            return self.repository.get_uploads_by_company_area(company_id, area_id)

        except Exception as e:
            logger.error(f"Error getting uploads for company {company_id}, area {area_id}: {e}")
            raise
