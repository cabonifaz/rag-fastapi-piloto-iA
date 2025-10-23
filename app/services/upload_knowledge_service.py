"""Service for handling upload knowledge operations."""

import uuid
import logging
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

    def generate_presigned_urls(
        self,
        company_id: int,
        area_id: int,
        user_id: int,
        embedding_model: str,
        pdf_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate presigned URLs for PDF uploads and create DynamoDB records.

        Args:
            company_id: Company identifier
            area_id: Area identifier
            user_id: User ID who is uploading
            embedding_model: Embedding model to use
            pdf_keys: List of PDF filenames

        Returns:
            List of objects containing presigned URLs and metadata

        Raises:
            Exception: If presigned URL generation or DynamoDB write fails
        """
        try:
            response_objects = []

            for filename in pdf_keys:
                # Generate unique UUID for this PDF
                process_id = str(uuid.uuid4())

                # Construct S3 key: <process_id>/<filename>
                pdf_key = f"{process_id}/{filename}"

                # Create DynamoDB record with process_stage = 0 (UPLOAD)
                record = self.repository.create_upload_record(
                    process_id=process_id,
                    uploaded_by_id=user_id,
                    company_id=company_id,
                    area_id=area_id,
                    embedding_model=embedding_model,
                    pdf_key=pdf_key,
                    process_stage=0,  # UPLOAD stage
                    is_error=False
                )

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
                    'process_stage': 0,
                    'is_text_based': True,  # Default assumption
                    'uploaded_by_id': user_id,
                    'company_id': company_id,
                    'area_id': area_id,
                    'embedding_model': embedding_model,
                    'created_at': record['created_at']
                }

                response_objects.append(response_obj)

                logger.info(f"Generated presigned URL for {filename} with process_id {process_id}")

            return response_objects

        except Exception as e:
            logger.error(f"Error generating presigned URLs: {e}")
            raise

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
