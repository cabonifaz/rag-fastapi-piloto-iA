"""API endpoints for upload knowledge operations."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging
from app.services.upload_knowledge_service import UploadKnowledgeService
from app.models.response_models import create_success_response, create_error_response
from app.utils.jwt_auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


def get_upload_knowledge_service() -> UploadKnowledgeService:
    """Get UploadKnowledgeService instance."""
    return UploadKnowledgeService()


class PresignedUrlRequest(BaseModel):
    """Request model for presigned URL generation."""
    company_id: int = Field(..., description="Company identifier")
    area_id: int = Field(..., description="Area identifier")
    user_id: int = Field(..., description="User identifier")
    embedding_model: str = Field(..., description="Embedding model to use (e.g., 'haiku3.5')")
    pdf_keys: List[str] = Field(..., description="List of PDF filenames to upload")

    class Config:
        json_schema_extra = {
            "example": {
                "company_id": 1,
                "area_id": 3,
                "user_id": 4,
                "embedding_model": "haiku3.5",
                "pdf_keys": ["norma_24.pdf", "norma_2.pdf", "condiciones.pdf"]
            }
        }


class PresignedUrlResponse(BaseModel):
    """Response model for presigned URL."""
    process_id: str
    pdf_key: str
    presigned_url: str
    process_stage: int
    is_text_based: bool
    uploaded_by_id: int
    company_id: int
    area_id: int
    embedding_model: str
    created_at: str


@router.post("/get_presigned_urls", response_model=List[PresignedUrlResponse])
async def get_presigned_urls_endpoint(
    request: PresignedUrlRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    service: UploadKnowledgeService = Depends(get_upload_knowledge_service)
):
    """
    Generate presigned URLs for PDF uploads.

    This endpoint:
    - Receives a list of PDFs to upload along with metadata
    - Generates a unique process_id for each PDF
    - Creates a DynamoDB record for each PDF with process_stage = 0 (UPLOAD)
    - Constructs the pdf_key in S3 using <process_id>/<filename>
    - Generates presigned PUT URLs for each PDF
    - Returns an array of objects containing all metadata and presigned URLs

    Args:
        request: PresignedUrlRequest with company_id, area_id, user_id, embedding_model, and pdf_keys

    Returns:
        List of objects with process_id, pdf_key, presigned_url, and metadata

    Raises:
        HTTPException: 400 for validation errors, 500 for server errors
    """
    try:
        # Validate input
        if not request.pdf_keys:
            error_response = create_error_response("No PDF files provided")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )

        # Validate filenames (no leading slashes)
        for filename in request.pdf_keys:
            if filename.startswith('/'):
                error_response = create_error_response(f"Invalid filename: {filename}. Filenames should not start with '/'")
                raise HTTPException(
                    status_code=400,
                    detail={"result": error_response.model_dump()}
                )

        # Generate presigned URLs and create DynamoDB records
        response_objects = service.generate_presigned_urls(
            company_id=request.company_id,
            area_id=request.area_id,
            user_id=request.user_id,
            embedding_model=request.embedding_model,
            pdf_keys=request.pdf_keys
        )

        logger.info(f"Generated {len(response_objects)} presigned URLs for user {request.user_id}")

        return response_objects

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in get_presigned_urls endpoint: {e}")
        error_response = create_error_response("Error generating presigned URLs")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.get("/upload_status/{process_id}")
async def get_upload_status_endpoint(
    process_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    service: UploadKnowledgeService = Depends(get_upload_knowledge_service)
):
    """
    Get the status of an upload process.

    Args:
        process_id: The UUID/process_id to query

    Returns:
        Upload record with current status

    Raises:
        HTTPException: 404 if not found, 500 for server errors
    """
    try:
        record = service.get_upload_status(process_id)

        if not record:
            error_response = create_error_response(f"Upload record not found for process_id: {process_id}")
            raise HTTPException(
                status_code=404,
                detail={"result": error_response.model_dump()}
            )

        success_response = create_success_response("Upload record retrieved successfully")
        return {
            "record": record,
            "result": success_response.model_dump()
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error getting upload status for {process_id}: {e}")
        error_response = create_error_response("Error retrieving upload status")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )


@router.get("/user_uploads/{user_id}")
async def get_user_uploads_endpoint(
    user_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
    service: UploadKnowledgeService = Depends(get_upload_knowledge_service)
):
    """
    Get all uploads for a specific user.

    Args:
        user_id: User identifier

    Returns:
        List of upload records

    Raises:
        HTTPException: 500 for server errors
    """
    try:
        uploads = service.get_user_uploads(user_id)

        success_response = create_success_response("User uploads retrieved successfully")
        return {
            "uploads": uploads,
            "count": len(uploads),
            "result": success_response.model_dump()
        }

    except Exception as e:
        logger.error(f"Error getting uploads for user {user_id}: {e}")
        error_response = create_error_response("Error retrieving user uploads")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )
