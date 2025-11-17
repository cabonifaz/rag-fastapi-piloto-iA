"""API endpoints for upload knowledge operations."""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any
import logging
from app.services.upload_knowledge_service import UploadKnowledgeService
from app.models.response_models import create_success_response, create_error_response
from app.models.upload_models import PresignedUrlRequest, PresignedUrlResponse
from app.utils.jwt_auth import get_current_user_with_company_area_validation

logger = logging.getLogger(__name__)

router = APIRouter()


def get_upload_knowledge_service() -> UploadKnowledgeService:
    """Get UploadKnowledgeService instance."""
    return UploadKnowledgeService()


@router.post("/get_presigned_urls", response_model=List[PresignedUrlResponse])
async def get_presigned_urls_endpoint(
    request: PresignedUrlRequest,
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_area_validation),
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
        response_objects = await service.generate_presigned_urls(
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


@router.post("/get_company_uploads")
async def get_company_uploads_endpoint(
    request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user_with_company_area_validation),
    service: UploadKnowledgeService = Depends(get_upload_knowledge_service)
):
    """
    Get all uploads for a company across all areas.

    Args:
        request: JSON body with company_id and optional limit

    Returns:
        List of upload records sorted by created_at (most recent first)
    """
    try:
        company_id = request.get("company_id")
        area_id = request.get("area_id")
        limit = request.get("limit", 100)

        # Validate inputs
        if not company_id:
            error_response = create_error_response("company_id is required")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )

        if not area_id:
            error_response = create_error_response("area_id is required")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )

        # Validate limit range
        if limit < 1 or limit > 500:
            error_response = create_error_response("limit must be between 1 and 500")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )

        # Get uploads for the company (access validation handled by dependency)
        uploads = await service.get_company_uploads(company_id=company_id, limit=limit)

        logger.info(f"Retrieved {len(uploads)} uploads for company {company_id}")

        return uploads

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error getting uploads for company: {e}")
        error_response = create_error_response("Error retrieving company uploads")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )
