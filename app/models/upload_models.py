"""Models for upload knowledge operations."""

from pydantic import BaseModel, Field
from typing import List


class PresignedUrlRequest(BaseModel):
    """Request model for presigned URL generation."""
    company_id: int = Field(..., description="Company identifier")
    area_id: int = Field(..., description="Area identifier")
    user_id: int = Field(..., description="User identifier")
    embedding_model: str = Field(..., description="Embedding model to use (e.g., 'haiku3.5')")
    pdf_keys: List[str] = Field(..., description="List of PDF filenames to upload", examples=[["norma_24.pdf", "norma_2.pdf", "condiciones.pdf"]])


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
