"""Document processing request and response models."""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class TaskStatus(BaseModel):
    """Task status model for document processing."""
    task_id: str
    status: str  # pending, running, completed, failed
    company_name: str
    area_name: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    files_processed: int = 0
    total_files: int = 0


class UploadResponse(BaseModel):
    """Response model for file upload operations."""
    task_id: str
    message: str
    files_uploaded: int
    status: str


class ProcessingRequest(BaseModel):
    """Request model for document processing."""
    company_name: str
    area_name: str
