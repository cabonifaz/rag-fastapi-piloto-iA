"""Models for file-based audio transcription (OpenAI)."""

from pydantic import BaseModel
from typing import Optional


class TranscribeFileRequest(BaseModel):
    """Request model for file transcription metadata."""
    language_code: str = "es-ES"


class TranscribeFileResponse(BaseModel):
    """Response model for file transcription."""
    transcript: str
    language: str
    duration: float
    confidence: Optional[float] = None
    model: str
    file_size: int
