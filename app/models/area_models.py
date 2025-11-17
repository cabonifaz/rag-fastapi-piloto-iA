"""Pydantic models for area management."""

from pydantic import BaseModel, Field


class AreaCreateRequest(BaseModel):
    """Request model for creating an area."""
    id_empresa: int = Field(..., description="Company ID")
    area: str = Field(..., min_length=1, max_length=100, description="Area name")
