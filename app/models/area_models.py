"""Pydantic models for area management."""

from pydantic import BaseModel, Field


class AreaCreateRequest(BaseModel):
    """Request model for creating an area."""
    id_empresa: int = Field(..., description="Company ID")
    area: str = Field(..., min_length=1, max_length=100, description="Area name")


class AreaUpdateStatusRequest(BaseModel):
    """Request model for updating area status."""
    id_empresa: int = Field(..., description="Company ID")
    id_area: int = Field(..., description="Area ID")
    status: int = Field(..., ge=0, le=1, description="Status (0=inactive, 1=active)")


class AreaUpdateNameRequest(BaseModel):
    """Request model for updating area name."""
    id_empresa: int = Field(..., description="Company ID")
    id_area: int = Field(..., description="Area ID")
    area: str = Field(..., min_length=1, max_length=200, description="New area name")
