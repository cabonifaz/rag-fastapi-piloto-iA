"""Pydantic models for IA models management."""

from pydantic import BaseModel, Field


class ModelCreateRequest(BaseModel):
    """Request model for creating an IA model."""
    model: str = Field(..., min_length=1, max_length=200, description="Model name/display name")
    id_model: str = Field(..., min_length=1, max_length=200, description="Model identifier")
    provider: str = Field(..., min_length=1, max_length=200, description="Provider name")
    type_id: int = Field(..., gt=0, description="Model type ID")
    extra_parameter: int = Field(..., ge=0, description="Type-specific parameter")
