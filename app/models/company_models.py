"""Pydantic models for company management."""

from pydantic import BaseModel, Field


class CompanyCreateRequest(BaseModel):
    """Request model for creating a company."""
    ruc: str = Field(..., min_length=1, max_length=30, description="Company RUC identifier")
    razon_social: str = Field(..., min_length=1, max_length=255, description="Company name")
