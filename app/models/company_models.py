"""Pydantic models for company management."""

from pydantic import BaseModel, Field


class CompanyCreateRequest(BaseModel):
    """Request model for creating a company."""
    ruc: str = Field(..., min_length=1, max_length=30, description="Company RUC identifier")
    razon_social: str = Field(..., min_length=1, max_length=255, description="Company name")


class CompanyStatusUpdateRequest(BaseModel):
    """Request model for updating company status."""
    id_empresa: int = Field(..., gt=0, description="Company ID")
    status: int = Field(..., description="Status value (0=inactive, 1=active)")


class CompanyLogoUploadRequest(BaseModel):
    """Request model for generating presigned URL for logo upload."""
    id_empresa: int = Field(..., gt=0, description="Company ID")
    logo_filename: str = Field(..., min_length=1, max_length=255, description="Logo filename (e.g., 'logo.png')")
