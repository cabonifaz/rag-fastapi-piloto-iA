"""Pydantic models for company management."""

from pydantic import BaseModel, Field
from typing import List, Optional


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


class GetCompaniesPaginatedRequest(BaseModel):
    """Request model for getting paginated companies"""
    num_pagina: int = 1
    tam_pagina: int = 10
    term_busqueda: Optional[str] = None
    campo_orden: Optional[str] = "RAZON_SOCIAL"
    dir_orden: Optional[str] = "ASC"
    filtro_estado: Optional[int] = None


class PaginationInfo(BaseModel):
    """Pagination information model."""
    total_records: int = Field(..., description="Total number of records")
    current_page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")


class CompanyPaginatedData(BaseModel):
    """Company data model for paginated response."""
    ID_EMPRESA: int
    RUC: str
    RAZON_SOCIAL: str
    FCHCRE: str
    LOGO: str | None
    ID_ESTADO_REGISTRO: int
    SECRET_KEY: str

    class Config:
        from_attributes = True


class PaginatedCompaniesResponse(BaseModel):
    """Response model for paginated companies endpoint."""
    data: List[CompanyPaginatedData]
    pagination: PaginationInfo