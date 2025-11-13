"""API endpoints for company management."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from pydantic import BaseModel, Field
import logging

from app.core.database import get_db
from app.core.container import container
from app.services.company_service import CompanyService
from app.models.response_models import create_success_response, create_error_response
from app.utils.jwt_auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


# Request Models
class CompanyCreateRequest(BaseModel):
    """Request model for creating a company."""
    ruc: str = Field(..., min_length=1, max_length=30, description="Company RUC identifier")
    razon_social: str = Field(..., min_length=1, max_length=255, description="Company name")

    class Config:
        json_schema_extra = {
            "example": {
                "ruc": "20123456789",
                "razon_social": "Empresa Demo SAC"
            }
        }


def get_company_service() -> CompanyService:
    """Get singleton CompanyService from container."""
    return container.get_company_service()


@router.post("/create_company")
async def create_company_endpoint(
    request: CompanyCreateRequest,
    company_service: CompanyService = Depends(get_company_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a new company base endpoint.

    Creates a new company with associated roles and areas using SP_CREATE_EMPRESA_BASE.
    Requires JWT authentication. Uses user ID from JWT token.

    Args:
        request: CompanyCreateRequest with ruc and razon_social

    Returns:
        Dict with company creation results including:
        - results: List of dictionaries with id_rol, message, id_company, id_area
        - result: Success/error response

    Raises:
        HTTPException: 400 for validation errors, 500 for server errors
    """
    try:
        user_id = current_user.get('ID_USUARIO')

        if not user_id:
            error_response = create_error_response("Informacion de usuario incompleta en el token")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )

        # Create company using service
        results = await company_service.create_company(
            db=db,
            id_usuario=user_id,
            ruc=request.ruc,
            razon_social=request.razon_social
        )

        if not results:
            error_response = create_error_response("Error al crear la empresa")
            raise HTTPException(
                status_code=500,
                detail={"result": error_response.model_dump()}
            )

        success_response = create_success_response("Empresa creada exitosamente")
        return {
            "results": results,
            "result": success_response.model_dump()
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error in create_company endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )
