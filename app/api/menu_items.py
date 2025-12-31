"""API endpoints for menu items management."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import logging

from app.core.database import get_db
from app.core.container import container
from app.services.menu_items_service import MenuItemsService
from app.models.response_models import create_success_response, create_error_response
from app.utils.jwt_auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


def get_menu_items_service() -> MenuItemsService:
    """Get singleton MenuItemsService from container."""
    return container.get_menu_items_service()


@router.get("/get_menu_items")
async def get_menu_items_endpoint(
    menu_items_service: MenuItemsService = Depends(get_menu_items_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get menu items for the current user endpoint.
    
    Retrieves menu items based on user's role using SP_MENU_ITEMS_LST.
    Requires JWT authentication.
    
    Returns:
        Dict with:
        - menu_items: List of menu item dictionaries with:
            - NUM1: Item ID
            - NUM2: Display order  
            - NUM3: Max role allowed
            - PATH: Route path
            - LABEL: Display label
            - ICON: Icon name
        - result: Success/error response
        
    Raises:
        HTTPException: 400 for validation errors, 401 for auth errors, 500 for server errors
    """
    try:
        user_id = current_user.get('ID_USUARIO')
        
        if not user_id:
            error_response = create_error_response("Informacion de usuario incompleta en el token")
            raise HTTPException(
                status_code=400,
                detail={"result": error_response.model_dump()}
            )
        
        # Get menu items using service
        menu_items = await menu_items_service.get_menu_items(
            db=db,
            id_usuario=user_id
        )
        
        success_response = create_success_response("Menu items obtenidos exitosamente")
        return {
            "menu_items": menu_items,
            "result": success_response.model_dump()
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in get_menu_items endpoint: {e}")
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=500,
            detail={"result": error_response.model_dump()}
        )