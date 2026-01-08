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
    """
    Dependency injection: Get singleton MenuItemsService from container.
    
    Returns:
        MenuItemsService: Singleton instance from IoC container
    """
    return container.get_menu_items_service()


@router.get("/get_menu_items")
async def get_menu_items_endpoint(
    menu_items_service: MenuItemsService = Depends(get_menu_items_service),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get menu items for the authenticated user.
    
    This endpoint retrieves menu items based on the user's:
    - Role (ID_TIPO_ROL) obtained via SP_ROL_LST_BY_ID_USUARIO
    - Company affiliation (ID_EMPRESA) from USUARIOS table
    - Assigned functionalities from ROL_FUNCIONALIDAD table
    
    The system uses a relationship-based approach through PARAMETROS table:
    - Maestro 9: Menu items (modules)
    - Maestro 10: Functionalities linked to menu items via NUM3 = NUM1
    - ROL_FUNCIONALIDAD: Permissions table linking roles to functionalities
    
    This design eliminates hardcoded CASE statements and allows dynamic menu
    generation without modifying stored procedures when adding new routes.
    
    Authentication:
        Requires valid JWT token in Authorization header (Bearer token).
        Token must contain 'ID_USUARIO' claim.
    
    Returns:
        JSON response with structure:
        {
            "menu_items": [
                {
                    "NUM1": 1,           // Item logical ID (business identifier)
                    "NUM2": 1,           // Display order
                    "PATH": "/n/rag/chat",  // Route path
                    "LABEL": "Chat",     // Display label
                    "ICON": "MessageSquare"  // Lucide React icon name
                },
                ...
            ],
            "result": {
                "success": true,
                "message": "Menu items obtenidos exitosamente",
                "timestamp": "2026-01-08T10:30:00Z"
            }
        }
        
    Raises:
        HTTPException 400: If user information is incomplete in JWT token
        HTTPException 401: If JWT token is invalid or expired (handled by get_current_user)
        HTTPException 500: If internal server error occurs
        
    Status Codes:
        200: Success - Menu items retrieved successfully
        400: Bad Request - Invalid or incomplete user information
        401: Unauthorized - Invalid or missing JWT token
        500: Internal Server Error - Unexpected error during processing
        
    Example:
```bash
        curl -X GET "http://localhost:8000/api/menu/get_menu_items" \
             -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```
        
    Notes:
        - Empty menu_items array indicates user has no assigned functionalities
        - Menu items are filtered by role and company automatically
        - Results are ordered by NUM2 (display order)
        - Only active items (ID_ESTADO_REGISTRO = 1) are returned
    """
    try:
        # Extract user ID from JWT token
        user_id = current_user.get('ID_USUARIO')
        
        # Validate user ID exists in token
        if not user_id:
            logger.warning(
                f"Incomplete user information in JWT token. "
                f"Token payload: {current_user}"
            )
            error_response = create_error_response(
                "Informacion de usuario incompleta en el token"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"result": error_response.model_dump()}
            )
        
        logger.info(f"Processing get_menu_items request for user {user_id}")
        
        # Get menu items using service layer
        menu_items = await menu_items_service.get_menu_items(
            db=db,
            id_usuario=user_id
        )
        
        # Create success response
        success_response = create_success_response("Menu items obtenidos exitosamente")
        
        logger.info(
            f"Successfully processed get_menu_items request for user {user_id}. "
            f"Returned {len(menu_items)} items"
        )
        
        return {
            "menu_items": menu_items,
            "result": success_response.model_dump()
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is (already logged above)
        raise
        
    except Exception as e:
        # Log unexpected errors with full stack trace
        logger.error(
            f"Unexpected error in get_menu_items endpoint for user {current_user.get('ID_USUARIO')}: {e}",
            exc_info=True
        )
        
        # Return generic error to client (don't expose internal details)
        error_response = create_error_response("Error interno del servidor")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"result": error_response.model_dump()}
        )