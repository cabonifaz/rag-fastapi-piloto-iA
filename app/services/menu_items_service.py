"""Service for managing menu items operations following hexagonal architecture."""

from typing import List, Dict, Any
import logging
from sqlalchemy.orm import Session
from app.infrastructure.repositories.menu_items_repository import MenuItemsRepository

logger = logging.getLogger(__name__)


class MenuItemsService:
    """
    Service for menu items operations.
    Handles business logic for retrieving menu items based on user role.
    """

    def __init__(self):
        """Initialize stateless MenuItemsService - no db parameter."""
        pass

    async def get_menu_items(
        self,
        db: Session,
        id_usuario: int
    ) -> List[Dict[str, Any]]:
        """
        Get menu items for a user based on their role.
        
        Uses SP_MENU_ITEMS_LST which:
        1. Gets user's role via SP_ROL_LST_BY_ID_USUARIO
        2. Filters menu items from PARAMETROS (ID_MAESTRO=9)
        3. Returns items where user_role <= NUM3 (role permission)

        Args:
            db: Database session
            id_usuario: User ID

        Returns:
            List of dictionaries containing:
            - NUM1: Item ID
            - NUM2: Display order
            - NUM3: Max role allowed
            - PATH: Route path
            - LABEL: Display label
            - ICON: Icon name
            Empty list if fetch failed or user has no permissions
        """
        try:
            # Validate input
            if id_usuario <= 0:
                logger.error("Invalid user ID")
                return []

            # Create repository for this request
            repository = MenuItemsRepository(db)

            # Get menu items from repository
            menu_items = repository.get_menu_items(id_usuario=id_usuario)

            if menu_items:
                logger.info(f"Retrieved {len(menu_items)} menu items for user {id_usuario}")
            else:
                logger.warning(f"No menu items found for user {id_usuario}")

            return menu_items

        except Exception as e:
            logger.error(f"Error in get_menu_items service: {e}")
            raise