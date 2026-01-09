"""Service for managing menu items operations following hexagonal architecture."""

from typing import List, Dict, Any
import logging
from sqlalchemy.orm import Session
from app.infrastructure.repositories.menu_items_repository import MenuItemsRepository

logger = logging.getLogger(__name__)


class MenuItemsService:
    """
    Service for menu items operations.
    
    Handles business logic for retrieving menu items based on user role,
    company affiliation, and assigned functionalities through ROL_FUNCIONALIDAD table.
    
    This service layer sits between the API endpoints and the data access layer,
    providing validation, business rules, and error handling.
    
    Architecture:
    - Stateless design (no instance variables)
    - Receives DB session per request
    - Delegates data access to MenuItemsRepository
    - Focuses on business logic and validation
    """

    def __init__(self):
        """
        Initialize stateless MenuItemsService.
        
        No database session is stored at initialization to maintain stateless design.
        Session is passed per request to ensure proper connection management.
        """
        pass

    async def get_menu_items(
        self,
        db: Session,
        id_usuario: int
    ) -> List[Dict[str, Any]]:
        """
        Get menu items for a user based on their role, company, and assigned functionalities.
        
        Business logic flow:
        1. Validates user ID is positive integer
        2. Creates repository instance for this request
        3. Retrieves menu items via SP_MENU_ITEMS_LST_BY_ROL_FUNC which:
           - Gets user's role via SP_ROL_LST_BY_ID_USUARIO
           - Gets user's company from USUARIOS table
           - Performs JOIN between:
             * PARAMETROS (Maestro 9 - menu items)
             * PARAMETROS (Maestro 10 - functionalities) via NUM3 = NUM1
             * ROL_FUNCIONALIDAD via ID_FUNCIONALIDAD = NUM1
           - Filters by exact role match, company match, and active status
           - Returns ordered items by NUM2 (display order)
        4. Logs result for monitoring and debugging
        
        The SP uses logical business identifiers (NUM1, NUM3) instead of technical
        auto-incremental IDs, ensuring the system is scalable when adding new modules
        without modifying stored procedure logic (no hardcoded CASE statements).

        Args:
            db: SQLAlchemy database session for this request
            id_usuario: User ID to retrieve menu items for (must be positive)

        Returns:
            List of dictionaries containing menu item data:
            - NUM1 (int): Item logical ID
            - NUM2 (int): Display order
            - PATH (str): Route path
            - LABEL (str): Display label
            - ICON (str): Lucide React icon name
            
            Returns empty list if:
            - User ID is invalid (≤ 0)
            - User has no assigned functionalities in ROL_FUNCIONALIDAD
            - User's role or company is not found
            - Database operation fails
            
        Raises:
            Exception: If critical error occurs during retrieval (after retries)
            
        Example:
            >>> service = MenuItemsService()
            >>> items = await service.get_menu_items(db, id_usuario=5)
            >>> print(items)
            [
                {
                    'NUM1': 1,
                    'NUM2': 1,
                    'PATH': '/n/rag/chat',
                    'LABEL': 'Chat',
                    'ICON': 'MessageSquare'
                },
                ...
            ]
        """
        try:
            # Validate user ID is positive
            if id_usuario <= 0:
                logger.error(
                    f"Invalid user ID provided: {id_usuario}. Must be positive integer."
                )
                return []

            # Create repository instance for this request
            # Following hexagonal architecture: service depends on repository interface
            repository = MenuItemsRepository(db)

            # Retrieve menu items from database via stored procedure
            menu_items = repository.get_menu_items(id_usuario=id_usuario)

            # Log result for monitoring and debugging
            if menu_items:
                # Extract labels for human-readable logging
                item_labels = [item.get('LABEL', 'Unknown') for item in menu_items]
                logger.info(
                    f"Retrieved {len(menu_items)} menu items for user {id_usuario}: {item_labels}"
                )
            else:
                logger.warning(
                    f"No menu items found for user {id_usuario}. "
                    f"Possible causes: "
                    f"(1) User has no entries in ROL_FUNCIONALIDAD table, "
                    f"(2) User's role has no assigned functionalities, "
                    f"(3) User's company does not match assigned functionalities, "
                    f"(4) All assigned functionalities are inactive (ID_ESTADO_REGISTRO = 0)"
                )

            return menu_items

        except Exception as e:
            logger.error(
                f"Error in get_menu_items service for user {id_usuario}: {e}",
                exc_info=True
            )
            raise