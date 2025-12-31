"""Repository for menu items database operations."""

from sqlalchemy.orm import Session
from typing import List, Dict, Any
import logging
from app.core.database import retry_on_db_error

logger = logging.getLogger(__name__)


class MenuItemsRepository:
    """
    Repository for menu items operations.
    
    Uses SP_MENU_ITEMS_LST to retrieve menu items based on user role.
    """

    def __init__(self, db: Session):
        self.db = db

    @retry_on_db_error(max_retries=3, delay=1)
    def get_menu_items(self, id_usuario: int) -> List[Dict[str, Any]]:
        """
        Get menu items for a user using stored procedure SP_MENU_ITEMS_LST.
        
        The SP internally calls SP_ROL_LST_BY_ID_USUARIO to get user's role,
        then filters menu items from PARAMETROS table where ID_MAESTRO = 9
        based on role permissions (NUM3 >= user_role).

        Args:
            id_usuario: User ID

        Returns:
            List of dictionaries with menu item data:
            - NUM1: Item ID
            - NUM2: Display order
            - NUM3: Max role allowed
            - PATH: Route path (STRING1)
            - LABEL: Display label (STRING2)
            - ICON: Icon name (STRING3)
            Empty list if fetch failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_MENU_ITEMS_LST @ID_USUARIO = ?",
                    id_usuario
                )

                results = []

                # Iterate through all result sets
                while True:
                    try:
                        # Check if we have columns (indicating data)
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            rows = cursor.fetchall()

                            if rows:
                                # Convert rows to dictionaries
                                for row in rows:
                                    result_dict = dict(zip(columns, row))
                                    # Convert Decimal to int/float for numeric fields
                                    numeric_fields = ['NUM1', 'NUM2', 'NUM3']
                                    for field in numeric_fields:
                                        if field in result_dict and result_dict[field] is not None:
                                            # Keep as float for decimals like 1.000000
                                            result_dict[field] = float(result_dict[field])
                                    results.append(result_dict)

                    except Exception as fetch_error:
                        logger.error(f"Fetch error in get_menu_items: {fetch_error}")

                    # Move to next result set
                    try:
                        if not cursor.nextset():
                            break
                    except Exception as nextset_error:
                        # Transaction error is expected when SP manages its own transactions
                        if "Transaction count after EXECUTE" in str(nextset_error):
                            logger.debug(f"SP manages its own transactions (expected): {nextset_error}")
                        else:
                            logger.error(f"Nextset error in get_menu_items: {nextset_error}")
                        break

                cursor.close()
                self.db.commit()
                return results

            except Exception as cursor_error:
                logger.error(f"Cursor error in get_menu_items: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error fetching menu items with SP: {e}")
            self.db.rollback()
            return []