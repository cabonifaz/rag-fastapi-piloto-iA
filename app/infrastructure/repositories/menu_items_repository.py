"""Repository for menu items database operations."""

from sqlalchemy.orm import Session
from typing import List, Dict, Any, Set
import logging
from app.core.database import retry_on_db_error

logger = logging.getLogger(__name__)


class MenuItemsRepository:
    """
    Repository for menu items operations.
    
    Uses SP_MENU_ITEMS_LST_BY_ROL_FUNC to retrieve menu items based on:
    - User role (via SP_ROL_LST_BY_ID_USUARIO)
    - User company (from USUARIOS table)
    - Assigned functionalities (from ROL_FUNCIONALIDAD)
    
    The stored procedure performs:
    1. Retrieves user's ID_TIPO_ROL via SP_ROL_LST_BY_ID_USUARIO
    2. Retrieves user's ID_EMPRESA from USUARIOS table
    3. Joins PARAMETROS (Maestro 9 - menu items) with PARAMETROS (Maestro 10 - functionalities)
       using the relationship: func.NUM3 = p.NUM1
    4. Joins with ROL_FUNCIONALIDAD table using: rf.ID_FUNCIONALIDAD = func.NUM1
    5. Filters by exact role match, company, and active status (ID_ESTADO_REGISTRO = 1)
    6. Returns results ordered by NUM2 (display order)
    
    This design eliminates hardcoded CASE statements and string concatenation,
    making the system scalable when adding new modules or routes.
    """

    # Expected columns from the stored procedure result set
    EXPECTED_COLUMNS: Set[str] = {'NUM1', 'NUM2', 'PATH', 'LABEL', 'ICON'}

    def __init__(self, db: Session):
        """
        Initialize repository with database session.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    @retry_on_db_error(max_retries=3, delay=1)
    def get_menu_items(self, id_usuario: int) -> List[Dict[str, Any]]:
        """
        Get menu items for a user using stored procedure SP_MENU_ITEMS_LST_BY_ROL_FUNC.
        
        The SP uses JOIN relationships based on logical identifiers (NUM1, NUM3) instead of
        technical auto-incremental IDs (ID_PARAMETRO), ensuring consistency across environments.
        
        Relationship flow:
        - PARAMETROS (Maestro 9).NUM1 ← linked by → PARAMETROS (Maestro 10).NUM3
        - PARAMETROS (Maestro 10).NUM1 ← linked by → ROL_FUNCIONALIDAD.ID_FUNCIONALIDAD
        
        Args:
            id_usuario: User ID to retrieve menu items for

        Returns:
            List of dictionaries with menu item data:
            - NUM1 (int): Item logical ID (business identifier)
            - NUM2 (int): Display order
            - PATH (str): Route path (e.g., '/n/rag/chat')
            - LABEL (str): Display label (e.g., 'Chat')
            - ICON (str): Lucide React icon name (e.g., 'MessageSquare')
            
            Returns empty list if:
            - Fetch operation failed
            - User has no assigned functionalities
            - User's role or company is invalid
            
        Raises:
            Exception: If critical database error occurs after retries
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                # Execute stored procedure with user ID parameter
                cursor.execute(
                    "EXEC SP_MENU_ITEMS_LST_BY_ROL_FUNC @ID_USUARIO = ?",
                    id_usuario
                )

                results = []

                # Iterate through all result sets (SP may return multiple)
                while True:
                    try:
                        # Check if current result set has data
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            
                            # Validate this is the expected menu items result set
                            # This prevents processing intermediate result sets from nested SPs
                            if set(columns) == self.EXPECTED_COLUMNS:
                                rows = cursor.fetchall()

                                if rows:
                                    logger.debug(f"Processing {len(rows)} menu items from SP for user {id_usuario}")
                                    
                                    # Convert rows to dictionaries
                                    for row in rows:
                                        result_dict = dict(zip(columns, row))
                                        
                                        # Convert Decimal/numeric types to int for ID and order fields
                                        # NUM1 is the logical business ID (1, 2, 3...)
                                        # NUM2 is the display order
                                        numeric_fields = ['NUM1', 'NUM2']
                                        for field in numeric_fields:
                                            if field in result_dict and result_dict[field] is not None:
                                                result_dict[field] = int(result_dict[field])
                                        
                                        results.append(result_dict)
                            else:
                                # Log skipped result sets for debugging
                                logger.debug(
                                    f"Skipping intermediate result set with columns: {columns} "
                                    f"(expected: {self.EXPECTED_COLUMNS})"
                                )

                    except Exception as fetch_error:
                        logger.error(
                            f"Fetch error in get_menu_items for user {id_usuario}: {fetch_error}",
                            exc_info=True
                        )

                    # Move to next result set
                    try:
                        if not cursor.nextset():
                            break
                    except Exception as nextset_error:
                        # Transaction error is expected when SP manages its own transactions
                        # This is normal behavior for some SQL Server stored procedures
                        if "Transaction count after EXECUTE" in str(nextset_error):
                            logger.debug(
                                f"SP manages its own transactions (expected behavior): {nextset_error}"
                            )
                        else:
                            logger.error(
                                f"Nextset error in get_menu_items for user {id_usuario}: {nextset_error}",
                                exc_info=True
                            )
                        break

                cursor.close()
                self.db.commit()
                
                logger.info(
                    f"Successfully retrieved {len(results)} menu items for user {id_usuario}"
                )
                return results

            except Exception as cursor_error:
                logger.error(
                    f"Cursor error in get_menu_items for user {id_usuario}: {cursor_error}",
                    exc_info=True
                )
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(
                f"Error fetching menu items with SP for user {id_usuario}: {e}",
                exc_info=True
            )
            self.db.rollback()
            return []