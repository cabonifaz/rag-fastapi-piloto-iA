"""Repository to fetch PARAMETROS values via stored procedure."""

from sqlalchemy.orm import Session
from typing import List, Dict, Any
from decimal import Decimal
import logging
from app.core.database import retry_on_db_error

logger = logging.getLogger(__name__)


class ParametrosRepository:
    """Repository to call SP_PARAMETROS_LST and return parameter values."""

    def __init__(self, db: Session):
        self.db = db

    @retry_on_db_error(max_retries=3, delay=1)
    def get_params_by_id_maestro(self, grp_id_maestro: str) -> List[Dict[str, Any]]:
        """Call SP_PARAMETROS_LST for a group id and return all parameter rows.

        This stored procedure returns two result sets:
        - First: message (NUM2, MENSAJE)
        - Second: parameters data (ID_PARAMETRO, ID_MAESTRO, ID_SUB_MAESTRO, NUM1, NUM2, NUM3, STRING1, STRING2, STRING3)

        Args:
            grp_id_maestro: Group ID (GRP_ID_MAESTRO) to look up

        Returns:
            List of dictionaries with all parameter rows for the group.
            Empty list if not found or on error.
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute("EXEC SP_PARAMETROS_LST @GRP_ID_MAESTRO = ?", grp_id_maestro)

                results = []

                # Iterate through all result sets
                while True:
                    try:
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]

                            # If this result set looks like the PARAMETERS table
                            if 'ID_PARAMETRO' in columns and 'ID_MAESTRO' in columns:
                                rows = cursor.fetchall()
                                for row in rows:
                                    row_dict = dict(zip(columns, row))

                                    # Convert Decimal to int for numeric fields
                                    numeric_fields = ['ID_PARAMETRO', 'ID_MAESTRO', 'ID_SUB_MAESTRO', 'NUM1', 'NUM2', 'NUM3']
                                    for field in numeric_fields:
                                        if field in row_dict and row_dict[field] is not None:
                                            if isinstance(row_dict[field], Decimal):
                                                row_dict[field] = int(row_dict[field])

                                    # Strip whitespace from string fields
                                    string_fields = ['STRING1', 'STRING2', 'STRING3']
                                    for field in string_fields:
                                        if field in row_dict and isinstance(row_dict[field], str):
                                            row_dict[field] = row_dict[field].strip()

                                    results.append(row_dict)

                    except Exception as fetch_error:
                        logger.error(f"Fetch error in get_params_by_id_maestro: {fetch_error}")

                    # Move to next result set
                    try:
                        if not cursor.nextset():
                            break
                    except Exception as nextset_error:
                        # Transaction error is expected when SP manages its own transactions
                        if "Transaction count after EXECUTE" in str(nextset_error):
                            logger.debug(f"SP manages its own transactions (expected): {nextset_error}")
                        else:
                            logger.error(f"Nextset error in get_params_by_id_maestro: {nextset_error}")
                        break

                cursor.close()
                self.db.commit()
                return results

            except Exception as cursor_error:
                logger.error(f"Cursor error in get_params_by_id_maestro: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error fetching params for grp_id_maestro={grp_id_maestro}: {e}")
            self.db.rollback()
            return []
