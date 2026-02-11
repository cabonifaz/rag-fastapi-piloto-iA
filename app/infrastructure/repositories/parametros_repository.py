"""Repository to fetch PARAMETROS values via stored procedure."""

from sqlalchemy.orm import Session
from typing import Optional
from decimal import Decimal
import logging
from app.core.database import retry_on_db_error

logger = logging.getLogger(__name__)


class ParametrosRepository:
    """Repository to call SP_PARAMETROS_LST and return parameter values."""

    def __init__(self, db: Session):
        self.db = db

    @retry_on_db_error(max_retries=3, delay=1)
    def get_params_by_id_maestro(self, grp_id_maestro: str) -> Optional[int]:
        """Call SP_PARAMETROS_LST for a group id (GRP_ID_MAESTRO) and return NUM1 for the matching ID_MAESTRO.

        This stored procedure returns two result sets (first: messages, second: parameters). We iterate result sets
        until we find the PARAMETERS result set (which contains columns like ID_PARAMETRO, ID_MAESTRO, NUM1, NUM2...).
        We then locate the row where ID_MAESTRO equals the requested `grp_id_maestro` and return its NUM1 value as int.

        Args:
            grp_id_maestro: Group ID (GRP_ID_MAESTRO) to look up

        Returns:
            Integer NUM1 value for the matching ID_MAESTRO, or None if not found or on error
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute("EXEC SP_PARAMETROS_LST @GRP_ID_MAESTRO = ?", grp_id_maestro)

                # Iterate over result sets until we find the one that has NUM1 in columns
                while True:
                    try:
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            # If this result set looks like the PARAMETERS table
                            if 'NUM1' in columns and 'ID_MAESTRO' in columns:
                                rows = cursor.fetchall()
                                for row in rows:
                                    row_dict = dict(zip(columns, row))
                                    try:
                                        if str(row_dict.get('ID_MAESTRO', '')) == grp_id_maestro:
                                            val = row_dict.get('NUM1')
                                            if val is None:
                                                continue
                                            if isinstance(val, Decimal):
                                                return int(val)
                                            try:
                                                return int(val)
                                            except Exception:
                                                continue
                                    except Exception:
                                        # ignore conversion errors and continue
                                        continue

                                # If we processed the PARAMETERS set but didn't find a matching ID_MAESTRO
                                cursor.close()
                                self.db.commit()
                                return None

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
                return None

            except Exception as cursor_error:
                logger.error(f"Cursor error in get_params_by_id_maestro: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error fetching parameter num1={num1}: {e}")
            self.db.rollback()
            return None
