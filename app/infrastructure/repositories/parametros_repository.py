"""Repository to fetch PARAMETROS values via stored procedure."""

import logging
from typing import Optional
from decimal import Decimal

logger = logging.getLogger(__name__)


class ParametrosRepository:
    """Repository to call SP_PARAMETROS_LST and return parameter values."""

    def __init__(self, db):
        self.db = db

    def get_param_by_num1(self, num1: int) -> Optional[int]:
        """Call SP_PARAMETROS_LST for a group id (GRP_ID_MAESTRO) and return NUM1 for the matching ID_MAESTRO.

        This stored procedure returns two result sets (first: messages, second: parameters). We iterate result sets
        until we find the PARAMETERS result set (which contains columns like ID_PARAMETRO, ID_MAESTRO, NUM1, NUM2...).
        We then locate the row where ID_MAESTRO equals the requested `num1` and return its NUM1 value as int.
        """
        try:
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            # Ensure we pass the group id as string (SP expects VARCHAR param)
            cursor.execute("EXEC SP_PARAMETROS_LST @GRP_ID_MAESTRO = ?", str(num1))

            # Iterate over result sets until we find the one that has NUM1 in columns
            while True:
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    # If this result set looks like the PARAMETERS table
                    if 'NUM1' in columns and 'ID_MAESTRO' in columns:
                        rows = cursor.fetchall()
                        for row in rows:
                            row_dict = dict(zip(columns, row))
                            try:
                                if int(str(row_dict.get('ID_MAESTRO', 0))) == int(num1):
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

                        # If we processed the PARAMETERS set but didn't find a matching ID_MAESTRO, return None
                        cursor.close()
                        return None

                # Advance to next result set, if any
                if not cursor.nextset():
                    break

            cursor.close()
            return None

        except Exception as e:
            logger.error(f"Error fetching parameter num1={num1}: {e}")
            return None