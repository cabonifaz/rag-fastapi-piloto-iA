"""Repository for phone code database operations."""

from sqlalchemy.orm import Session
from typing import List, Dict, Any
import logging
from app.core.database import retry_on_db_error

logger = logging.getLogger(__name__)


class PhoneCodeRepository:
    """
    Repository for phone code operations.

    Returns country phone codes with:
    - CODIGO_NUMERICO: Numeric country code
    - CODIGO_ISO: ISO country code (e.g., PE, ES, MX)
    - NOMBRE_PAIS: Country name
    - PREFIJO_TELEFONICO: Phone prefix (e.g., +51, +34, +52)
    """

    def __init__(self, db: Session):
        self.db = db

    @retry_on_db_error(max_retries=3, delay=1)
    def get_phone_codes(self) -> List[Dict[str, Any]]:
        """
        Get all phone codes using stored procedure SP_CODIGOS_PAIS_LST

        Returns:
            List of dictionaries with phone code data:
            - CODIGO_NUMERICO: Numeric country code
            - CODIGO_ISO: ISO country code
            - NOMBRE_PAIS: Country name
            - PREFIJO_TELEFONICO: Phone prefix
            Empty list if fetch failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute("EXEC SP_CODIGOS_PAIS_LST")

                phone_codes = []

                # Get the phone code data
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()

                    # Convert rows to list of dictionaries
                    for row in rows:
                        phone_code_dict = dict(zip(columns, row))
                        phone_codes.append(phone_code_dict)

                cursor.close()
                return phone_codes

            except Exception as cursor_error:
                logger.error(f"Cursor error in get_phone_codes: {cursor_error}")
                cursor.close()
                raise

        except Exception as e:
            logger.error(f"Error fetching phone codes with SP: {e}")
            return []
