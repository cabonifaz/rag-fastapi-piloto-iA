"""Repository for company database operations using actual COMPANY table structure."""

from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CompanyRepository:
    """
    Repository for COMPANY table operations.

    Table structure:
    - ID_EMPRESA: Primary key (auto-generated)
    - RUC: RUC identifier
    - RAZON_SOCIAL: Company identifier
    - USUCRE: User who created the company
    - USUMOD: User who last modified the company
    - FCHMOD: Last modification date
    - FCHCRE: Creation date (auto-generated)
    - ID_ESTADO_REGISTRO: Status (1=active, 0=deleted)
    """

    def __init__(self, db: Session):
        self.db = db

    def create_company(
        self,
        id_usuario: int,
        ruc: str,
        razon_social: str
    ) -> List[Dict[str, Any]]:
        """
        Create a new empresa base using stored procedure SP_CREATE_EMPRESA_BASE

        Args:
            id_usuario: User ID
            ruc: RUC identifier (max 20 chars)
            razon_social: Company name (max 200 chars)

        Returns:
            List of dictionaries with: id_rol, message, id_company, id_area
            Empty list if creation failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_CREATE_EMPRESA_BASE @ID_USUARIO = ?, @RUC = ?, @RAZON_SOCIAL = ?",
                    id_usuario,
                    ruc,
                    razon_social
                )

                results = []

                # Check if we have results
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()

                    # Convert rows to list of dictionaries
                    for row in rows:
                        result_dict = dict(zip(columns, row))
                        results.append(result_dict)

                cursor.close()
                self.db.commit()
                return results

            except Exception as cursor_error:
                logger.error(f"Cursor error in create_company: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error creating empresa base with SP: {e}")
            self.db.rollback()
            return []
