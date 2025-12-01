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

                # Iterate through all result sets
                while True:
                    try:
                        # Check if we have columns (indicating data)
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            rows = cursor.fetchall()

                            # Check if this result set contains the message columns
                            has_message_columns = 'ID_TIPO_MENSAJE' in columns and 'MENSAJE' in columns

                            if has_message_columns and rows:
                                # This is the result set we want to capture
                                for row in rows:
                                    result_dict = dict(zip(columns, row))
                                    # Convert Decimal to int for ID_TIPO_MENSAJE
                                    if 'ID_TIPO_MENSAJE' in result_dict:
                                        result_dict['ID_TIPO_MENSAJE'] = int(result_dict['ID_TIPO_MENSAJE'])
                                    results.append(result_dict)

                    except Exception as fetch_error:
                        logger.error(f"Fetch error: {fetch_error}")

                    # Move to next result set
                    try:
                        if not cursor.nextset():
                            break
                    except Exception as nextset_error:
                        # Transaction error is expected when SP manages its own transactions
                        if "Transaction count after EXECUTE" in str(nextset_error):
                            logger.debug(f"SP manages its own transactions (expected): {nextset_error}")
                        else:
                            logger.error(f"Nextset error: {nextset_error}")
                        break

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

    def get_companies(self) -> List[Dict[str, Any]]:
        """
        Get all companies using stored procedure SP_EMPRESAS_LST

        Returns:
            List of dictionaries with company data
            Empty list if fetch failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute("EXEC SP_EMPRESAS_LST")

                companies = []

                # Get the company data
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()

                    # Convert rows to list of dictionaries
                    for row in rows:
                        company_dict = dict(zip(columns, row))
                        companies.append(company_dict)

                cursor.close()
                return companies

            except Exception as cursor_error:
                logger.error(f"Cursor error in get_companies: {cursor_error}")
                cursor.close()
                raise

        except Exception as e:
            logger.error(f"Error fetching companies with SP: {e}")
            return []

    def update_company_status(
        self,
        id_usuario: int,
        id_empresa: int,
        status: int
    ) -> List[Dict[str, Any]]:
        """
        Update company status using stored procedure SP_UPDATE_EMPRESA_STATUS

        Args:
            id_usuario: User ID
            id_empresa: Company ID
            status: Status value to set (typically 1=active, 0=deleted)

        Returns:
            List of dictionaries with result information
            Empty list if update failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_UPDATE_EMPRESA_STATUS @ID_USUARIO = ?, @ID_EMPRESA = ?, @STATUS = ?",
                    id_usuario,
                    id_empresa,
                    status
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
                                for row in rows:
                                    result_dict = dict(zip(columns, row))
                                    # Convert Decimal to int for numeric IDs
                                    for key, value in result_dict.items():
                                        if isinstance(value, type(1.0)) and key.startswith('ID_'):
                                            result_dict[key] = int(value)
                                    results.append(result_dict)

                    except Exception as fetch_error:
                        logger.error(f"Fetch error: {fetch_error}")

                    # Move to next result set
                    try:
                        if not cursor.nextset():
                            break
                    except Exception as nextset_error:
                        # Transaction error is expected when SP manages its own transactions
                        if "Transaction count after EXECUTE" in str(nextset_error):
                            logger.debug(f"SP manages its own transactions (expected): {nextset_error}")
                        else:
                            logger.error(f"Nextset error: {nextset_error}")
                        break

                cursor.close()
                self.db.commit()
                return results

            except Exception as cursor_error:
                logger.error(f"Cursor error in update_company_status: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error updating company status with SP: {e}")
            self.db.rollback()
            return []
