"""Repository for company database operations using actual COMPANY table structure."""

from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List, Dict, Any
import logging
from app.core.database import retry_on_db_error

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

    @retry_on_db_error(max_retries=3, delay=1)
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

    @retry_on_db_error(max_retries=3, delay=1)
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

    @retry_on_db_error(max_retries=3, delay=1)
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

    @retry_on_db_error(max_retries=3, delay=1)
    def get_companies_login(self) -> List[Dict[str, Any]]:
        """
        Get companies with their secret keys using stored procedure SP_EMPRESAS_LST_LOGIN

        Returns:
            List of dictionaries with RAZON_SOCIAL and SECRET_KEY
            Empty list if fetch failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute("EXEC SP_EMPRESAS_LST_LOGIN")

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
                logger.error(f"Cursor error in get_companies_login: {cursor_error}")
                cursor.close()
                raise

        except Exception as e:
            logger.error(f"Error fetching companies login with SP: {e}")
            return []

    @retry_on_db_error(max_retries=3, delay=1)
    def update_company_logo(
        self,
        id_usuario: int,
        id_empresa: int,
        logo_url: str
    ) -> List[Dict[str, Any]]:
        """
        Update company logo URL using stored procedure SP_UPDATE_EMPRESA_LOGO

        Args:
            id_usuario: User ID performing the update
            id_empresa: Company ID
            logo_url: S3 URL/path for the company logo (max 255 chars)

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
                    "EXEC SP_UPDATE_EMPRESA_LOGO @ID_USUARIO = ?, @ID_EMPRESA = ?, @LOGO_URL = ?",
                    id_usuario,
                    id_empresa,
                    logo_url
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
                logger.error(f"Cursor error in update_company_logo: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error updating company logo with SP: {e}")
            self.db.rollback()
            return []


    @retry_on_db_error(max_retries=3, delay=1)
    def get_companies_paginated(
        self,
        id_usuario: int,
        num_pagina: int = 1,
        tam_pagina: int = 10,
        term_busqueda: Optional[str] = None,
        campo_orden: str = 'RAZON_SOCIAL',
        dir_orden: str = 'ASC',
        filtro_estado: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get paginated companies using stored procedure SP_EMPRESAS_LST_PAG

        Args:
            id_usuario: User ID for role validation
            num_pagina: Page number (starting at 1)
            tam_pagina: Number of rows per page
            term_busqueda: Optional search term for RAZON_SOCIAL or RUC
            campo_orden: Field to sort by (default 'RAZON_SOCIAL')
            dir_orden: Sort direction ASC or DESC (default 'ASC')
            filtro_estado: Optional filter for ID_ESTADO_REGISTRO (None=all, 1=active, 0=inactive)

        Returns:
            Dictionary with:
                - data: List of company dictionaries
                - pagination: Dictionary with pagination metadata (total_records, current_page, page_size, total_pages)
                - message_result: Dict with ID_TIPO_MENSAJE and MENSAJE if error/authorization
            Empty dict with empty list if fetch failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_EMPRESAS_LST_PAG @NUM_PAGINA = ?, @TAM_PAGINA = ?, @TERM_BUSQUEDA = ?, @CAMPO_ORDEN = ?, @DIR_ORDEN = ?, @ID_USUARIO = ?, @FILTRO_ESTADO = ?",
                    num_pagina,
                    tam_pagina,
                    term_busqueda,
                    campo_orden,
                    dir_orden,
                    id_usuario,
                    filtro_estado
                )

                companies = []
                pagination_info = {}
                message_result = None

                # Iterate through all result sets
                while True:
                    try:
                        # Check if we have columns (indicating data)
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            rows = cursor.fetchall()

                            # Check if this result set contains the message columns
                            has_message_columns = 'ID_TIPO_MENSAJE' in columns and 'MENSAJE' in columns

                            # Check if this result set contains COMPANY data (not role data)
                            has_company_data = 'ID_EMPRESA' in columns and 'RUC' in columns and 'RAZON_SOCIAL' in columns

                            if has_message_columns and rows:
                                # This is an error/authorization message result
                                for row in rows:
                                    result_dict = dict(zip(columns, row))
                                    # Convert Decimal to int for ID_TIPO_MENSAJE
                                    if 'ID_TIPO_MENSAJE' in result_dict:
                                        result_dict['ID_TIPO_MENSAJE'] = int(result_dict['ID_TIPO_MENSAJE'])
                                    message_result = result_dict
                                    
                            elif has_company_data and rows:  # Solo procesar si tiene datos de empresas
                                # This is the company data with pagination
                                # Extract pagination info from first row
                                first_row = rows[0]
                                row_dict = dict(zip(columns, first_row))
                                
                                # Solo extraer pagination si existen esas columnas
                                if 'TotalRecords' in columns:
                                    pagination_info = {
                                        'total_records': int(row_dict.get('TotalRecords', 0)),
                                        'current_page': int(row_dict.get('CurrentPage', num_pagina)),
                                        'page_size': int(row_dict.get('PageSize', tam_pagina)),
                                        'total_pages': int(row_dict.get('TotalPages', 0))
                                    }

                                # Convert rows to list of dictionaries and remove pagination metadata
                                for row in rows:
                                    company_dict = dict(zip(columns, row))
                                    # Remove pagination metadata columns
                                    company_dict.pop('TotalRecords', None)
                                    company_dict.pop('CurrentPage', None)
                                    company_dict.pop('PageSize', None)
                                    company_dict.pop('TotalPages', None)
                                    companies.append(company_dict)

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
                
                return {
                    'data': companies,
                    'pagination': pagination_info,
                    'message_result': message_result
                }

            except Exception as cursor_error:
                logger.error(f"Cursor error in get_companies_paginated: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error fetching paginated companies with SP: {e}")
            self.db.rollback()
            return {'data': [], 'pagination': {}, 'message_result': None}