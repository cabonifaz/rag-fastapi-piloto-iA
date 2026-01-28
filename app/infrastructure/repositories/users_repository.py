"""Repository for users database operations."""

from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List, Dict, Any
import logging
from app.core.database import retry_on_db_error

logger = logging.getLogger(__name__)


class UsersRepository:
    """
    Repository for USUARIOS table operations.
    """

    def __init__(self, db: Session):
        self.db = db


    @retry_on_db_error(max_retries=3, delay=1)
    def create_usuario(
        self,
        id_usuario: int,
        nuevo_usuario: str,
        password: str,
        nombres: str,
        apellidos: str,
        codigo_pais: str,
        telefono: str,
        nuevo_rol: int,
        id_empresa: int,
        areas_string: str
    ) -> List[Dict[str, Any]]:
        """
        Create a new user using stored procedure SP_CREATE_USUARIO

        Args:
            id_usuario: User ID
            nuevo_usuario: Username (max 100 chars)
            password: Password (max 100 chars)
            nombres: First names (max 100 chars)
            apellidos: Last names (max 100 chars)
            codigo_pais: Country code (max 8 chars)
            telefono: Phone number (max 15 chars)
            nuevo_rol: Role type ID
            id_empresa: Company ID
            areas_string: Comma-separated area IDs (max 100 chars)

        Returns:
            List of dictionaries with: ID_TIPO_MENSAJE, MENSAJE
            Empty list if creation failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_CREATE_USUARIO @ID_USUARIO = ?, @NUEVO_USUARIO = ?, @PASSWORD = ?, @NOMBRES = ?, @APELLIDOS = ?, @CODIGO_PAIS = ?, @TELEFONO = ?, @NUEVO_ROL = ?, @ID_EMPRESA = ?, @AREAS_STRING = ?",
                    id_usuario,
                    nuevo_usuario,
                    password,
                    nombres,
                    apellidos,
                    codigo_pais,
                    telefono,
                    nuevo_rol,
                    id_empresa,
                    areas_string
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
                logger.error(f"Cursor error in create_usuario: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error creating usuario with SP_CREATE_USUARIO: {e}")
            self.db.rollback()
            return []

    @retry_on_db_error(max_retries=3, delay=1)
    def update_usuario(
        self,
        id_admin: int,
        id_usuario: int,
        usuario: str,
        nombres: str,
        apellidos: str,
        telefono: str = None
    ) -> List[Dict[str, Any]]:
        """
        Update user data using stored procedure SP_UPDATE_DATOS_USUARIO

        Args:
            id_admin: Admin user ID performing the update
            id_usuario: User ID to update
            usuario: New username (max 200 chars)
            nombres: New first names (max 200 chars)
            apellidos: New last names (max 200 chars)
            telefono: Phone number (max 15 chars), optional (default None)

        Returns:
            List of dictionaries with: ID_TIPO_MENSAJE, MENSAJE
            Empty list if update failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_UPDATE_DATOS_USUARIO @ID_ADMIN = ?, @ID_USUARIO = ?, @USUARIO = ?, @NOMBRES = ?, @APELLIDOS = ?, @TELEFONO = ?",
                    id_admin,
                    id_usuario,
                    usuario,
                    nombres,
                    apellidos,
                    telefono
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
                logger.error(f"Cursor error in update_usuario: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error updating usuario with SP_UPDATE_DATOS_USUARIO: {e}")
            self.db.rollback()
            return []

    @retry_on_db_error(max_retries=3, delay=1)
    def update_usuario_status(
        self,
        id_admin: int,
        id_usuario: int,
        status: int
    ) -> List[Dict[str, Any]]:
        """
        Update user status using stored procedure SP_UPDATE_USUARIO_STATUS

        Args:
            id_admin: Admin user ID performing the update
            id_usuario: User ID to update
            status: New status (0 = inactive, 1 = active)

        Returns:
            List of dictionaries with: ID_TIPO_MENSAJE, MENSAJE
            Empty list if update failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_UPDATE_USUARIO_STATUS @ID_ADMIN = ?, @ID_USUARIO = ?, @STATUS = ?",
                    id_admin,
                    id_usuario,
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
                logger.error(f"Cursor error in update_usuario_status: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error updating usuario status with SP_UPDATE_USUARIO_STATUS: {e}")
            self.db.rollback()
            return []

    @retry_on_db_error(max_retries=3, delay=1)
    def update_usuario_password(
        self,
        id_admin: int,
        id_usuario: int,
        clave_acceso: str
    ) -> List[Dict[str, Any]]:
        """
        Update user password using stored procedure SP_UPDATE_USUARIO_PASSWORD

        Args:
            id_admin: Admin user ID performing the update
            id_usuario: User ID to update
            clave_acceso: New password (max 100 chars)

        Returns:
            List of dictionaries with: ID_TIPO_MENSAJE, MENSAJE
            Empty list if update failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_UPDATE_USUARIO_PASSWORD @ID_ADMIN = ?, @ID_USUARIO = ?, @CLAVE_ACCESO = ?",
                    id_admin,
                    id_usuario,
                    clave_acceso
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
                logger.error(f"Cursor error in update_usuario_password: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error updating usuario password with SP_UPDATE_USUARIO_PASSWORD: {e}")
            self.db.rollback()
            return []

    @retry_on_db_error(max_retries=3, delay=1)
    def update_usuario_access(
        self,
        id_admin: int,
        id_usuario: int,
        nuevo_rol: int,
        id_empresa: int,
        areas_string: str
    ) -> List[Dict[str, Any]]:
        """
        Update user role and areas access using stored procedure SP_UPDATE_USUARIO_ACCESS

        Args:
            id_admin: Admin user ID performing the update
            id_usuario: User ID to update
            nuevo_rol: New role type ID
            id_empresa: Company ID
            areas_string: Comma-separated area IDs (max 100 chars)

        Returns:
            List of dictionaries with: ID_TIPO_MENSAJE, MENSAJE
            Empty list if update failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_UPDATE_USUARIO_ACCESS @ID_ADMIN = ?, @ID_USUARIO = ?, @NUEVO_ROL = ?, @ID_EMPRESA = ?, @AREAS_STRING = ?",
                    id_admin,
                    id_usuario,
                    nuevo_rol,
                    id_empresa,
                    areas_string
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
                logger.error(f"Cursor error in update_usuario_access: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error updating usuario access with SP_UPDATE_USUARIO_ACCESS: {e}")
            self.db.rollback()
            return []

#FUNCIONES DE N8N

    @retry_on_db_error(max_retries=3, delay=1)
    def get_user_data_for_n8n(self, id_agente: int, telefono: str) -> List[Dict[str, Any]]:
        """
        Get complete user data for n8n integration using stored procedure SP_GET_USER_DATA_FOR_N8N

        Returns user data along with company, area, IA area, and chat information.

        Args:
            id_agente: ID agent
            telefono: Cell Phone number

        Returns:
            List of dictionaries containing:
            - Message result (status)
            - User data (if found)
            - Company and Area IDs (ID_EMPRESA, ID_AREA)
            - IA Area ID (ID_IA_AREA)
            - Chat ID (ID_CHAT)
            Empty list if query failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_GET_USER_DATA_FOR_N8N @ID_AGENTE = ?, @TELEFONO = ?",
                    id_agente,
                    telefono
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

                                    # Convert Decimal to int for numeric fields
                                    numeric_fields = [
                                        'ID_TIPO_MENSAJE', 'ID_USUARIO', 'ID_EMPRESA',
                                        'ID_AREA', 'ID_IA_AREA', 'ID_CHAT'
                                    ]
                                    for field in numeric_fields:
                                        if field in result_dict and result_dict[field] is not None:
                                            result_dict[field] = int(result_dict[field])

                                    # Include all relevant result sets:
                                    # 1. Message result (ID_TIPO_MENSAJE, MENSAJE)
                                    # 2. User data (ID_USUARIO, USUARIO, etc.)
                                    # 3. Company/Area data (ID_EMPRESA, ID_AREA)
                                    # 4. IA Area data (ID_IA_AREA)
                                    # 5. Chat data (ID_CHAT)
                                    is_message_result = 'ID_TIPO_MENSAJE' in result_dict and 'MENSAJE' in result_dict
                                    is_user_data_result = 'ID_USUARIO' in result_dict and 'USUARIO' in result_dict
                                    is_company_area_result = 'ID_EMPRESA' in result_dict and 'ID_AREA' in result_dict
                                    is_ia_area_result = 'ID_IA_AREA' in result_dict
                                    is_chat_result = 'ID_CHAT' in result_dict

                                    # Include all relevant result types (exclude only agent info)
                                    if (is_message_result or is_user_data_result or
                                        is_company_area_result or is_ia_area_result or is_chat_result):
                                        results.append(result_dict)

                    except Exception as fetch_error:
                        logger.error(f"Fetch error in get_user_data_for_n8n: {fetch_error}")

                    # Move to next result set
                    try:
                        if not cursor.nextset():
                            break
                    except Exception as nextset_error:
                        # Transaction error is expected when SP manages its own transactions
                        if "Transaction count after EXECUTE" in str(nextset_error):
                            logger.debug(f"SP manages its own transactions (expected): {nextset_error}")
                        else:
                            logger.error(f"Nextset error in get_user_data_for_n8n: {nextset_error}")
                        break

                cursor.close()
                self.db.commit()
                return results

            except Exception as cursor_error:
                logger.error(f"Cursor error in get_user_data_for_n8n: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error getting user data for n8n with SP_GET_USER_DATA_FOR_N8N: {e}")
            self.db.rollback()
            return []


    @retry_on_db_error(max_retries=3, delay=1)
    def get_usuarios_paginated(
        self,
        id_usuario: int,
        id_empresa: int,
        num_pagina: int = 1,
        tam_pagina: int = 10,
        term_busqueda: Optional[str] = None,
        campo_orden: str = 'APELLIDOS',
        dir_orden: str = 'ASC',
        filtro_estado: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get paginated users using stored procedure SP_USUARIOS_LST_PAG

        Args:
            id_usuario: User ID for role validation
            id_empresa: Company ID
            num_pagina: Page number (starting at 1)
            tam_pagina: Number of rows per page
            term_busqueda: Optional search term for USUARIO, NOMBRES, APELLIDOS, TELEFONO, AREA, ROL
            campo_orden: Field to sort by (default 'APELLIDOS')
            dir_orden: Sort direction ASC or DESC (default 'ASC')
            filtro_estado: Optional filter for ID_ESTADO_REGISTRO (None=all, 1=active, 0=inactive)

        Returns:
            Dictionary with:
                - data: List of user dictionaries
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
                    "EXEC SP_USUARIOS_LST_PAG @NUM_PAGINA = ?, @TAM_PAGINA = ?, @TERM_BUSQUEDA = ?, @CAMPO_ORDEN = ?, @DIR_ORDEN = ?, @ID_USUARIO = ?, @ID_EMPRESA = ?, @FILTRO_ESTADO = ?",
                    num_pagina,
                    tam_pagina,
                    term_busqueda,
                    campo_orden,
                    dir_orden,
                    id_usuario,
                    id_empresa,
                    filtro_estado
                )

                usuarios = []
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

                            # Check if this result set contains USER data
                            has_user_data = 'ID_USUARIO' in columns and 'USUARIO' in columns

                            if has_message_columns and rows:
                                # This is an error/authorization message result
                                for row in rows:
                                    result_dict = dict(zip(columns, row))
                                    # Convert Decimal to int for ID_TIPO_MENSAJE
                                    if 'ID_TIPO_MENSAJE' in result_dict:
                                        result_dict['ID_TIPO_MENSAJE'] = int(result_dict['ID_TIPO_MENSAJE'])
                                    message_result = result_dict
                                    
                            elif has_user_data and rows:  
                                # This is the user data with pagination
                                # Extract pagination info from first row
                                first_row = rows[0]
                                row_dict = dict(zip(columns, first_row))
                                
                                if 'TotalRecords' in columns:
                                    pagination_info = {
                                        'total_records': int(row_dict.get('TotalRecords', 0)),
                                        'current_page': int(row_dict.get('CurrentPage', num_pagina)),
                                        'page_size': int(row_dict.get('PageSize', tam_pagina)),
                                        'total_pages': int(row_dict.get('TotalPages', 0))
                                    }

                                # Convert rows to list of dictionaries and remove pagination metadata
                                for row in rows:
                                    user_dict = dict(zip(columns, row))
                                    
                                    # Convert Decimal to int for numeric IDs
                                    numeric_fields = [
                                        'ID_USUARIO_EMPR_AREA',
                                        'ID_USUARIO',
                                        'ID_ESTADO_REGISTRO',
                                        'ID_EMPRESA',
                                        'ID_AREA',
                                        'ID_TIPO_ROL'
                                    ]
                                    for field in numeric_fields:
                                        if field in user_dict and user_dict[field] is not None:
                                            user_dict[field] = int(user_dict[field])
                                    
                                    # Remove pagination metadata columns
                                    user_dict.pop('TotalRecords', None)
                                    user_dict.pop('CurrentPage', None)
                                    user_dict.pop('PageSize', None)
                                    user_dict.pop('TotalPages', None)
                                    usuarios.append(user_dict)  
                    except Exception as fetch_error:
                        logger.error(f"Fetch error in get_usuarios_paginated: {fetch_error}")

                    # Move to next result set
                    try:
                        if not cursor.nextset():
                            break
                    except Exception as nextset_error:
                        # Transaction error is expected when SP manages its own transactions
                        if "Transaction count after EXECUTE" in str(nextset_error):
                            logger.debug(f"SP manages its own transactions (expected): {nextset_error}")
                        else:
                            logger.error(f"Nextset error in get_usuarios_paginated: {nextset_error}")
                        break

                cursor.close()
                self.db.commit()
                
                return {
                    'data': usuarios,
                    'pagination': pagination_info,
                    'message_result': message_result
                }

            except Exception as cursor_error:
                logger.error(f"Cursor error in get_usuarios_paginated: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error fetching paginated usuarios with SP: {e}")
            self.db.rollback()
            return {'data': [], 'pagination': {}, 'message_result': None}