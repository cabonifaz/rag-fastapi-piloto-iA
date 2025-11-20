"""Repository for users database operations."""

from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class UsersRepository:
    """
    Repository for USUARIOS table operations.
    """

    def __init__(self, db: Session):
        self.db = db

    def get_usuarios(self, id_empresa: int) -> List[Dict[str, Any]]:
        """
        Get all users for a company using stored procedure SP_USUARIOS_LST

        Args:
            id_empresa: Company ID

        Returns:
            List of dictionaries containing user information
            Empty list if query failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_USUARIOS_LST @ID_EMPRESA = ?",
                    id_empresa
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
                                        if field in result_dict and result_dict[field] is not None:
                                            result_dict[field] = int(result_dict[field])
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
                logger.error(f"Cursor error in get_usuarios: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error getting usuarios with SP_USUARIOS_LST: {e}")
            self.db.rollback()
            return []

    def create_usuario(
        self,
        nuevo_usuario: str,
        password: str,
        nombres: str,
        apellidos: str,
        id_tipo_rol: int,
        id_empresa: int,
        areas_string: str
    ) -> List[Dict[str, Any]]:
        """
        Create a new user using stored procedure SP_USUARIO_CREATE

        Args:
            nuevo_usuario: Username (max 100 chars)
            password: Password (max 100 chars)
            nombres: First names (max 100 chars)
            apellidos: Last names (max 100 chars)
            id_tipo_rol: Role type ID
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
                    "EXEC SP_USUARIO_CREATE @NUEVO_USUARIO = ?, @PASSWORD = ?, @NOMBRES = ?, @APELLIDOS = ?, @ID_TIPO_ROL = ?, @ID_EMPRESA = ?, @AREAS_STRING = ?",
                    nuevo_usuario,
                    password,
                    nombres,
                    apellidos,
                    id_tipo_rol,
                    id_empresa,
                    areas_string
                )

                results = []
                result_set_num = 1

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
                        result_set_num += 1
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
            logger.error(f"Error creating usuario with SP_USUARIO_CREATE: {e}")
            self.db.rollback()
            return []
