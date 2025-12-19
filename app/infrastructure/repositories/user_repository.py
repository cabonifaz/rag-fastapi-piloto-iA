"""Repository for user database operations - handles only DB calls and stored procedures."""

from sqlalchemy.orm import Session
from sqlalchemy import text, and_
from app.models.user_models import Usuario
from typing import Optional, List, Dict, Any, Tuple
import logging
from app.core.database import retry_on_db_error

logger = logging.getLogger(__name__)


class UserRepository:
    """
    Repository for user database operations.

    Handles raw database calls and stored procedure execution only.
    All business logic remains in the service layer.
    """

    def __init__(self, db: Session):
        self.db = db

    # =============================================
    # Stored Procedure Calls - Raw DB Operations
    # =============================================

    @retry_on_db_error(max_retries=3, delay=1)
    def verify_user_acceso_sp(self, usuario: str, password: str, secret_key: str) -> Tuple[int, Optional[int]]:
        """
        Execute SP_VERIFY_USUARIO_ACCESO and return status with company ID

        Args:
            usuario: Username
            password: Plain text password
            secret_key: Company secret key

        Returns:
            Tuple of (status, id_empresa) where status is 1 on success or 0 on failure
        """
        try:
            query = text("""
                EXEC SP_VERIFY_USUARIO_ACCESO
                @USERNAME = :username,
                @PASSWORD = :password,
                @SECRET_KEY = :secret_key
            """)

            result = self.db.execute(query, {
                'username': usuario,
                'password': password,
                'secret_key': secret_key
            })

            status_data = result.fetchone()
            result.close()

            if not status_data:
                return 0, None

            status_dict = dict(status_data._mapping) if hasattr(status_data, '_mapping') else dict(zip(result.keys(), status_data))
            status = status_dict.get('STATUS', 0)
            id_empresa = status_dict.get('ID_EMPRESA')

            return status, id_empresa

        except Exception as e:
            logger.error(f"Error executing SP_VERIFY_USUARIO_ACCESO: {e}")
            raise

    @retry_on_db_error(max_retries=3, delay=1)
    def get_user_login_data_sp(self, usuario: str, id_empresa: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Execute SP_USUARIO_LOGIN_DATA and return all result sets

        The SP returns multiple result sets:
        - Result set 2: User data
        - Result set 3: Roles data
        - Result set 4: Company areas data

        Args:
            usuario: Username
            id_empresa: Company ID

        Returns:
            Tuple of (user_data dict, roles_list, company_areas_list)
        """
        try:
            # Use raw connection to handle multiple result sets
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute("EXEC SP_USUARIO_LOGIN_DATA @USUARIO = ?, @ID_EMPRESA = ?", (usuario, id_empresa))

                user_data = {}
                roles_data = []
                company_areas_data = []
                result_set_num = 1

                while True:
                    try:
                        # Check if we have columns (indicating data)
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            rows = cursor.fetchall()

                            if result_set_num == 2 and rows:
                                user_row = rows[0]
                                user_data = dict(zip(columns, user_row))
                            elif result_set_num == 3 and rows:
                                for row in rows:
                                    role_dict = dict(zip(columns, row))
                                    roles_data.append(role_dict)
                            elif result_set_num == 4 and rows:
                                for row in rows:
                                    area_dict = dict(zip(columns, row))
                                    company_areas_data.append(area_dict)

                    except Exception as fetch_error:
                        logger.error(f"Fetch error: {fetch_error}")

                    # Move to next result set
                    try:
                        if not cursor.nextset():
                            break
                    except Exception as nextset_error:
                        logger.error(f"Nextset error: {nextset_error}")
                        break

                    result_set_num += 1

                cursor.close()
                return user_data, roles_data, company_areas_data

            except Exception as cursor_error:
                logger.error(f"Cursor error: {cursor_error}")
                cursor.close()
                raise

        except Exception as e:
            logger.error(f"Error executing SP_USUARIO_LOGIN_DATA: {e}")
            raise

    @retry_on_db_error(max_retries=3, delay=1)
    def get_user_company_areas_sp(self, user_id: int, role_id: int) -> List[Dict[str, Any]]:
        """
        Execute SP_USUARIO_EMPR_AREA_LST and return company areas

        Args:
            user_id: User identifier
            role_id: Role identifier

        Returns:
            List of dictionaries with company area data
        """
        try:
            query = text("""
                EXEC SP_USUARIO_EMPR_AREA_LST
                @ID_USUARIO = :user_id,
                @ID_TIPO_ROL = :role_id
            """)

            result = self.db.execute(query, {
                'user_id': user_id,
                'role_id': role_id
            })

            columns = result.keys()
            rows = result.fetchall()
            result.close()

            company_areas = []
            for row in rows:
                area_dict = dict(zip(columns, row))
                company_areas.append(area_dict)

            return company_areas

        except Exception as e:
            logger.error(f"Error executing SP_USUARIO_EMPR_AREA_LST: {e}")
            raise

    @retry_on_db_error(max_retries=3, delay=1)
    def update_login_status(self, user_id: int) -> None:
        """
        Update user login status using SP_USUARIO_LOGIN_STATUS

        Args:
            user_id: User identifier
        """
        try:
            query = text("""
                EXEC SP_USUARIO_LOGIN_STATUS
                @ID_USUARIO = :user_id
            """)

            self.db.execute(query, {'user_id': user_id})
            self.db.commit()

        except Exception as e:
            logger.error(f"Error executing SP_USUARIO_LOGIN_STATUS for user {user_id}: {e}")
            self.db.rollback()
            raise

    @retry_on_db_error(max_retries=3, delay=1)
    def update_logout_status(self, user_id: int) -> None:
        """
        Update user logout status using SP_USUARIO_LOGOUT

        Args:
            user_id: User identifier
        """
        try:
            query = text("""
                EXEC SP_USUARIO_LOGOUT
                @ID_USUARIO = :user_id
            """)

            self.db.execute(query, {'user_id': user_id})
            self.db.commit()

        except Exception as e:
            logger.error(f"Error executing SP_USUARIO_LOGOUT for user {user_id}: {e}")
            self.db.rollback()
            raise