"""Repository for user database operations - handles only DB calls and stored procedures."""

from sqlalchemy.orm import Session
from sqlalchemy import text, and_
from app.models.user_models import Usuario
from typing import Optional, List, Dict, Any, Tuple
import logging

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

    def verify_user_password_sp(self, usuario: str, password: str) -> int:
        """
        Execute SP_VERIFY_USER_PASS and return status

        Args:
            usuario: Username
            password: Plain text password

        Returns:
            Status code from SP (1 = valid, 0 = invalid)
        """
        try:
            query = text("""
                EXEC SP_VERIFY_USER_PASS
                @Username = :username,
                @Password = :password
            """)

            result = self.db.execute(query, {
                'username': usuario,
                'password': password
            })

            status_data = result.fetchone()
            result.close()

            if not status_data:
                return 0

            status_dict = dict(status_data._mapping) if hasattr(status_data, '_mapping') else dict(zip(result.keys(), status_data))
            return status_dict.get('Status', 0)

        except Exception as e:
            logger.error(f"Error executing SP_VERIFY_USER_PASS: {e}")
            raise

    def get_user_data_sp(self, usuario: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Execute SP_USUARIO_LOGIN and return all result sets

        The SP returns multiple result sets:
        - Result set 3: User data
        - Result set 4: Roles data
        - Result set 5: Company areas data

        Args:
            usuario: Username to retrieve data for

        Returns:
            Tuple of (user_data dict, roles_list, company_areas_list)
        """
        try:
            # Use raw connection to handle multiple result sets
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute("EXEC SP_USUARIO_LOGIN @USUARIO = ?", usuario)

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

                            if result_set_num == 3 and rows:  # User data
                                user_row = rows[0]
                                user_data = dict(zip(columns, user_row))
                            elif result_set_num == 4 and rows:  # Role data
                                for row in rows:
                                    role_dict = dict(zip(columns, row))
                                    roles_data.append(role_dict)
                            elif result_set_num == 5 and rows:  # Company Areas data
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
            logger.error(f"Error executing SP_USUARIO_LOGIN: {e}")
            raise

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

    # =============================================
    # Direct Table Query Operations
    # =============================================

    def get_user_by_id(self, user_id: int) -> Optional[Usuario]:
        """
        Query USUARIO table by ID

        Args:
            user_id: User identifier

        Returns:
            Usuario model instance or None
        """
        try:
            user = self.db.query(Usuario).filter(Usuario.ID_USUARIO == user_id).first()
            return user
        except Exception as e:
            logger.error(f"Error querying user by ID {user_id}: {e}")
            raise

    def get_active_user_by_id(self, user_id: int) -> Optional[Usuario]:
        """
        Query USUARIO table by ID with active status filter

        Args:
            user_id: User identifier

        Returns:
            Usuario model instance or None
        """
        try:
            user = self.db.query(Usuario).filter(
                and_(
                    Usuario.ID_USUARIO == user_id,
                    Usuario.ID_ESTADO_REGISTRO == 1
                )
            ).first()
            return user
        except Exception as e:
            logger.error(f"Error querying active user by ID {user_id}: {e}")
            raise

    def update_connection_status(self, user_id: int, connected: bool) -> bool:
        """
        Update user connection status in USUARIO table

        Args:
            user_id: User identifier
            connected: True for logged in, False for logged out

        Returns:
            True if successful, False otherwise
        """
        try:
            user = self.db.query(Usuario).filter(Usuario.ID_USUARIO == user_id).first()
            if user:
                user.ID_CONECTADO = connected
                self.db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating connection status for user {user_id}: {e}")
            self.db.rollback()
            raise
