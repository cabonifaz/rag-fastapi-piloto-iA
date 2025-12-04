"""Service for managing users operations following hexagonal architecture."""

from typing import List, Dict, Any
import logging
from sqlalchemy.orm import Session
from app.infrastructure.repositories.users_repository import UsersRepository

logger = logging.getLogger(__name__)


class UsersService:
    """
    Service for users operations.
    Handles business logic for retrieving users.
    """

    def __init__(self):
        """Initialize stateless UsersService - no db parameter."""
        pass

    async def get_usuarios(
        self,
        db: Session,
        id_empresa: int
    ) -> List[Dict[str, Any]]:
        """
        Get all users for a company using stored procedure.

        Args:
            db: Database session
            id_empresa: Company ID

        Returns:
            List of dictionaries containing user information:
            - ID_USUARIO_EMPR_AREA: User Company Area ID
            - ID_USUARIO: User ID
            - USUARIO: Username
            - NOMBRES: First names
            - APELLIDOS: Last names
            - ID_ESTADO_REGISTRO: Record status
            - ID_EMPRESA: Company ID
            - ID_AREA: Area ID
            - AREA: Area name
            - ID_TIPO_ROL: Role type ID
            - ROL: Role name
            Empty list if query failed
        """
        try:
            # Create repository for this request
            repository = UsersRepository(db)

            # Validate input
            if not isinstance(id_empresa, int) or id_empresa <= 0:
                logger.error(f"Invalid id_empresa: {id_empresa}")
                return []

            # Use repository to get users with SP_USUARIOS_LST
            results = repository.get_usuarios(id_empresa=id_empresa)

            if results:
                logger.info(f"Users retrieved successfully: Company={id_empresa}, Count={len(results)}")
            else:
                logger.info(f"No users found for company: {id_empresa}")

            return results

        except Exception as e:
            logger.error(f"Error in get_usuarios service: {e}")
            raise

    async def create_usuario(
        self,
        db: Session,
        id_usuario: int,
        nuevo_usuario: str,
        password: str,
        nombres: str,
        apellidos: str,
        telefono: str,
        nuevo_rol: int,
        id_empresa: int,
        areas_string: str
    ) -> List[Dict[str, Any]]:
        """
        Create a new user using stored procedure.

        Args:
            db: Database session
            id_usuario: User ID
            nuevo_usuario: Username (max 100 chars)
            password: Password (max 100 chars)
            nombres: First names (max 100 chars)
            apellidos: Last names (max 100 chars)
            telefono: Phone number (max 15 chars)
            nuevo_rol: Role type ID (1=Super Admin, 2=Admin, 3=User)
            id_empresa: Company ID
            areas_string: Comma-separated area IDs (max 100 chars)

        Returns:
            List of dictionaries containing:
            - ID_TIPO_MENSAJE: Message type ID
            - MENSAJE: Status message
            Empty list if creation failed
        """
        try:
            # Create repository for this request
            repository = UsersRepository(db)

            # Validate input
            if not isinstance(id_usuario, int):
                logger.error(f"Invalid id_usuario: {id_usuario}")
                return []

            if not nuevo_usuario or len(nuevo_usuario.strip()) == 0:
                logger.error("Username cannot be empty")
                return []

            if not password or len(password.strip()) == 0:
                logger.error("Password cannot be empty")
                return []

            if not nombres or len(nombres.strip()) == 0:
                logger.error("First names cannot be empty")
                return []

            if not apellidos or len(apellidos.strip()) == 0:
                logger.error("Last names cannot be empty")
                return []

            if not isinstance(nuevo_rol, int) or nuevo_rol <= 0:
                logger.error(f"Invalid nuevo_rol: {nuevo_rol}")
                return []

            if not isinstance(id_empresa, int) or id_empresa <= 0:
                logger.error(f"Invalid id_empresa: {id_empresa}")
                return []

            if not areas_string or len(areas_string.strip()) == 0:
                logger.error("Areas string cannot be empty")
                return []

            # Trim inputs to match database constraints
            nuevo_usuario = nuevo_usuario.strip()[:100]
            password = password.strip()[:100]
            nombres = nombres.strip()[:100]
            apellidos = apellidos.strip()[:100]
            telefono = telefono.strip()[:15] if telefono else ""
            areas_string = areas_string.strip()[:100]

            # Use repository to create user with SP_CREATE_USUARIO
            results = repository.create_usuario(
                id_usuario=id_usuario,
                nuevo_usuario=nuevo_usuario,
                password=password,
                nombres=nombres,
                apellidos=apellidos,
                telefono=telefono,
                nuevo_rol=nuevo_rol,
                id_empresa=id_empresa,
                areas_string=areas_string
            )

            if results:
                logger.info(f"User created successfully: Usuario={nuevo_usuario}, ID_EMPRESA={id_empresa}, Results count={len(results)}")
            else:
                logger.warning(f"User creation returned no results: Usuario={nuevo_usuario}")

            return results

        except Exception as e:
            logger.error(f"Error in create_usuario service: {e}")
            raise

    async def update_usuario(
        self,
        db: Session,
        id_admin: int,
        id_usuario: int,
        usuario: str,
        nombres: str,
        apellidos: str,
        telefono: str = None
    ) -> List[Dict[str, Any]]:
        """
        Update user data using stored procedure.

        Args:
            db: Database session
            id_admin: Admin user ID performing the update
            id_usuario: User ID to update
            usuario: New username (max 200 chars)
            nombres: New first names (max 200 chars)
            apellidos: New last names (max 200 chars)
            telefono: Phone number (max 15 chars), optional (default None)

        Returns:
            List of dictionaries containing:
            - ID_TIPO_MENSAJE: Message type ID
            - MENSAJE: Status message
            Empty list if update failed
        """
        try:
            # Create repository for this request
            repository = UsersRepository(db)

            # Validate input
            if not isinstance(id_admin, int):
                logger.error(f"Invalid id_admin: {id_admin}")
                return []

            if not isinstance(id_usuario, int):
                logger.error(f"Invalid id_usuario: {id_usuario}")
                return []

            if not usuario or len(usuario.strip()) == 0:
                logger.error("Username cannot be empty")
                return []

            if not nombres or len(nombres.strip()) == 0:
                logger.error("First names cannot be empty")
                return []

            if not apellidos or len(apellidos.strip()) == 0:
                logger.error("Last names cannot be empty")
                return []

            # Trim inputs to match database constraints
            usuario = usuario.strip()[:200]
            nombres = nombres.strip()[:200]
            apellidos = apellidos.strip()[:200]
            telefono = telefono.strip()[:15]

            # Use repository to update user with SP_UPDATE_DATOS_USUARIO
            results = repository.update_usuario(
                id_admin=id_admin,
                id_usuario=id_usuario,
                usuario=usuario,
                nombres=nombres,
                apellidos=apellidos,
                telefono=telefono
            )

            if results:
                logger.info(f"User updated successfully: Usuario={usuario}, ID_USUARIO={id_usuario}, Results count={len(results)}")
            else:
                logger.warning(f"User update returned no results: ID_USUARIO={id_usuario}")

            return results

        except Exception as e:
            logger.error(f"Error in update_usuario service: {e}")
            raise

    async def update_usuario_status(
        self,
        db: Session,
        id_admin: int,
        id_usuario: int,
        status: int
    ) -> List[Dict[str, Any]]:
        """
        Update user status using stored procedure.

        Args:
            db: Database session
            id_admin: Admin user ID performing the update
            id_usuario: User ID to update
            status: New status (0 = inactive, 1 = active)

        Returns:
            List of dictionaries containing:
            - ID_TIPO_MENSAJE: Message type ID
            - MENSAJE: Status message
            Empty list if update failed
        """
        try:
            # Create repository for this request
            repository = UsersRepository(db)

            # Validate input
            if not isinstance(id_admin, int):
                logger.error(f"Invalid id_admin: {id_admin}")
                return []

            if not isinstance(id_usuario, int):
                logger.error(f"Invalid id_usuario: {id_usuario}")
                return []

            if not isinstance(status, int) or status not in [0, 1]:
                logger.error(f"Invalid status: {status}. Must be 0 (inactive) or 1 (active)")
                return []

            # Use repository to update user status with SP_UPDATE_USUARIO_STATUS
            results = repository.update_usuario_status(
                id_admin=id_admin,
                id_usuario=id_usuario,
                status=status
            )

            if results:
                status_text = "active" if status == 1 else "inactive"
                logger.info(f"User status updated successfully: ID_USUARIO={id_usuario}, Status={status_text}, Results count={len(results)}")
            else:
                logger.warning(f"User status update returned no results: ID_USUARIO={id_usuario}")

            return results

        except Exception as e:
            logger.error(f"Error in update_usuario_status service: {e}")
            raise

    async def update_usuario_password(
        self,
        db: Session,
        id_admin: int,
        id_usuario: int,
        clave_acceso: str
    ) -> List[Dict[str, Any]]:
        """
        Update user password using stored procedure.

        Args:
            db: Database session
            id_admin: Admin user ID performing the update
            id_usuario: User ID to update
            clave_acceso: New password (max 100 chars)

        Returns:
            List of dictionaries containing:
            - ID_TIPO_MENSAJE: Message type ID
            - MENSAJE: Status message
            Empty list if update failed
        """
        try:
            # Create repository for this request
            repository = UsersRepository(db)

            # Validate input
            if not isinstance(id_admin, int):
                logger.error(f"Invalid id_admin: {id_admin}")
                return []

            if not isinstance(id_usuario, int):
                logger.error(f"Invalid id_usuario: {id_usuario}")
                return []

            if not clave_acceso or len(clave_acceso.strip()) == 0:
                logger.error("Password cannot be empty")
                return []

            # Trim input to match database constraints
            clave_acceso = clave_acceso.strip()[:100]

            # Use repository to update user password with SP_UPDATE_USUARIO_PASSWORD
            results = repository.update_usuario_password(
                id_admin=id_admin,
                id_usuario=id_usuario,
                clave_acceso=clave_acceso
            )

            if results:
                logger.info(f"User password updated successfully: ID_USUARIO={id_usuario}, Results count={len(results)}")
            else:
                logger.warning(f"User password update returned no results: ID_USUARIO={id_usuario}")

            return results

        except Exception as e:
            logger.error(f"Error in update_usuario_password service: {e}")
            raise

    async def update_usuario_access(
        self,
        db: Session,
        id_admin: int,
        id_usuario: int,
        nuevo_rol: int,
        id_empresa: int,
        areas_string: str
    ) -> List[Dict[str, Any]]:
        """
        Update user role and areas access using stored procedure.

        Args:
            db: Database session
            id_admin: Admin user ID performing the update
            id_usuario: User ID to update
            nuevo_rol: New role type ID (1=Super Admin, 2=Admin, 3=User)
            id_empresa: Company ID
            areas_string: Comma-separated area IDs (max 100 chars)

        Returns:
            List of dictionaries containing:
            - ID_TIPO_MENSAJE: Message type ID
            - MENSAJE: Status message
            Empty list if update failed
        """
        try:
            # Create repository for this request
            repository = UsersRepository(db)

            # Validate input
            if not isinstance(id_admin, int):
                logger.error(f"Invalid id_admin: {id_admin}")
                return []

            if not isinstance(id_usuario, int):
                logger.error(f"Invalid id_usuario: {id_usuario}")
                return []

            if not isinstance(nuevo_rol, int) or nuevo_rol <= 0:
                logger.error(f"Invalid nuevo_rol: {nuevo_rol}")
                return []

            if not isinstance(id_empresa, int) or id_empresa <= 0:
                logger.error(f"Invalid id_empresa: {id_empresa}")
                return []

            if not areas_string or len(areas_string.strip()) == 0:
                logger.error("Areas string cannot be empty")
                return []

            # Trim input to match database constraints
            areas_string = areas_string.strip()[:100]

            # Use repository to update user access with SP_UPDATE_USUARIO_ACCESS
            results = repository.update_usuario_access(
                id_admin=id_admin,
                id_usuario=id_usuario,
                nuevo_rol=nuevo_rol,
                id_empresa=id_empresa,
                areas_string=areas_string
            )

            if results:
                logger.info(f"User access updated successfully: ID_USUARIO={id_usuario}, NUEVO_ROL={nuevo_rol}, AREAS={areas_string}, Results count={len(results)}")
            else:
                logger.warning(f"User access update returned no results: ID_USUARIO={id_usuario}")

            return results

        except Exception as e:
            logger.error(f"Error in update_usuario_access service: {e}")
            raise

    async def get_usuario_by_telefono(
        self,
        db: Session,
        id_agente: int,
        telefono: str
    ) -> List[Dict[str, Any]]:
        """
        Get user by phone number using stored procedure (N8N integration).

        Args:
            db: Database session
            id_agente: Agent ID
            telefono: Phone number (max 15 chars)

        Returns:
            List of dictionaries containing:
            - On failure (1 result set): ID_TIPO_MENSAJE, MENSAJE
            - On success (2 result sets): ID_TIPO_MENSAJE, MENSAJE + ID_USUARIO
            Empty list if query failed
        """
        try:
            # Create repository for this request
            repository = UsersRepository(db)

            # Validate input
            if not isinstance(id_agente, int) or id_agente <= 0:
                logger.error(f"Invalid id_agente: {id_agente}")
                return []

            if not telefono or len(telefono.strip()) == 0:
                logger.error("Phone number cannot be empty")
                return []

            # Trim input to match database constraints
            telefono = telefono.strip()[:15]

            # Use repository to get user by phone with SP_GET_USUARIO_BY_TELEFONO
            results = repository.get_usuario_by_telefono(
                id_agente=id_agente,
                telefono=telefono
            )

            if results:
                # Check if user was found (2 result sets means success)
                has_user_data = any('ID_USUARIO' in result for result in results)
                if has_user_data:
                    logger.info(f"User found by phone: Telefono={telefono}, Results count={len(results)}")
                else:
                    logger.info(f"User not found by phone: Telefono={telefono}, Results count={len(results)}")
            else:
                logger.warning(f"Get user by phone returned no results: Telefono={telefono}")

            return results

        except Exception as e:
            logger.error(f"Error in get_usuario_by_telefono service: {e}")
            raise
