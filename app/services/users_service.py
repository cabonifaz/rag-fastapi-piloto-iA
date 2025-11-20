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
        nuevo_usuario: str,
        password: str,
        nombres: str,
        apellidos: str,
        id_tipo_rol: int,
        id_empresa: int,
        areas_string: str
    ) -> List[Dict[str, Any]]:
        """
        Create a new user using stored procedure.

        Args:
            db: Database session
            nuevo_usuario: Username (max 100 chars)
            password: Password (max 100 chars)
            nombres: First names (max 100 chars)
            apellidos: Last names (max 100 chars)
            id_tipo_rol: Role type ID (1=Super Admin, 2=Admin, 3=User)
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

            if not isinstance(id_tipo_rol, int) or id_tipo_rol <= 0:
                logger.error(f"Invalid id_tipo_rol: {id_tipo_rol}")
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
            areas_string = areas_string.strip()[:100]

            # Use repository to create user with SP_USUARIO_CREATE
            results = repository.create_usuario(
                nuevo_usuario=nuevo_usuario,
                password=password,
                nombres=nombres,
                apellidos=apellidos,
                id_tipo_rol=id_tipo_rol,
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
