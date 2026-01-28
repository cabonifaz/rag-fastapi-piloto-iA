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


    async def create_usuario(
        self,
        db: Session,
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
        Create a new user using stored procedure.

        Args:
            db: Database session
            id_usuario: User ID
            nuevo_usuario: Username (max 100 chars)
            password: Password (max 100 chars)
            nombres: First names (max 100 chars)
            apellidos: Last names (max 100 chars)
            codigo_pais: Country code (max 8 chars)
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
            codigo_pais = codigo_pais.strip()[:8]
            telefono = telefono.strip()[:15]
            areas_string = areas_string.strip()[:100]

            # Use repository to create user with SP_CREATE_USUARIO
            results = repository.create_usuario(
                id_usuario=id_usuario,
                nuevo_usuario=nuevo_usuario,
                password=password,
                nombres=nombres,
                apellidos=apellidos,
                codigo_pais=codigo_pais,
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
        Get complete user data for n8n integration using stored procedure.

        Args:
            db: Database session
            id_agente: Agent ID
            telefono: Phone number (max 15 chars)

        Returns:
            List of dictionaries containing:
            - On failure (1 result set):
                * ID_TIPO_MENSAJE, MENSAJE
            - On success (5 result sets):
                * ID_TIPO_MENSAJE, MENSAJE (status message)
                * ID_USUARIO, USUARIO, ... (user data)
                * ID_EMPRESA, ID_AREA (company and area IDs)
                * ID_IA_AREA (IA area configuration ID)
                * ID_CHAT (existing chat ID)
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

            # Use repository to get complete user data with SP_GET_USER_DATA_FOR_N8N
            results = repository.get_user_data_for_n8n(
                id_agente=id_agente,
                telefono=telefono
            )

            if results:
                # Check if user was found (user data result set present)
                has_user_data = any('ID_USUARIO' in result for result in results)
                has_company_area = any('ID_EMPRESA' in result and 'ID_AREA' in result for result in results)
                has_ia_area = any('ID_IA_AREA' in result for result in results)
                has_chat = any('ID_CHAT' in result for result in results)

                if has_user_data:
                    logger.info(
                        f"User data found by phone: Telefono={telefono}, "
                        f"Results count={len(results)}, "
                        f"Has company/area={has_company_area}, "
                        f"Has IA area={has_ia_area}, "
                        f"Has chat={has_chat}"
                    )
                else:
                    logger.info(f"User not found by phone: Telefono={telefono}, Results count={len(results)}")
            else:
                logger.warning(f"Get user data by phone returned no results: Telefono={telefono}")

            return results

        except Exception as e:
            logger.error(f"Error in get_usuario_by_telefono service: {e}")
            raise


    async def get_usuarios_paginated(
        self,
        db: Session,
        id_usuario: int,
        id_empresa: int,
        num_pagina: int = 1,
        tam_pagina: int = 10,
        term_busqueda: str = None,
        campo_orden: str = 'APELLIDOS',
        dir_orden: str = 'ASC',
        filtro_estado: int = None
    ) -> Dict[str, Any]:
        """
        Get paginated users using stored procedure.

        Args:
            db: Database session
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
            # Create repository for this request
            repository = UsersRepository(db)

            # Validate input
            if not isinstance(id_usuario, int) or id_usuario <= 0:
                logger.error(f"Invalid id_usuario: {id_usuario}")
                return {'data': [], 'pagination': {}, 'message_result': None}

            if not isinstance(id_empresa, int) or id_empresa <= 0:
                logger.error(f"Invalid id_empresa: {id_empresa}")
                return {'data': [], 'pagination': {}, 'message_result': None}

            if not isinstance(num_pagina, int) or num_pagina < 1:
                logger.warning(f"Invalid num_pagina: {num_pagina}, using default 1")
                num_pagina = 1

            if not isinstance(tam_pagina, int) or tam_pagina < 1:
                logger.warning(f"Invalid tam_pagina: {tam_pagina}, using default 10")
                tam_pagina = 10

            if campo_orden not in ['USUARIO', 'NOMBRES', 'APELLIDOS', 'TELEFONO', 'AREA', 'ROL', 'FCHCRE', 'ID_ESTADO_REGISTRO']:
                logger.warning(f"Invalid campo_orden: {campo_orden}, using default 'APELLIDOS'")
                campo_orden = 'APELLIDOS'

            if dir_orden not in ['ASC', 'DESC']:
                logger.warning(f"Invalid dir_orden: {dir_orden}, using default 'ASC'")
                dir_orden = 'ASC'

            if filtro_estado is not None and filtro_estado not in [0, 1]:
                logger.warning(f"Invalid filtro_estado: {filtro_estado}, using None")
                filtro_estado = None

            # Trim search term if provided
            if term_busqueda:
                term_busqueda = term_busqueda.strip()[:200]

            # Use repository to get paginated users with SP_USUARIOS_LST_PAG
            result = repository.get_usuarios_paginated(
                id_usuario=id_usuario,
                id_empresa=id_empresa,
                num_pagina=num_pagina,
                tam_pagina=tam_pagina,
                term_busqueda=term_busqueda,
                campo_orden=campo_orden,
                dir_orden=dir_orden,
                filtro_estado=filtro_estado
            )

            # Log results
            if result.get('message_result'):
                # Authorization or error message
                msg_type = result['message_result'].get('ID_TIPO_MENSAJE')
                mensaje = result['message_result'].get('MENSAJE')
                logger.warning(f"Users pagination returned message: Type={msg_type}, Message={mensaje}")
            elif result.get('data'):
                # Successful pagination
                pagination = result.get('pagination', {})
                logger.info(
                    f"Users paginated successfully: Company={id_empresa}, "
                    f"Page={pagination.get('current_page')}/{pagination.get('total_pages')}, "
                    f"Records={len(result['data'])}/{pagination.get('total_records')}, "
                    f"Search='{term_busqueda}', Order={campo_orden} {dir_orden}, Status={filtro_estado}"
                )
            else:
                logger.info(f"No users found for company: {id_empresa}")

            return result

        except Exception as e:
            logger.error(f"Error in get_usuarios_paginated service: {e}")
            raise