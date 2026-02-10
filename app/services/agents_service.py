"""Service for managing agents operations following hexagonal architecture."""

from typing import List, Dict, Any, Optional
import logging
from sqlalchemy.orm import Session
from app.infrastructure.repositories.agents_repository import AgentsRepository
from app.utils.agent_jwt_auth import AgentJWTAuth

logger = logging.getLogger(__name__)


class AgentsService:
    """
    Service for agents operations.
    Handles business logic for retrieving agents.
    """

    def __init__(self):
        """Initialize stateless AgentsService - no db parameter."""
        pass


    async def create_agente(
        self,
        db: Session,
        id_usuario: int,
        numero_telf: str,
        codigo_pais: str,
        id_tipo_agente: int,
        id_empresa: int,
        acceso_general: int,
        areas_string: str
    ) -> List[Dict[str, Any]]:
        """
        Create a new agent using stored procedure.

        Args:
            db: Database session
            id_usuario: User ID
            numero_telf: Phone number (max 20 chars)
            codigo_pais: Country code composite (max 8 chars, e.g., '51-PE')
            id_tipo_agente: Agent type ID
            id_empresa: Company ID
            acceso_general: General access flag
            areas_string: Comma-separated area IDs (max 100 chars)

        Returns:
            List of dictionaries containing:
            - ID_TIPO_MENSAJE: Message type ID
            - MENSAJE: Status message
            Empty list if creation failed
        """
        try:
            # Create repository for this request
            repository = AgentsRepository(db)

            # Validate input
            if not isinstance(id_usuario, int):
                logger.error(f"Invalid id_usuario: {id_usuario}")
                return []

            if not numero_telf or len(numero_telf.strip()) == 0:
                logger.error("Phone number cannot be empty")
                return []

            if not codigo_pais or len(codigo_pais.strip()) == 0:
                logger.error("Country code cannot be empty")
                return []

            if not isinstance(id_tipo_agente, int) or id_tipo_agente <= 0:
                logger.error(f"Invalid id_tipo_agente: {id_tipo_agente}")
                return []

            if not isinstance(id_empresa, int) or id_empresa <= 0:
                logger.error(f"Invalid id_empresa: {id_empresa}")
                return []

            if not isinstance(acceso_general, int) or acceso_general not in [0, 1]:
                logger.error(f"Invalid acceso_general: {acceso_general}. Must be 0 or 1")
                return []

            if not areas_string or len(areas_string.strip()) == 0:
                logger.error("Areas string cannot be empty")
                return []

            # Trim inputs to match database constraints
            numero_telf = numero_telf.strip()[:20]
            codigo_pais = codigo_pais.strip()[:8]
            areas_string = areas_string.strip()[:100]

            # Use repository to create agent with SP_CREATE_AGENTE
            results = repository.create_agente(
                id_usuario=id_usuario,
                numero_telf=numero_telf,
                codigo_pais=codigo_pais,
                id_tipo_agente=id_tipo_agente,
                id_empresa=id_empresa,
                acceso_general=acceso_general,
                areas_string=areas_string
            )

            if results:
                logger.info(f"Agent created successfully: Telefono={numero_telf}, ID_EMPRESA={id_empresa}, Results count={len(results)}")
            else:
                logger.warning(f"Agent creation returned no results: Telefono={numero_telf}")

            return results

        except Exception as e:
            logger.error(f"Error in create_agente service: {e}")
            raise

    async def get_agentes_paginated(
        self,
        db: Session,
        id_usuario: int,
        id_empresa: int,
        num_pagina: int = 1,
        tam_pagina: int = 10,
        term_busqueda: str = None,
        campo_orden: str = 'NUMERO_TELF',
        dir_orden: str = 'ASC',
        filtro_estado: int = None,
        filtro_operativo: int = None
    ) -> Dict[str, Any]:
        """
        Get paginated agents using stored procedure.

        Args:
            db: Database session
            id_usuario: User ID for role validation
            id_empresa: Company ID
            num_pagina: Page number (starting at 1)
            tam_pagina: Number of rows per page
            term_busqueda: Optional search term for NUMERO_TELF, AREA
            campo_orden: Field to sort by (default 'NUMERO_TELF')
            dir_orden: Sort direction ASC or DESC (default 'ASC')
            filtro_estado: Optional filter for ID_ESTADO_REGISTRO (None=all, 1=active, 0=inactive)
            filtro_operativo: Optional filter for ESTADO_OPERATIVO (None=all, 1=operative, 0=inoperative)

        Returns:
            Dictionary with:
                - data: List of agent dictionaries
                - pagination: Dictionary with pagination metadata (total_records, current_page, page_size, total_pages)
                - message_result: Dict with ID_TIPO_MENSAJE and MENSAJE if error/authorization
            Empty dict with empty list if fetch failed
        """
        try:
            # Create repository for this request
            repository = AgentsRepository(db)

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

            if campo_orden not in ['NUMERO_TELF', 'ID_TIPO_AGENTE', 'ACCESO_GENERAL', 'ESTADO_OPERATIVO', 'AREA', 'ID_ESTADO_REGISTRO']:
                logger.warning(f"Invalid campo_orden: {campo_orden}, using default 'NUMERO_TELF'")
                campo_orden = 'NUMERO_TELF'

            if dir_orden not in ['ASC', 'DESC']:
                logger.warning(f"Invalid dir_orden: {dir_orden}, using default 'ASC'")
                dir_orden = 'ASC'

            if filtro_estado is not None and filtro_estado not in [0, 1]:
                logger.warning(f"Invalid filtro_estado: {filtro_estado}, using None")
                filtro_estado = None

            if filtro_operativo is not None and filtro_operativo not in [0, 1]:
                logger.warning(f"Invalid filtro_operativo: {filtro_operativo}, using None")
                filtro_operativo = None

            # Trim search term if provided
            if term_busqueda:
                term_busqueda = term_busqueda.strip()[:200]

            # Use repository to get paginated agents with SP_AGENTES_LST_PAG
            result = repository.get_agentes_paginated(
                id_usuario=id_usuario,
                id_empresa=id_empresa,
                num_pagina=num_pagina,
                tam_pagina=tam_pagina,
                term_busqueda=term_busqueda,
                campo_orden=campo_orden,
                dir_orden=dir_orden,
                filtro_estado=filtro_estado,
                filtro_operativo=filtro_operativo
            )

            # Log results
            if result.get('message_result'):
                # Authorization or error message
                msg_type = result['message_result'].get('ID_TIPO_MENSAJE')
                mensaje = result['message_result'].get('MENSAJE')
                logger.warning(f"Agents pagination returned message: Type={msg_type}, Message={mensaje}")
            elif result.get('data'):
                # Successful pagination
                pagination = result.get('pagination', {})
                logger.info(
                    f"Agents paginated successfully: Company={id_empresa}, "
                    f"Page={pagination.get('current_page')}/{pagination.get('total_pages')}, "
                    f"Records={len(result['data'])}/{pagination.get('total_records')}, "
                    f"Search='{term_busqueda}', Order={campo_orden} {dir_orden}, "
                    f"StatusFilter={filtro_estado}, OperativeFilter={filtro_operativo}"
                )
            else:
                logger.info(f"No agents found for company: {id_empresa}")

            return result

        except Exception as e:
            logger.error(f"Error in get_agentes_paginated service: {e}")
            raise
    
    async def update_datos_agente(
        self,
        db: Session,
        id_usuario: int,
        id_agente: int,
        numero_telf: str,
        codigo_pais: str,
        id_tipo_agente: int,
        acceso_general: int
    ) -> List[Dict[str, Any]]:
        """
        Update agent data using stored procedure.

        Args:
            db: Database session
            id_usuario: User ID performing the update
            id_agente: Agent ID to update
            numero_telf: Phone number (max 20 chars)
            codigo_pais: Country code (max 8 chars)
            id_tipo_agente: Agent type ID
            acceso_general: General access flag (0 or 1)

        Returns:
            List of dictionaries containing:
            - ID_TIPO_MENSAJE: Message type ID
            - MENSAJE: Status message
            Empty list if update failed
        """
        try:
            # Create repository for this request
            repository = AgentsRepository(db)

            # Validate input
            if not isinstance(id_usuario, int):
                logger.error(f"Invalid id_usuario: {id_usuario}")
                return []

            if not isinstance(id_agente, int):
                logger.error(f"Invalid id_agente: {id_agente}")
                return []

            if not numero_telf or len(numero_telf.strip()) == 0:
                logger.error("Phone number cannot be empty")
                return []

            if not codigo_pais or len(codigo_pais.strip()) == 0:
                logger.error("Country code cannot be empty")
                return []

            if not isinstance(id_tipo_agente, int) or id_tipo_agente <= 0:
                logger.error(f"Invalid id_tipo_agente: {id_tipo_agente}")
                return []

            if not isinstance(acceso_general, int) or acceso_general not in [0, 1]:
                logger.error(f"Invalid acceso_general: {acceso_general}. Must be 0 or 1")
                return []

            # Trim inputs to match database constraints
            numero_telf = numero_telf.strip()[:20]
            codigo_pais = codigo_pais.strip()[:8]

            # Use repository to update agent with SP_UPDATE_DATOS_AGENTE
            results = repository.update_datos_agente(
                id_usuario=id_usuario,
                id_agente=id_agente,
                numero_telf=numero_telf,
                codigo_pais=codigo_pais,
                id_tipo_agente=id_tipo_agente,
                acceso_general=acceso_general
            )

            if results:
                logger.info(f"Agent updated successfully: ID_AGENTE={id_agente}, Results count={len(results)}")
            else:
                logger.warning(f"Agent update returned no results: ID_AGENTE={id_agente}")

            return results

        except Exception as e:
            logger.error(f"Error in update_datos_agente service: {e}")
            raise

    async def update_agente_status(
        self,
        db: Session,
        id_usuario: int,
        id_agente: int,
        status: int
    ) -> List[Dict[str, Any]]:
        """
        Update agent status using stored procedure.

        Args:
            db: Database session
            id_usuario: User ID performing the update
            id_agente: Agent ID to update
            status: New status (0 = inactive, 1 = active)

        Returns:
            List of dictionaries containing:
            - ID_TIPO_MENSAJE: Message type ID
            - MENSAJE: Status message
            Empty list if update failed
        """
        try:
            # Create repository for this request
            repository = AgentsRepository(db)

            # Validate input
            if not isinstance(id_usuario, int):
                logger.error(f"Invalid id_usuario: {id_usuario}")
                return []

            if not isinstance(id_agente, int):
                logger.error(f"Invalid id_agente: {id_agente}")
                return []

            if not isinstance(status, int) or status not in [0, 1]:
                logger.error(f"Invalid status: {status}. Must be 0 (inactive) or 1 (active)")
                return []

            # Use repository to update agent status with SP_UPDATE_AGENTE_STATUS
            results = repository.update_agente_status(
                id_usuario=id_usuario,
                id_agente=id_agente,
                status=status
            )

            if results:
                status_text = "active" if status == 1 else "inactive"
                logger.info(f"Agent status updated successfully: ID_AGENTE={id_agente}, Status={status_text}, Results count={len(results)}")
            else:
                logger.warning(f"Agent status update returned no results: ID_AGENTE={id_agente}")

            return results

        except Exception as e:
            logger.error(f"Error in update_agente_status service: {e}")
            raise

    async def update_agente_operativo(
        self,
        db: Session,
        id_usuario: int,
        id_agente: int,
        operativo: int
    ) -> List[Dict[str, Any]]:
        """
        Update agent operative status using stored procedure.

        Args:
            db: Database session
            id_usuario: User ID performing the update
            id_agente: Agent ID to update
            operativo: Operative status (0 = inoperative, 1 = operative)

        Returns:
            List of dictionaries containing:
            - ID_TIPO_MENSAJE: Message type ID
            - MENSAJE: Status message
            Empty list if update failed
        """
        try:
            # Create repository for this request
            repository = AgentsRepository(db)

            # Validate input
            if not isinstance(id_usuario, int):
                logger.error(f"Invalid id_usuario: {id_usuario}")
                return []

            if not isinstance(id_agente, int):
                logger.error(f"Invalid id_agente: {id_agente}")
                return []

            if not isinstance(operativo, int) or operativo not in [0, 1]:
                logger.error(f"Invalid operativo: {operativo}. Must be 0 (inoperative) or 1 (operative)")
                return []

            # Use repository to update agent operative status with SP_UPDATE_AGENTE_OPERATIVO
            results = repository.update_agente_operativo(
                id_usuario=id_usuario,
                id_agente=id_agente,
                operativo=operativo
            )

            if results:
                operativo_text = "operative" if operativo == 1 else "inoperative"
                logger.info(f"Agent operative status updated successfully: ID_AGENTE={id_agente}, Operativo={operativo_text}, Results count={len(results)}")
            else:
                logger.warning(f"Agent operative status update returned no results: ID_AGENTE={id_agente}")

            return results

        except Exception as e:
            logger.error(f"Error in update_agente_operativo service: {e}")
            raise

    async def update_agente_secret_key(
        self,
        db: Session,
        id_usuario: int,
        id_agente: int
    ) -> List[Dict[str, Any]]:
        """
        Update agent secret key using stored procedure.

        Args:
            db: Database session
            id_usuario: User ID performing the update
            id_agente: Agent ID to regenerate secret key for

        Returns:
            List of dictionaries containing:
            - ID_TIPO_MENSAJE: Message type ID
            - MENSAJE: Status message
            Empty list if update failed
        """
        try:
            # Create repository for this request
            repository = AgentsRepository(db)

            # Validate input
            if not isinstance(id_usuario, int):
                logger.error(f"Invalid id_usuario: {id_usuario}")
                return []

            if not isinstance(id_agente, int):
                logger.error(f"Invalid id_agente: {id_agente}")
                return []

            # Use repository to update agent secret key with SP_UPDATE_AGENTE_SECRET_KEY
            results = repository.update_agente_secret_key(
                id_usuario=id_usuario,
                id_agente=id_agente
            )

            if results:
                logger.info(f"Agent secret key updated successfully: ID_AGENTE={id_agente}, Results count={len(results)}")
            else:
                logger.warning(f"Agent secret key update returned no results: ID_AGENTE={id_agente}")

            return results

        except Exception as e:
            logger.error(f"Error in update_agente_secret_key service: {e}")
            raise

    async def update_agente_access(
        self,
        db: Session,
        id_usuario: int,
        id_agente: int,
        id_empresa: int,
        areas_string: str
    ) -> List[Dict[str, Any]]:
        """
        Update agent area access using stored procedure.

        Args:
            db: Database session
            id_usuario: User ID performing the update
            id_agente: Agent ID to update
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
            repository = AgentsRepository(db)

            # Validate input
            if not isinstance(id_usuario, int):
                logger.error(f"Invalid id_usuario: {id_usuario}")
                return []

            if not isinstance(id_agente, int):
                logger.error(f"Invalid id_agente: {id_agente}")
                return []

            if not isinstance(id_empresa, int) or id_empresa <= 0:
                logger.error(f"Invalid id_empresa: {id_empresa}")
                return []

            if not areas_string or len(areas_string.strip()) == 0:
                logger.error("Areas string cannot be empty")
                return []

            # Trim input to match database constraints
            areas_string = areas_string.strip()[:100]

            # Use repository to update agent access with SP_UPDATE_AGENTE_ACCESS
            results = repository.update_agente_access(
                id_usuario=id_usuario,
                id_agente=id_agente,
                id_empresa=id_empresa,
                areas_string=areas_string
            )

            if results:
                logger.info(f"Agent access updated successfully: ID_AGENTE={id_agente}, ID_EMPRESA={id_empresa}, AREAS={areas_string}, Results count={len(results)}")
            else:
                logger.warning(f"Agent access update returned no results: ID_AGENTE={id_agente}")

            return results

        except Exception as e:
            logger.error(f"Error in update_agente_access service: {e}")
            raise

    async def verify_acceso_agente(
        self,
        db: Session,
        numero_telf: str,
        secret_key: str
    ) -> Dict[str, Any]:
        """
        Verify agent access credentials and generate JWT token on success.

        Args:
            db: Database session
            numero_telf: Phone number (max 20 chars)
            secret_key: Secret key (max 64 chars)

        Returns:
            On success (ID_TIPO_MENSAJE = 2):
                {
                    'token': str,
                    'id_tipo_mensaje': int,
                    'mensaje': str
                }
            On failure:
                {
                    'id_tipo_mensaje': int,
                    'mensaje': str
                }
            Empty dict on error
        """
        try:
            # Create repository for this request
            repository = AgentsRepository(db)

            # Validate input
            if not numero_telf or len(numero_telf.strip()) == 0:
                logger.error("Phone number cannot be empty")
                return {
                    'id_tipo_mensaje': 1,
                    'mensaje': 'Número de teléfono requerido'
                }

            if not secret_key or len(secret_key.strip()) == 0:
                logger.error("Secret key cannot be empty")
                return {
                    'id_tipo_mensaje': 1,
                    'mensaje': 'Secret key requerido'
                }

            # Trim inputs to match database constraints
            numero_telf = numero_telf.strip()[:20]
            secret_key = secret_key.strip()[:64]

            # Use repository to verify agent access with SP_VERIFY_ACCESO_AGENTE
            result = repository.verify_acceso_agente(
                numero_telf=numero_telf,
                secret_key=secret_key
            )

            # Extract mensaje info
            if not result or not result.get('mensaje'):
                logger.warning(f"Agent verification returned no results: Telefono={numero_telf}")
                return {
                    'id_tipo_mensaje': 1,
                    'mensaje': 'Error en la verificación'
                }

            mensaje_info = result['mensaje']
            id_tipo_mensaje = mensaje_info.get('ID_TIPO_MENSAJE')
            mensaje = mensaje_info.get('MENSAJE')

            logger.info(f"Agent verification completed: Telefono={numero_telf}, ID_TIPO_MENSAJE={id_tipo_mensaje}, Mensaje={mensaje}")

            # Check if authentication was successful (ID_TIPO_MENSAJE = 2)
            if id_tipo_mensaje == 2:
                # Generate JWT token
                jwt_token = AgentJWTAuth.create_agent_jwt_token(result)

                return {
                    'token': jwt_token,
                    'id_tipo_mensaje': id_tipo_mensaje,
                    'mensaje': mensaje
                }
            else:
                # Authentication failed
                return {
                    'id_tipo_mensaje': id_tipo_mensaje,
                    'mensaje': mensaje
                }

        except Exception as e:
            logger.error(f"Error in verify_acceso_agente service: {e}")
            return {
                'id_tipo_mensaje': 1,
                'mensaje': 'Error interno del servidor'
            }