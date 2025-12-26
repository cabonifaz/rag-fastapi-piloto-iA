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

    async def get_agentes(
        self,
        db: Session,
        id_empresa: int
    ) -> List[Dict[str, Any]]:
        """
        Get all agents for a company using stored procedure.

        Args:
            db: Database session
            id_empresa: Company ID

        Returns:
            List of dictionaries containing agent information:
            - ID_AGENTE_EMPR_AREA: Agent Company Area ID
            - ID_AGENTE: Agent ID
            - NUMERO_TELF: Phone number
            - ID_TIPO_AGENTE: Agent type ID
            - ACCESO_GENERAL: General access flag
            - ESTADO_OPERATIVO: Operational status
            - ID_ESTADO_REGISTRO: Record status
            - ID_EMPRESA: Company ID
            - ID_AREA: Area ID
            - AREA: Area name
            Empty list if query failed
        """
        try:
            # Create repository for this request
            repository = AgentsRepository(db)

            # Validate input
            if not isinstance(id_empresa, int) or id_empresa <= 0:
                logger.error(f"Invalid id_empresa: {id_empresa}")
                return []

            # Use repository to get agents with SP_AGENTES_LST
            results = repository.get_agentes(id_empresa=id_empresa)

            if results:
                logger.info(f"Agents retrieved successfully: Company={id_empresa}, Count={len(results)}")
            else:
                logger.info(f"No agents found for company: {id_empresa}")

            return results

        except Exception as e:
            logger.error(f"Error in get_agentes service: {e}")
            raise

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