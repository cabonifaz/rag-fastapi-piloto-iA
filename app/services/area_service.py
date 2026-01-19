"""Service for managing area operations following hexagonal architecture."""

from typing import List, Dict, Any, Optional
import logging
from sqlalchemy.orm import Session
from app.infrastructure.repositories.area_repository import AreaRepository

logger = logging.getLogger(__name__)


class AreaService:
    """
    Service for area operations.
    Handles business logic for creating areas.
    """

    def __init__(self):
        """Initialize stateless AreaService - no db parameter."""
        pass

    async def create_area(
        self,
        db: Session,
        id_usuario: int,
        id_empresa: int,
        area: str
    ) -> List[Dict[str, Any]]:
        """
        Create a new area base using stored procedure.

        Args:
            db: Database session
            id_usuario: User ID creating the area
            id_empresa: Company ID
            area: Area name (max 100 chars)

        Returns:
            List of dictionaries containing:
            - ID_TIPO_MENSAJE: Message type ID
            - MENSAJE: Status message
            Empty list if creation failed
        """
        try:
            # Create repository for this request
            repository = AreaRepository(db)

            # Validate input
            if not area or len(area.strip()) == 0:
                logger.error("Area cannot be empty")
                return []

            # Trim inputs to match database constraints
            area = area.strip()[:100]

            # Use repository to create area with SP_CREATE_AREA_BASE
            results = repository.create_area(
                id_usuario=id_usuario,
                id_empresa=id_empresa,
                area=area
            )

            if results:
                logger.info(f"Area created successfully: Area={area}, Results count={len(results)}")
            else:
                logger.warning(f"Area creation returned no results: Area={area}")

            return results

        except Exception as e:
            logger.error(f"Error in create_area service: {e}")
            raise

    async def update_area_status(
        self,
        db: Session,
        id_usuario: int,
        id_empresa: int,
        id_area: int,
        status: int
    ) -> List[Dict[str, Any]]:
        """
        Update area status (activate or deactivate).

        Args:
            db: Database session
            id_usuario: User ID performing the update
            id_empresa: Company ID
            id_area: Area ID to update
            status: New status value (0 = inactive, 1 = active)

        Returns:
            List of dictionaries containing:
            - ID_TIPO_MENSAJE: Message type ID
            - MENSAJE: Status message
            Empty list if update failed
        """
        try:
            # Create repository for this request
            repository = AreaRepository(db)

            # Validate input
            if not isinstance(id_area, int) or id_area <= 0:
                logger.error(f"Invalid id_area: {id_area}")
                return []

            if not isinstance(id_usuario, int) or id_usuario <= 0:
                logger.error(f"Invalid id_usuario: {id_usuario}")
                return []

            if not isinstance(id_empresa, int) or id_empresa <= 0:
                logger.error(f"Invalid id_empresa: {id_empresa}")
                return []

            if status not in [0, 1]:
                logger.error(f"Invalid status: {status}. Must be 0 or 1")
                return []

            # Use repository to update area status
            results = repository.update_area_status(
                id_usuario=id_usuario,
                id_empresa=id_empresa,
                id_area=id_area,
                status=status
            )

            if results:
                logger.info(f"Area status updated successfully: ID_AREA={id_area}, STATUS={status}, ID_USUARIO={id_usuario}, ID_EMPRESA={id_empresa}, Results count={len(results)}")
            else:
                logger.warning(f"Area status update returned no results: ID_AREA={id_area}")

            return results

        except Exception as e:
            logger.error(f"Error in update_area_status service: {e}")
            raise

    async def update_area_nombre(
        self,
        db: Session,
        id_usuario: int,
        id_empresa: int,
        id_area: int,
        area: str
    ) -> List[Dict[str, Any]]:
        """
        Update area name.

        Args:
            db: Database session
            id_usuario: User ID performing the update
            id_empresa: Company ID
            id_area: Area ID to update
            area: New area name (max 200 chars)

        Returns:
            List of dictionaries containing:
            - ID_TIPO_MENSAJE: Message type ID
            - MENSAJE: Status message
            Empty list if update failed
        """
        try:
            # Create repository for this request
            repository = AreaRepository(db)

            # Validate input
            if not isinstance(id_area, int) or id_area <= 0:
                logger.error(f"Invalid id_area: {id_area}")
                return []

            if not isinstance(id_usuario, int) or id_usuario <= 0:
                logger.error(f"Invalid id_usuario: {id_usuario}")
                return []

            if not isinstance(id_empresa, int) or id_empresa <= 0:
                logger.error(f"Invalid id_empresa: {id_empresa}")
                return []

            if not area or len(area.strip()) == 0:
                logger.error("Area name cannot be empty")
                return []

            # Trim area name to match database constraints
            area = area.strip()[:200]

            # Use repository to update area name
            results = repository.update_area_nombre(
                id_usuario=id_usuario,
                id_empresa=id_empresa,
                id_area=id_area,
                area=area
            )

            if results:
                logger.info(f"Area name updated successfully: ID_AREA={id_area}, AREA={area}, ID_USUARIO={id_usuario}, ID_EMPRESA={id_empresa}, Results count={len(results)}")
            else:
                logger.warning(f"Area name update returned no results: ID_AREA={id_area}")

            return results

        except Exception as e:
            logger.error(f"Error in update_area_nombre service: {e}")
            raise

    async def get_areas(
        self,
        db: Session,
        id_empresa: int
    ) -> List[Dict[str, Any]]:
        """
        Get all areas for a company using stored procedure.

        Args:
            db: Database session
            id_empresa: Company ID

        Returns:
            List of dictionaries containing area information:
            - ID_AREA: Area ID
            - ID_EMPRESA: Company ID
            - AREA: Area name
            - FCHCRE: Creation date
            - ID_ESTADO_REGISTRO: Record status
            Empty list if query failed
        """
        try:
            # Create repository for this request
            repository = AreaRepository(db)

            # Validate input
            if not isinstance(id_empresa, int) or id_empresa <= 0:
                logger.error(f"Invalid id_empresa: {id_empresa}")
                return []

            # Use repository to get areas with SP_AREAS_LST
            results = repository.get_areas(id_empresa=id_empresa)

            if results:
                logger.info(f"Areas retrieved successfully: Company={id_empresa}, Count={len(results)}")
            else:
                logger.info(f"No areas found for company: {id_empresa}")

            return results

        except Exception as e:
            logger.error(f"Error in get_areas service: {e}")
            raise

    async def get_areas_paginated(
        self,
        db: Session,
        id_usuario: int,
        id_empresa: int,
        num_pagina: int = 1,
        tam_pagina: int = 10,
        term_busqueda: Optional[str] = None,
        campo_orden: str = 'AREA',
        dir_orden: str = 'ASC',
        filtro_estado: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get paginated areas using stored procedure SP_AREAS_LST_PAG.

        Args:
            db: Database session
            id_usuario: User ID for role validation
            id_empresa: Company ID to filter areas
            num_pagina: Page number (starting at 1)
            tam_pagina: Number of rows per page
            term_busqueda: Optional search term for AREA
            campo_orden: Field to sort by (default 'AREA')
            dir_orden: Sort direction ASC or DESC (default 'ASC')
            filtro_estado: Optional filter for ID_ESTADO_REGISTRO (None=all, 1=active, 0=inactive)

        Returns:
            Dictionary with:
                - data: List of area dictionaries
                - pagination: Dictionary with pagination metadata
                - message_result: Dict with ID_TIPO_MENSAJE and MENSAJE if error/authorization
        """
        try:
            repository = AreaRepository(db)

            result = repository.get_areas_paginated(
                id_usuario=id_usuario,
                id_empresa=id_empresa,
                num_pagina=num_pagina,
                tam_pagina=tam_pagina,
                term_busqueda=term_busqueda,
                campo_orden=campo_orden,
                dir_orden=dir_orden,
                filtro_estado=filtro_estado
            )

            # Check if there's a message from the SP
            if result.get('message_result'):
                message_result = result['message_result']
                tipo_mensaje = message_result.get('ID_TIPO_MENSAJE')
                
                # Only return empty if it's an error (type 1 or 3), not success (type 2)
                if tipo_mensaje == 1:
                    logger.warning(f"Authorization error for user {id_usuario}: {message_result.get('MENSAJE')}")
                    return {
                        'data': [],
                        'pagination': {},
                        'message_result': message_result
                    }
                elif tipo_mensaje == 3:
                    logger.error(f"Technical error for user {id_usuario}: {message_result.get('MENSAJE')}")
                    return {
                        'data': [],
                        'pagination': {},
                        'message_result': message_result
                    }
                # If tipo_mensaje == 2 (success), just log and continue normally
                else:
                    logger.info(f"SP success message: {message_result.get('MENSAJE')}")

            if result and result.get('data'):
                logger.info(
                    f"Retrieved {len(result['data'])} areas "
                    f"(page {result['pagination'].get('current_page')} of {result['pagination'].get('total_pages')}, "
                    f"total: {result['pagination'].get('total_records')}, "
                    f"ordered by: {campo_orden} {dir_orden})"
                )
            else:
                logger.warning("No paginated areas found")

            return result

        except Exception as e:
            logger.error(f"Error in get_areas_paginated service: {e}")
            raise