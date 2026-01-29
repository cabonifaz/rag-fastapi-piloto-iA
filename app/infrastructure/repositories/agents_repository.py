"""Repository for agents database operations."""

from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List, Dict, Any
import logging
from app.core.database import retry_on_db_error

logger = logging.getLogger(__name__)


class AgentsRepository:
    """
    Repository for AGENTES table operations.
    """

    def __init__(self, db: Session):
        self.db = db


    @retry_on_db_error(max_retries=3, delay=1)
    def create_agente(
        self,
        id_usuario: int,
        numero_telf: str,
        codigo_pais: str,
        id_tipo_agente: int,
        id_empresa: int,
        acceso_general: int,
        areas_string: str
    ) -> List[Dict[str, Any]]:
        """
        Create a new agent using stored procedure SP_CREATE_AGENTE

        Args:
            id_usuario: User ID
            numero_telf: Phone number (max 20 chars)
            codigo_pais: Country code composite (max 8 chars, e.g., '51-PE')
            id_tipo_agente: Agent type ID
            id_empresa: Company ID
            acceso_general: General access flag
            areas_string: Comma-separated area IDs (max 100 chars)

        Returns:
            List of dictionaries with: ID_TIPO_MENSAJE, MENSAJE
            Empty list if creation faile d
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_CREATE_AGENTE @ID_USUARIO = ?, @NUMERO_TELF = ?, @CODIGO_PAIS = ?, @ID_TIPO_AGENTE = ?, @ID_EMPRESA = ?, @ACCESO_GENERAL = ?, @AREAS_STRING = ?",
                    id_usuario,
                    numero_telf,
                    codigo_pais,
                    id_tipo_agente,
                    id_empresa,
                    acceso_general,
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
                logger.error(f"Cursor error in create_agente: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error creating agente with SP_CREATE_AGENTE: {e}")
            self.db.rollback()
            return []

    @retry_on_db_error(max_retries=3, delay=1)
    def verify_acceso_agente(
        self,
        numero_telf: str,
        secret_key: str
    ) -> Dict[str, Any]:
        """
        Verify agent access using stored procedure SP_VERIFY_ACCESO_AGENTE

        Args:
            numero_telf: Phone number (max 20 chars)
            secret_key: Secret key (max 64 chars)

        Returns:
            Dictionary containing:
            - mensaje: Dict with ID_TIPO_MENSAJE, MENSAJE
            - agente: Dict with ID_AGENTE, ID_EMPRESA
            - rol: Dict with ID_TIPO_ROL, ROL
            - company_areas: List of dicts with ID_EMPRESA, EMPRESA, ID_AREA, AREA
            Empty dict if verification failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_VERIFY_ACCESO_AGENTE @NUMERO_TELF = ?, @SECRET_KEY = ?",
                    numero_telf,
                    secret_key
                )

                result = {
                    'mensaje': None,
                    'agente': None,
                    'rol': None,
                    'company_areas': []
                }

                result_set_index = 0

                # Iterate through all result sets
                while True:
                    try:
                        # Check if we have columns (indicating data)
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            rows = cursor.fetchall()

                            if rows:
                                # Process based on result set index
                                if result_set_index == 0:
                                    # First result set: ID_TIPO_MENSAJE, MENSAJE (always present)
                                    row = rows[0]
                                    result['mensaje'] = dict(zip(columns, row))
                                    if 'ID_TIPO_MENSAJE' in result['mensaje']:
                                        result['mensaje']['ID_TIPO_MENSAJE'] = int(result['mensaje']['ID_TIPO_MENSAJE'])

                                    # If authentication failed (ID_TIPO_MENSAJE != 2), SP returns only this result set
                                    if result['mensaje']['ID_TIPO_MENSAJE'] != 2:
                                        cursor.close()
                                        self.db.commit()
                                        return {'mensaje': result['mensaje']}

                                elif result_set_index == 1:
                                    # Second result set: ID_AGENTE, ID_EMPRESA
                                    row = rows[0]
                                    result['agente'] = dict(zip(columns, row))
                                    if 'ID_AGENTE' in result['agente']:
                                        result['agente']['ID_AGENTE'] = int(result['agente']['ID_AGENTE'])
                                    if 'ID_EMPRESA' in result['agente']:
                                        result['agente']['ID_EMPRESA'] = int(result['agente']['ID_EMPRESA'])

                                elif result_set_index == 2:
                                    # Third result set: ID_TIPO_ROL, ROL
                                    row = rows[0]
                                    result['rol'] = dict(zip(columns, row))
                                    if 'ID_TIPO_ROL' in result['rol']:
                                        result['rol']['ID_TIPO_ROL'] = int(result['rol']['ID_TIPO_ROL'])

                                elif result_set_index == 3:
                                    # Fourth result set: ID_EMPRESA, EMPRESA, ID_AREA, AREA (multiple rows)
                                    for row in rows:
                                        area_dict = dict(zip(columns, row))
                                        if 'ID_EMPRESA' in area_dict:
                                            area_dict['ID_EMPRESA'] = int(area_dict['ID_EMPRESA'])
                                        if 'ID_AREA' in area_dict:
                                            area_dict['ID_AREA'] = int(area_dict['ID_AREA'])
                                        result['company_areas'].append(area_dict)

                                result_set_index += 1

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
                return result

            except Exception as cursor_error:
                logger.error(f"Cursor error in verify_acceso_agente: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error verifying acceso agente with SP_VERIFY_ACCESO_AGENTE: {e}")
            self.db.rollback()
            return {}

    
    @retry_on_db_error(max_retries=3, delay=1)
    def get_agentes_paginated(
        self,
        id_usuario: int,
        id_empresa: int,
        num_pagina: int = 1,
        tam_pagina: int = 10,
        term_busqueda: Optional[str] = None,
        campo_orden: str = 'NUMERO_TELF',
        dir_orden: str = 'ASC',
        filtro_estado: Optional[int] = None,
        filtro_operativo: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get paginated agents using stored procedure SP_AGENTES_LST_PAG

        Args:
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
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_AGENTES_LST_PAG @NUM_PAGINA = ?, @TAM_PAGINA = ?, @TERM_BUSQUEDA = ?, @CAMPO_ORDEN = ?, @DIR_ORDEN = ?, @ID_USUARIO = ?, @ID_EMPRESA = ?, @FILTRO_ESTADO = ?, @FILTRO_OPERATIVO = ?",
                    num_pagina,
                    tam_pagina,
                    term_busqueda,
                    campo_orden,
                    dir_orden,
                    id_usuario,
                    id_empresa,
                    filtro_estado,
                    filtro_operativo
                )

                agentes = []
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

                            # Check if this result set contains AGENT data
                            has_agent_data = 'ID_AGENTE' in columns and 'NUMERO_TELF' in columns

                            if has_message_columns and rows:
                                # This is an error/authorization message result
                                for row in rows:
                                    result_dict = dict(zip(columns, row))
                                    # Convert Decimal to int for ID_TIPO_MENSAJE
                                    if 'ID_TIPO_MENSAJE' in result_dict:
                                        result_dict['ID_TIPO_MENSAJE'] = int(result_dict['ID_TIPO_MENSAJE'])
                                    message_result = result_dict
                                    
                            elif has_agent_data and rows:  
                                # This is the agent data with pagination
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
                                    agent_dict = dict(zip(columns, row))
                                    
                                    # Convert Decimal to int for numeric IDs
                                    numeric_fields = [
                                        'ID_AGENTE_EMPR_AREA',
                                        'ID_AGENTE',
                                        'ID_TIPO_AGENTE',
                                        'ACCESO_GENERAL',
                                        'ESTADO_OPERATIVO',
                                        'ID_ESTADO_REGISTRO',
                                        'ID_EMPRESA',
                                        'ID_AREA'
                                    ]
                                    for field in numeric_fields:
                                        if field in agent_dict and agent_dict[field] is not None:
                                            agent_dict[field] = int(agent_dict[field])
                                    
                                    # Remove pagination metadata columns
                                    agent_dict.pop('TotalRecords', None)
                                    agent_dict.pop('CurrentPage', None)
                                    agent_dict.pop('PageSize', None)
                                    agent_dict.pop('TotalPages', None)
                                    agentes.append(agent_dict)  
                    except Exception as fetch_error:
                        logger.error(f"Fetch error in get_agentes_paginated: {fetch_error}")

                    # Move to next result set
                    try:
                        if not cursor.nextset():
                            break
                    except Exception as nextset_error:
                        # Transaction error is expected when SP manages its own transactions
                        if "Transaction count after EXECUTE" in str(nextset_error):
                            logger.debug(f"SP manages its own transactions (expected): {nextset_error}")
                        else:
                            logger.error(f"Nextset error in get_agentes_paginated: {nextset_error}")
                        break

                cursor.close()
                self.db.commit()
                
                return {
                    'data': agentes,
                    'pagination': pagination_info,
                    'message_result': message_result
                }

            except Exception as cursor_error:
                logger.error(f"Cursor error in get_agentes_paginated: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error fetching paginated agentes with SP: {e}")
            self.db.rollback()
            return {'data': [], 'pagination': {}, 'message_result': None}