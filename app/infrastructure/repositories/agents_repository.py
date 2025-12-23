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
    def get_agentes(self, id_empresa: int) -> List[Dict[str, Any]]:
        """
        Get all agents for a company using stored procedure SP_AGENTES_LST

        Args:
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
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_AGENTES_LST @ID_EMPRESA = ?",
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
                                        if field in result_dict and result_dict[field] is not None:
                                            result_dict[field] = int(result_dict[field])
                                    # Ensure NUMERO_TELF field is always present (even if NULL)
                                    if 'NUMERO_TELF' not in result_dict:
                                        result_dict['NUMERO_TELF'] = None
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
                logger.error(f"Cursor error in get_agentes: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error getting agentes with SP_AGENTES_LST: {e}")
            self.db.rollback()
            return []

    @retry_on_db_error(max_retries=3, delay=1)
    def create_agente(
        self,
        id_usuario: int,
        numero_telf: str,
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
            id_tipo_agente: Agent type ID
            id_empresa: Company ID
            acceso_general: General access flag
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
                    "EXEC SP_CREATE_AGENTE @ID_USUARIO = ?, @NUMERO_TELF = ?, @ID_TIPO_AGENTE = ?, @ID_EMPRESA = ?, @ACCESO_GENERAL = ?, @AREAS_STRING = ?",
                    id_usuario,
                    numero_telf,
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