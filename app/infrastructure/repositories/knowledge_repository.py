"""Repository for knowledge/document database operations using SQL Server."""

from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class KnowledgeRepository:
    """
    Repository for CARGA_CONOCIMIENTO table operations.
    Uses stored procedures for knowledge/document management.
    """

    def __init__(self, db: Session):
        self.db = db

    def get_knowledge_by_company(
        self,
        id_usuario: int,
        id_empresa: int,
        id_area: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all knowledge/documents for a company using stored procedure SP_CARGA_CONOCIMIENTO_EMPRESA_LST

        Args:
            id_usuario: User ID requesting the documents
            id_empresa: Company ID
            id_area: Area ID (optional, if None retrieves all areas)

        Returns:
            List of dictionaries containing knowledge/document information
            Empty list if query failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_CARGA_CONOCIMIENTO_EMPRESA_LST @ID_USUARIO = ?, @ID_EMPRESA = ?, @ID_AREA = ?",
                    id_usuario,
                    id_empresa,
                    id_area
                )

                results = []

                # Iterate through all result sets
                result_set_num = 0
                while True:
                    result_set_num += 1
                    try:
                        # Check if we have columns (indicating data)
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            rows = cursor.fetchall()

                            # Look for result set with knowledge data (has 'ID_CARGA' or 'documento' column)
                            has_knowledge_data = any(col in ['ID_CARGA', 'documento', 'NOMBRE_DOCUMENTO'] for col in columns)
                            if has_knowledge_data and rows:
                                # Convert rows to dictionaries
                                for row in rows:
                                    result_dict = dict(zip(columns, row))

                                    # The SP already returns lowercase column names, so just use the result_dict as-is
                                    # Convert numeric IDs to int
                                    if 'id' in result_dict and result_dict['id'] is not None:
                                        result_dict['id'] = str(result_dict['id'])

                                    for id_field in ['id_usuario', 'id_empresa', 'id_area', 'id_estado_proceso', 'id_modelo_embedding']:
                                        if id_field in result_dict and result_dict[id_field] is not None:
                                            result_dict[id_field] = int(result_dict[id_field])

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
                logger.error(f"Cursor error in get_knowledge_by_company: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error getting knowledge with SP_CARGA_CONOCIMIENTO_EMPRESA_LST: {e}")
            self.db.rollback()
            return []

    def update_knowledge_process_state(
        self,
        id_carga: int,
        id_estado_proceso: int,
        usumod: str = "System"
    ) -> List[Dict[str, Any]]:
        """
        Update the process state for a knowledge/document record using stored procedure SP_CARGA_CONOC_ESTADO_PROCESO_UPD

        Args:
            id_carga: Knowledge/document ID to update
            id_estado_proceso: New process state ID
            usumod: User who modified the record (max 200 chars, default: "System")

        Returns:
            List of dictionaries with: ID_TIPO_MENSAJE, MENSAJE
            Empty list if update failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_CARGA_CONOC_ESTADO_PROCESO_UPD @ID_CARGA = ?, @ID_ESTADO_PROCESO = ?, @USUMOD = ?",
                    id_carga,
                    id_estado_proceso,
                    usumod
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
                logger.error(f"Cursor error in update_knowledge_process_state: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error updating knowledge process state with SP_CARGA_CONOC_ESTADO_PROCESO_UPD: {e}")
            self.db.rollback()
            return []

    def batch_create_knowledge(
        self,
        id_usuario: int,
        id_empresa: int,
        id_area: int,
        records: List[Dict[str, str]],
        id_modelo_embedding: str = "4"
    ) -> List[Dict[str, Any]]:
        """
        Create multiple knowledge/document records using stored procedure SP_CARGA_CONOCIMIENTO_BATCH_INS
        with table-valued parameter CARGA_CONOCIMIENTO_BATCH.

        Args:
            id_usuario: User ID who uploaded the documents
            id_empresa: Company ID
            id_area: Area ID
            records: List of record dictionaries, each containing:
                - ruta_documento: Document path/route (max 500 chars)
                - nombre_documento: Document name (max 500 chars)
            id_modelo_embedding: Embedding model ID (default: "4")

        Returns:
            List of dictionaries with: ID_TIPO_MENSAJE, MENSAJE
            Empty list if creation failed
        """
        try:
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                # Prepare table-valued parameter data matching CARGA_CONOCIMIENTO_BATCH type
                # Each record is a tuple: (RUTA_DOCUMENTO VARCHAR(500), NOMBRE_DOCUMENTO VARCHAR(500))
                tvp_data = [
                    (record['ruta_documento'], record['nombre_documento'])
                    for record in records
                ]

                # Execute SP with table-valued parameter
                cursor.execute(
                    "EXEC SP_CARGA_CONOCIMIENTO_BATCH_INS @ID_USUARIO = ?, @ID_EMPRESA = ?, @ID_AREA = ?, @ID_MODELO_EMBEDDING = ?, @CARGAS = ?",
                    id_usuario,
                    id_empresa,
                    id_area,
                    id_modelo_embedding,
                    tvp_data
                )

                message_result = None
                created_ids = []
                result_set_num = 0

                # Iterate through all result sets
                while True:
                    result_set_num += 1
                    try:
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            rows = cursor.fetchall()

                            # Check if this result set contains the message columns
                            has_message_columns = 'ID_TIPO_MENSAJE' in columns and 'MENSAJE' in columns

                            # Check if this result set contains the ID_CARGA columns (knowledge IDs)
                            has_id_carga = 'ID_CARGA' in columns

                            if has_message_columns and rows:
                                # This is the result set we want to capture (only take first row)
                                row = rows[0]
                                result_dict = dict(zip(columns, row))
                                # Convert Decimal to int for ID_TIPO_MENSAJE
                                if 'ID_TIPO_MENSAJE' in result_dict:
                                    result_dict['ID_TIPO_MENSAJE'] = int(result_dict['ID_TIPO_MENSAJE'])
                                message_result = result_dict

                            elif has_id_carga and rows:
                                # Capture the ID_CARGA values separately
                                for row in rows:
                                    result_dict = dict(zip(columns, row))
                                    if 'ID_CARGA' in result_dict and result_dict['ID_CARGA'] is not None:
                                        id_carga = int(result_dict['ID_CARGA'])
                                        created_ids.append(id_carga)

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
                return {
                    'message_result': message_result,
                    'created_ids': created_ids
                }

            except Exception as cursor_error:
                logger.error(f"Cursor error in batch_create_knowledge: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error batch creating knowledge with SP_CARGA_CONOCIMIENTO_BATCH_INS: {e}")
            self.db.rollback()
            return []

    def batch_update_knowledge_state(
        self,
        id_usuario: int,
        id_cargas: List[int],
        id_estado_proceso: int,
        usumod: str = "System"
    ) -> Dict[str, Any]:
        """
        Update process state for multiple knowledge/document records using stored procedure
        SP_CARGA_CONOC_ESTADO_PROCESO_BATCH_UPD with table-valued parameter.

        Args:
            id_usuario: User ID who is updating (for audit purposes)
            id_cargas: List of knowledge/document IDs to update
            id_estado_proceso: New process state ID
            usumod: User who modified the records (max 200 chars, default: "System")

        Returns:
            Dictionary with ID_TIPO_MENSAJE and MENSAJE from the stored procedure
        """
        try:
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                # Prepare table-valued parameter data matching UPDATE_CONOCIMIENTO_BATCH type
                # Each record is a tuple: (ID_CARGA INT)
                tvp_data = [
                    (id_carga,)
                    for id_carga in id_cargas
                ]

                # Execute SP with table-valued parameter
                cursor.execute(
                    "EXEC SP_CARGA_CONOC_ESTADO_PROCESO_BATCH_UPD @ID_USUARIO = ?, @USUMOD = ?, @ID_ESTADO_PROCESO = ?, @CARGAS = ?",
                    id_usuario,
                    usumod,
                    id_estado_proceso,
                    tvp_data
                )

                message_result = None
                affected_count = 0
                result_set_num = 0

                # Iterate through all result sets
                while True:
                    result_set_num += 1
                    try:
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            rows = cursor.fetchall()

                            # Check if this result set contains the message columns
                            has_message_columns = 'ID_TIPO_MENSAJE' in columns and 'MENSAJE' in columns

                            if has_message_columns and rows:
                                # This is the result set we want to capture (only take first row)
                                row = rows[0]
                                result_dict = dict(zip(columns, row))
                                # Convert Decimal to int for ID_TIPO_MENSAJE
                                if 'ID_TIPO_MENSAJE' in result_dict:
                                    result_dict['ID_TIPO_MENSAJE'] = int(result_dict['ID_TIPO_MENSAJE'])
                                message_result = result_dict

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
                logger.info(f"Batch updated {len(id_cargas)} knowledge records to state {id_estado_proceso}")
                return message_result

            except Exception as cursor_error:
                logger.error(f"Cursor error in batch_update_knowledge_state: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error batch updating knowledge state with SP_CARGA_CONOC_ESTADO_PROCESO_BATCH_UPD: {e}")
            self.db.rollback()
            return None
