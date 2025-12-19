"""Repository for knowledge/document database operations using SQL Server."""

from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
import logging
from app.core.database import retry_on_db_error

logger = logging.getLogger(__name__)


class KnowledgeRepository:
    """
    Repository for CARGA_CONOCIMIENTO table operations.
    Uses stored procedures for knowledge/document management.
    """

    def __init__(self, db: Session):
        self.db = db

    @retry_on_db_error(max_retries=3, delay=1)
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

    @retry_on_db_error(max_retries=3, delay=1)
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

    @retry_on_db_error(max_retries=3, delay=1)
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

    @retry_on_db_error(max_retries=3, delay=1)
    def get_knowledge_by_ids(
        self,
        id_usuario: int,
        id_cargas: List[int]
    ) -> Dict[str, Any]:
        """
        Get knowledge/document records by multiple IDs using stored procedure
        SP_CARGA_CONOCIMIENTO_SEL_BY_IDS with table-valued parameter.

        Validates user permissions (admin/superadmin only) and returns records.

        Args:
            id_usuario: User ID requesting the documents (for permission validation)
            id_cargas: List of knowledge/document IDs to retrieve

        Returns:
            Dictionary with:
            - message_result: Dict with ID_TIPO_MENSAJE and MENSAJE from SP
            - records: List of records with ID_CARGA, ID_EMPRESA, ID_AREA
        """
        try:
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                # Prepare table-valued parameter data matching DELETE_CONOCIMIENTO_BATCH type
                # Each record is a tuple: (ID_CARGA INT)
                tvp_data = [
                    (id_carga,)
                    for id_carga in id_cargas
                ]

                # Execute SP with table-valued parameter
                cursor.execute(
                    "EXEC SP_CARGA_CONOCIMIENTO_SEL_BY_IDS @ID_USUARIO = ?, @CARGAS = ?",
                    id_usuario,
                    tvp_data
                )

                message_result = None
                records = []
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

                            # Check if this result set contains knowledge records
                            has_knowledge_records = 'ID_CARGA' in columns and 'ID_EMPRESA' in columns

                            if has_message_columns and rows:
                                # This is the result set with the message
                                row = rows[0]
                                result_dict = dict(zip(columns, row))
                                # Convert Decimal to int for ID_TIPO_MENSAJE
                                if 'ID_TIPO_MENSAJE' in result_dict:
                                    result_dict['ID_TIPO_MENSAJE'] = int(result_dict['ID_TIPO_MENSAJE'])
                                message_result = result_dict

                            elif has_knowledge_records and rows:
                                # Capture the knowledge records
                                for row in rows:
                                    result_dict = dict(zip(columns, row))
                                    # Convert to int
                                    if 'ID_CARGA' in result_dict and result_dict['ID_CARGA'] is not None:
                                        result_dict['ID_CARGA'] = int(result_dict['ID_CARGA'])
                                    if 'ID_EMPRESA' in result_dict and result_dict['ID_EMPRESA'] is not None:
                                        result_dict['ID_EMPRESA'] = int(result_dict['ID_EMPRESA'])
                                    if 'ID_AREA' in result_dict and result_dict['ID_AREA'] is not None:
                                        result_dict['ID_AREA'] = int(result_dict['ID_AREA'])
                                    if 'ID_ESTADO_PROCESO' in result_dict and result_dict['ID_ESTADO_PROCESO'] is not None:
                                        result_dict['ID_ESTADO_PROCESO'] = int(result_dict['ID_ESTADO_PROCESO'])
                                    if 'EN_EJECUCION' in result_dict and result_dict['EN_EJECUCION'] is not None:
                                        result_dict['EN_EJECUCION'] = int(result_dict['EN_EJECUCION'])
                                    # RUTA_* fields remain as strings (can be None/NULL)
                                    records.append(result_dict)

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
                logger.info(f"Retrieved {len(records)} knowledge records by IDs")
                return {
                    'message_result': message_result,
                    'records': records
                }

            except Exception as cursor_error:
                logger.error(f"Cursor error in get_knowledge_by_ids: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error getting knowledge by IDs with SP_CARGA_CONOCIMIENTO_SEL_BY_IDS: {e}")
            self.db.rollback()
            return None

    @retry_on_db_error(max_retries=3, delay=1)
    def batch_delete_knowledge(
        self,
        id_usuario: int,
        id_cargas: List[int],
        usumod: str = "System"
    ) -> Dict[str, Any]:
        """
        Logically delete multiple knowledge/document records using stored procedure
        SP_CARGA_CONOCIMIENTO_BATCH_DEL with table-valued parameter.

        Args:
            id_usuario: User ID who is deleting (for audit purposes)
            id_cargas: List of knowledge/document IDs to delete
            usumod: User who deleted the records (max 200 chars, default: "System")

        Returns:
            Dictionary with:
            - message_result: Dict with ID_TIPO_MENSAJE and MENSAJE from SP
            - deleted_records: List of deleted records with ID_CARGA, NOMBRE_DOCUMENTO, ID_EMPRESA, ID_AREA
        """
        try:
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                # Each record is a tuple: (ID_CARGA INT)
                tvp_data = [
                    (id_carga,)
                    for id_carga in id_cargas
                ]

                # Execute SP with table-valued parameter
                cursor.execute(
                    "EXEC SP_CARGA_CONOCIMIENTO_BATCH_DEL @ID_USUARIO = ?, @USUMOD = ?, @CARGAS = ?",
                    id_usuario,
                    usumod,
                    tvp_data
                )

                message_result = None
                deleted_records = []
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

                            # Check if this result set contains deleted records info
                            has_deleted_records = 'ID_CARGA' in columns and 'NOMBRE_DOCUMENTO' in columns

                            if has_message_columns and rows:
                                # This is the result set with the message
                                row = rows[0]
                                result_dict = dict(zip(columns, row))
                                # Convert Decimal to int for ID_TIPO_MENSAJE
                                if 'ID_TIPO_MENSAJE' in result_dict:
                                    result_dict['ID_TIPO_MENSAJE'] = int(result_dict['ID_TIPO_MENSAJE'])
                                message_result = result_dict

                            elif has_deleted_records and rows:
                                # Capture the deleted records info
                                for row in rows:
                                    result_dict = dict(zip(columns, row))
                                    # Convert to int
                                    if 'ID_CARGA' in result_dict and result_dict['ID_CARGA'] is not None:
                                        result_dict['ID_CARGA'] = int(result_dict['ID_CARGA'])
                                    if 'ID_EMPRESA' in result_dict and result_dict['ID_EMPRESA'] is not None:
                                        result_dict['ID_EMPRESA'] = int(result_dict['ID_EMPRESA'])
                                    if 'ID_AREA' in result_dict and result_dict['ID_AREA'] is not None:
                                        result_dict['ID_AREA'] = int(result_dict['ID_AREA'])
                                    deleted_records.append(result_dict)

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
                logger.info(f"Batch deleted {len(deleted_records)} knowledge records")
                return {
                    'message_result': message_result,
                    'deleted_records': deleted_records
                }

            except Exception as cursor_error:
                logger.error(f"Cursor error in batch_delete_knowledge: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error batch deleting knowledge with SP_CARGA_CONOCIMIENTO_BATCH_DEL: {e}")
            self.db.rollback()
            return None

    @retry_on_db_error(max_retries=3, delay=1)
    def batch_restore_knowledge(
        self,
        id_usuario: int,
        id_cargas: List[int],
        usumod: str = "System"
    ) -> Dict[str, Any]:
        """
        Restore multiple logically deleted knowledge/document records using stored procedure
        SP_CARGA_CONOCIMIENTO_BATCH_RESTORE with table-valued parameter.

        Used for rollback/compensation when Weaviate deletion fails after SQL deletion.

        Args:
            id_usuario: User ID who is restoring (for audit purposes)
            id_cargas: List of knowledge/document IDs to restore
            usumod: User who restored the records (max 200 chars, default: "System")

        Returns:
            Dictionary with:
            - message_result: Dict with ID_TIPO_MENSAJE and MENSAJE from SP
            - restored_records: List of restored records with ID_CARGA, NOMBRE_DOCUMENTO, ID_EMPRESA, ID_AREA
        """
        try:
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                # Each record is a tuple: (ID_CARGA INT)
                tvp_data = [
                    (id_carga,)
                    for id_carga in id_cargas
                ]

                # Execute SP with table-valued parameter
                cursor.execute(
                    "EXEC SP_CARGA_CONOCIMIENTO_BATCH_RESTORE @ID_USUARIO = ?, @USUMOD = ?, @CARGAS = ?",
                    id_usuario,
                    usumod,
                    tvp_data
                )

                message_result = None
                restored_records = []
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

                            # Check if this result set contains restored records info
                            has_restored_records = 'ID_CARGA' in columns and 'NOMBRE_DOCUMENTO' in columns

                            if has_message_columns and rows:
                                # This is the result set with the message
                                row = rows[0]
                                result_dict = dict(zip(columns, row))
                                # Convert Decimal to int for ID_TIPO_MENSAJE
                                if 'ID_TIPO_MENSAJE' in result_dict:
                                    result_dict['ID_TIPO_MENSAJE'] = int(result_dict['ID_TIPO_MENSAJE'])
                                message_result = result_dict

                            elif has_restored_records and rows:
                                # Capture the restored records info
                                for row in rows:
                                    result_dict = dict(zip(columns, row))
                                    # Convert to int
                                    if 'ID_CARGA' in result_dict and result_dict['ID_CARGA'] is not None:
                                        result_dict['ID_CARGA'] = int(result_dict['ID_CARGA'])
                                    if 'ID_EMPRESA' in result_dict and result_dict['ID_EMPRESA'] is not None:
                                        result_dict['ID_EMPRESA'] = int(result_dict['ID_EMPRESA'])
                                    if 'ID_AREA' in result_dict and result_dict['ID_AREA'] is not None:
                                        result_dict['ID_AREA'] = int(result_dict['ID_AREA'])
                                    restored_records.append(result_dict)

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
                logger.info(f"Batch restored {len(restored_records)} knowledge records")
                return {
                    'message_result': message_result,
                    'restored_records': restored_records
                }

            except Exception as cursor_error:
                logger.error(f"Cursor error in batch_restore_knowledge: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error batch restoring knowledge with SP_CARGA_CONOCIMIENTO_BATCH_RESTORE: {e}")
            self.db.rollback()
            return None
    
    @retry_on_db_error(max_retries=3, delay=1)
    def batch_update_en_ejecucion(
        self,
        id_usuario: int,
        id_cargas: List[int],
        usumod: str = "System"
    ) -> Dict[str, Any]:
        """
        Update EN_EJECUCION field to 0 for knowledge records that have EN_EJECUCION = 1.

        This is called after validating that records with EN_EJECUCION = 1 exist,
        to set them back to 0 during deletion process.

        Args:
            id_usuario: User ID who is updating (for audit and permission validation)
            id_cargas: List of knowledge/document IDs to update
            usumod: User who updated the records (max 200 chars, default: "System")

        Returns:
            Dictionary with ID_TIPO_MENSAJE and MENSAJE from the stored procedure
        """
        try:
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                # Each record is a tuple: (ID_CARGA INT)
                tvp_data = [
                    (id_carga,)
                    for id_carga in id_cargas
                ]

                # Execute SP with table-valued parameter
                cursor.execute(
                    "EXEC SP_CARGA_CONOCIMIENTO_DETENER_EJECUCION @ID_USUARIO = ?, @USUMOD = ?, @CARGAS = ?",
                    id_usuario,
                    usumod,
                    tvp_data
                )

                message_result = None
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
                                # This is the result set with the message
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
                logger.info(f"Batch updated EN_EJECUCION for {len(id_cargas)} knowledge records")
                return message_result

            except Exception as cursor_error:
                logger.error(f"Cursor error in batch_update_en_ejecucion: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error batch updating EN_EJECUCION with SP_CARGA_CONOCIMIENTO_DETENER_EJECUCION: {e}")
            self.db.rollback()
            return None
