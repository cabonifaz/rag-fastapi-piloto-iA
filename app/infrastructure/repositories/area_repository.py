"""Repository for area database operations."""

from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AreaRepository:
    """
    Repository for AREA table operations.
    """

    def __init__(self, db: Session):
        self.db = db

    def create_area(
        self,
        id_usuario: int,
        id_empresa: int,
        area: str
    ) -> List[Dict[str, Any]]:
        """
        Create a new area base using stored procedure SP_CREATE_AREA_BASE

        Args:
            id_usuario: User ID
            id_empresa: Company ID
            area: Area name (max 100 chars)

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
                    "EXEC SP_CREATE_AREA_BASE @ID_USUARIO = ?, @ID_EMPRESA = ?, @AREA = ?",
                    id_usuario,
                    id_empresa,
                    area
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
                logger.error(f"Cursor error in create_area: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error creating area base with SP: {e}")
            self.db.rollback()
            return []

    def update_area_status(
        self,
        id_usuario: int,
        id_empresa: int,
        id_area: int,
        status: int
    ) -> List[Dict[str, Any]]:
        """
        Update area status using stored procedure SP_UPDATE_AREA_STATUS

        Args:
            id_usuario: User ID performing the update
            id_empresa: Company ID
            id_area: Area ID
            status: New status value (0 = inactive, 1 = active)

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
                    "EXEC SP_UPDATE_AREA_STATUS @ID_USUARIO = ?, @ID_EMPRESA = ?, @ID_AREA = ?, @STATUS = ?",
                    id_usuario,
                    id_empresa,
                    id_area,
                    status
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
                logger.error(f"Cursor error in update_area_status: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error updating area status with SP: {e}")
            self.db.rollback()
            return []

    def update_area_nombre(
        self,
        id_usuario: int,
        id_empresa: int,
        id_area: int,
        area: str
    ) -> List[Dict[str, Any]]:
        """
        Update area name using stored procedure SP_UPDATE_AREA_NOMBRE

        Args:
            id_usuario: User ID performing the update
            id_empresa: Company ID
            id_area: Area ID
            area: New area name (max 200 chars)

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
                    "EXEC SP_UPDATE_AREA_NOMBRE @ID_USUARIO = ?, @ID_EMPRESA = ?, @ID_AREA = ?, @AREA = ?",
                    id_usuario,
                    id_empresa,
                    id_area,
                    area
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
                logger.error(f"Cursor error in update_area_nombre: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error updating area name with SP: {e}")
            self.db.rollback()
            return []

    def get_areas(self, id_empresa: int) -> List[Dict[str, Any]]:
        """
        Get all areas for a company using stored procedure SP_AREAS_LST

        Args:
            id_empresa: Company ID

        Returns:
            List of dictionaries containing area information
            Empty list if query failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_AREAS_LST @ID_EMPRESA = ?",
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
                                    if 'ID_AREA' in result_dict:
                                        result_dict['ID_AREA'] = int(result_dict['ID_AREA'])
                                    if 'ID_EMPRESA' in result_dict:
                                        result_dict['ID_EMPRESA'] = int(result_dict['ID_EMPRESA'])
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
                logger.error(f"Cursor error in get_areas: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error getting areas with SP_AREAS_LST: {e}")
            self.db.rollback()
            return []
