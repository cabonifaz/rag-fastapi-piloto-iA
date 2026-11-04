"""Repository for area-specific MCP configuration."""

import re
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session


class AreaMCPRepository:
    """Reads active MCP configuration for a company and area."""

    def __init__(self, db: Session):
        self.db = db

    def is_available(self, company_id: int, area_id: int) -> bool:
        result = self.db.execute(
            text(
                """
                SELECT 1
                FROM dbo.AREA_MCP
                WHERE ID_EMPRESA = :company_id
                  AND ID_AREA = :area_id
                  AND ID_ESTADO_REGISTRO = 1
                """
            ),
            {"company_id": company_id, "area_id": area_id},
        ).first()
        return result is not None

    def get_chat_config(
        self, company_id: int, area_id: int
    ) -> Optional[Dict[str, Any]]:
        raw_connection = self.db.connection().connection
        cursor = raw_connection.cursor()
        try:
            cursor.execute(
                "EXEC dbo.SP_GET_AREA_MCP_CONFIG "
                "@ID_EMPRESA = ?, @ID_AREA = ?",
                company_id,
                area_id,
            )
            if not cursor.description:
                return None
            columns = [column[0] for column in cursor.description]
            row = cursor.fetchone()
            return dict(zip(columns, row)) if row else None
        except Exception as exc:
            raise ValueError(self._get_database_error_message(exc)) from exc
        finally:
            cursor.close()

    @staticmethod
    def _get_database_error_message(exc: Exception) -> str:
        message = str(exc)
        match = re.search(
            r"\[SQL Server\](.*?)(?:\s+\(\d+\)\s+\(SQLExecDirectW\)|$)",
            message,
        )
        return (
            match.group(1).strip() if match else "Error al validar la configuración MCP"
        )
