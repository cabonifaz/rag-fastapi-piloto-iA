"""Repository for IA models database operations."""

from sqlalchemy.orm import Session
from typing import List, Dict, Any
import logging
from app.core.database import retry_on_db_error

logger = logging.getLogger(__name__)


class IAModelsRepository:
    """
    Repository for IA models operations.
    Handles interactions with stored procedures for IA models.
    """

    def __init__(self, db: Session):
        self.db = db

    @retry_on_db_error(max_retries=3, delay=1)
    def get_models(self) -> List[Dict[str, Any]]:
        """
        Get all available IA models using stored procedure SP_MODELS_LST

        Returns:
            List of dictionaries containing model information
            Empty list if query failed
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute("EXEC SP_MODELS_LST")

                models = []

                # Get the models data
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()

                    # Convert rows to list of dictionaries
                    for row in rows:
                        model_dict = dict(zip(columns, row))
                        # Convert Decimal to int for numeric IDs
                        numeric_fields = ['ID_MODELO', 'ID_ESTADO_REGISTRO']
                        for field in numeric_fields:
                            if field in model_dict and model_dict[field] is not None:
                                model_dict[field] = int(model_dict[field])
                        models.append(model_dict)

                cursor.close()
                return models

            except Exception as cursor_error:
                logger.error(f"Cursor error in get_models: {cursor_error}")
                cursor.close()
                raise

        except Exception as e:
            logger.error(f"Error fetching models with SP_MODELS_LST: {e}")
            return []

    @retry_on_db_error(max_retries=3, delay=1)
    async def create_model(
        self,
        id_usuario: int,
        model: str,
        id_model: str,
        provider: str,
        type_id: int,
        extra_parameter: int
    ) -> Dict[str, Any]:
        """
        Create a new IA model using stored procedure SP_CREATE_MODEL

        Args:
            id_usuario: User ID creating the model
            model: Model name/display name
            id_model: Model identifier (e.g., anthropic.claude-3-haiku-20240307-v1:0)
            provider: Provider name (e.g., Anthropic, OpenAI)
            type_id: Model type ID (1=Embeddings, 2=Text/Vision, etc.)
            extra_parameter: Type-specific parameter (vector_size for Embeddings, max_tokens for Text/Vision)

        Returns:
            Dictionary containing ID_TIPO_MENSAJE and message
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_CREATE_MODEL @ID_USUARIO = ?, @MODEL = ?, @ID_MODEL = ?, "
                    "@PROVIDER = ?, @TYPE = ?, @EXTRA_PARAMETER = ?",
                    id_usuario,
                    model,
                    id_model,
                    provider,
                    type_id,
                    extra_parameter
                )

                result = {}

                # SP returns 2 result sets, we need the second one with ID_TIPO_MENSAJE and MENSAJE
                # First result set - skip it
                if cursor.description:
                    cursor.fetchall()

                # Move to second result set
                if cursor.nextset():
                    if cursor.description:
                        columns = [desc[0] for desc in cursor.description]
                        row = cursor.fetchone()

                        if row:
                            full_result = dict(zip(columns, row))
                            # Return only ID_TIPO_MENSAJE and message
                            if 'ID_TIPO_MENSAJE' in full_result:
                                result['ID_TIPO_MENSAJE'] = int(full_result['ID_TIPO_MENSAJE'])
                            if 'MENSAJE' in full_result:
                                result['MENSAJE'] = full_result['MENSAJE']
                cursor.close()
                self.db.commit()
                return result

            except Exception as cursor_error:
                logger.error(f"Cursor error in create_model: {cursor_error}")
                cursor.close()
                raise

        except Exception as e:
            logger.error(f"Error creating model with SP_CREATE_MODEL: {e}")
            raise
