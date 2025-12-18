"""Repository for IA area configuration operations."""

import logging
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
from decimal import Decimal

logger = logging.getLogger(__name__)


class IaConfigRepository:
    """
    Repository for managing IA area configuration.
    Handles database operations for loading IA area-specific settings.
    """

    def __init__(self, db: Session):
        """
        Initialize the repository with a database session.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    def get_ia_area_config(self, id_ia_area: int) -> Optional[str]:
        """
        Load IA area configuration from database using stored procedure.

        Calls SP_IA_AREA_CONFIG_LOAD to retrieve configuration text for the specified IA area.

        Args:
            id_ia_area: ID of the IA area

        Returns:
            Configuration string (max 1000 characters) from stored procedure, or None if not found

        Raises:
            Exception: If database operation fails
        """
        try:
            query = text("""
                EXEC SP_IA_AREA_CONFIG_LOAD
                @ID_IA_AREA = :id_ia_area
            """)

            result = self.db.execute(query, {
                'id_ia_area': id_ia_area
            })

            config_data = result.fetchone()
            result.close()

            if not config_data:
                logger.info(f"No config found for id_ia_area={id_ia_area}")
                return None

            # Convert result to dictionary
            config_dict = dict(config_data._mapping) if hasattr(config_data, '_mapping') else dict(zip(result.keys(), config_data))

            # Get the first value from the result (config text)
            config_text = list(config_dict.values())[0] if config_dict else None

            if config_text and str(config_text).strip():
                # Limit to 1000 characters as per business rule
                return str(config_text)[:1000]

            return None

        except Exception as e:
            logger.error(f"Error loading IA area config for id_ia_area={id_ia_area}: {e}")
            raise

    def get_ia_area_config_full(self, id_area: int) -> Optional[Dict[str, Any]]:
        """
        Load full IA area configuration from database using stored procedure.

        Calls SP_GET_IA_AREA_CONFIG to retrieve all configuration parameters for the specified area.

        Args:
            id_area: ID of the area

        Returns:
            Dictionary containing ID_IA_AREA, ID_AREA, ID_EMBEDDINGS, ID_LLM, EMBEDDINGS_DIMENSIONS,
            LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TOP_P, RAG_TOP_K_RESULTS, RAG_SIMILARITY_THRESHOLD,
            RAG_ALPHA, ROLE_BEHAVIOR, or None if not found

        Raises:
            Exception: If database operation fails
        """
        try:
            query = text("""
                EXEC SP_GET_IA_AREA_CONFIG
                @ID_AREA = :id_area
            """)

            result = self.db.execute(query, {
                'id_area': id_area
            })

            config_data = result.fetchone()
            result.close()

            if not config_data:
                logger.info(f"No config found for id_area={id_area}")
                return None

            # Convert result to dictionary
            config_dict = dict(config_data._mapping) if hasattr(config_data, '_mapping') else dict(zip(result.keys(), config_data))

            if config_dict:
                logger.info(f"Retrieved IA area config for id_area={id_area}")
                return config_dict

            return None

        except Exception as e:
            logger.error(f"Error loading full IA area config for id_area={id_area}: {e}")
            raise

    def get_ia_area_config_rag(self, id_empresa: int, id_area: int) -> Optional[Dict[str, Any]]:
        """
        Load IA area RAG configuration from database using stored procedure.

        Calls SP_GET_IA_AREA_CONFIG_RAG to retrieve RAG-specific configuration parameters
        for the specified company and area.

        Args:
            id_empresa: ID of the company
            id_area: ID of the area

        Returns:
            Dictionary containing 'config' (first result set with all config params) and
            'general_area' (second result set), or None if not found

        Raises:
            Exception: If database operation fails
        """
        try:
            # Use raw connection to handle multiple result sets
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_GET_IA_AREA_CONFIG_RAG @ID_EMPRESA = ?, @ID_AREA = ?",
                    id_empresa,
                    id_area
                )

                result = {}

                # First result set - config parameters
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    row = cursor.fetchone()

                    if row:
                        config_dict = dict(zip(columns, row))

                        # Convert Decimal types to proper Python types
                        for key, value in config_dict.items():
                            if isinstance(value, Decimal):
                                # Convert to int if it's a whole number, otherwise float
                                if value % 1 == 0:
                                    config_dict[key] = int(value)
                                else:
                                    config_dict[key] = float(value)

                        result['config'] = config_dict
                    else:
                        logger.info(f"No RAG config found for id_empresa={id_empresa}, id_area={id_area}")
                        cursor.close()
                        return None

                # Second result set - GENERAL_AREA
                if cursor.nextset():
                    if cursor.description:
                        columns = [desc[0] for desc in cursor.description]
                        row = cursor.fetchone()

                        if row:
                            general_area_dict = dict(zip(columns, row))
                            general_area_value = general_area_dict.get('GENERAL_AREA')

                            # Convert Decimal to int if needed
                            if isinstance(general_area_value, Decimal):
                                general_area_value = int(general_area_value)

                            result['general_area'] = general_area_value

                cursor.close()

                if result:
                    logger.info(f"Retrieved IA area RAG config for id_empresa={id_empresa}, id_area={id_area}")
                    return result

                return None

            except Exception as cursor_error:
                logger.error(f"Cursor error in get_ia_area_config_rag: {cursor_error}")
                cursor.close()
                raise

        except Exception as e:
            logger.error(f"Error loading IA area RAG config for id_empresa={id_empresa}, id_area={id_area}: {e}")
            raise

    async def update_ia_area_config(
        self,
        id_usuario: int,
        id_empresa: int,
        id_area: int,
        id_embeddings: int,
        id_llm: int,
        embeddings_dimensions: int,
        llm_max_tokens: int,
        llm_temperature: Decimal,
        llm_top_p: Decimal,
        rag_top_k_results: int,
        rag_similarity_threshold: Decimal,
        rag_alpha: Decimal,
        role_behavior: str
    ) -> Dict[str, Any]:
        """
        Update IA area configuration using stored procedure SP_UPDATE_IA_AREA_BASE

        Args:
            id_usuario: User ID performing the update
            id_empresa: Company ID
            id_area: Area ID
            id_embeddings: Embeddings model ID
            id_llm: LLM model ID
            embeddings_dimensions: Embedding vector dimensions
            llm_max_tokens: Maximum tokens for LLM
            llm_temperature: Temperature parameter for LLM
            llm_top_p: Top-P parameter for LLM
            rag_top_k_results: Number of top-K results for RAG
            rag_similarity_threshold: Similarity threshold for RAG
            rag_alpha: Alpha parameter for RAG
            role_behavior: Role behavior prompt/instructions

        Returns:
            Dictionary containing ID_TIPO_MENSAJE and message
        """
        try:
            # Use raw connection to handle stored procedure execution
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()

            try:
                cursor.execute(
                    "EXEC SP_UPDATE_IA_AREA_BASE @ID_USUARIO = ?, @ID_EMPRESA = ?, @ID_AREA = ?, "
                    "@ID_EMBEDDINGS = ?, @ID_LLM = ?, @EMBEDDINGS_DIMENSIONS = ?, "
                    "@LLM_MAX_TOKENS = ?, @LLM_TEMPERATURE = ?, @LLM_TOP_P = ?, "
                    "@RAG_TOP_K_RESULTS = ?, @RAG_SIMILARITY_THRESHOLD = ?, @RAG_ALPHA = ?, "
                    "@ROLE_BEHAVIOR = ?",
                    id_usuario,
                    id_empresa,
                    id_area,
                    id_embeddings,
                    id_llm,
                    embeddings_dimensions,
                    llm_max_tokens,
                    llm_temperature,
                    llm_top_p,
                    rag_top_k_results,
                    rag_similarity_threshold,
                    rag_alpha,
                    role_behavior
                )

                result = {}

                # SP returns 2 result sets
                # First result set - skip it
                if cursor.description:
                    cursor.fetchall()

                # Move to second result set - contains ID_TIPO_MENSAJE and MENSAJE
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
                logger.error(f"Cursor error in update_ia_area_config: {cursor_error}")
                cursor.close()
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error updating IA area config with SP_UPDATE_IA_AREA_BASE: {e}")
            raise
