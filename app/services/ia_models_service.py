"""Service for managing IA models operations following hexagonal architecture."""

from typing import List, Dict, Any
import logging
from sqlalchemy.orm import Session
from app.infrastructure.repositories.ia_models_repository import IAModelsRepository

logger = logging.getLogger(__name__)


class IAModelsService:
    """
    Service for IA models operations.
    Handles business logic for retrieving and managing IA models.
    """

    def __init__(self):
        """Initialize stateless IAModelsService - no db parameter."""
        pass

    async def get_models(
        self,
        db: Session
    ) -> List[Dict[str, Any]]:
        """
        Get all available IA models using stored procedure.

        Args:
            db: Database session

        Returns:
            List of dictionaries containing model information:
            - ID_MODELO: Model ID
            - MODELO: Model name
            - ID_ESTADO_REGISTRO: Record status (1=active, 0=inactive)
            Empty list if query failed
        """
        try:
            # Create repository for this request
            repository = IAModelsRepository(db)

            # Use repository to get models with SP_MODELS_LST
            results = repository.get_models()

            if results:
                logger.info(f"Models retrieved successfully: Count={len(results)}")
            else:
                logger.info("No models found")

            return results

        except Exception as e:
            logger.error(f"Error in get_models service: {e}")
            raise

    async def create_model(
        self,
        db: Session,
        id_usuario: int,
        model: str,
        id_model: str,
        provider: str,
        type_id: int,
        extra_parameter: int
    ) -> Dict[str, Any]:
        """
        Create a new IA model using stored procedure.

        Args:
            db: Database session
            id_usuario: User ID creating the model
            model: Model name/display name
            id_model: Model identifier
            provider: Provider name
            type_id: Model type ID
            extra_parameter: Type-specific parameter

        Returns:
            Dictionary containing ID_TIPO_MENSAJE and message
        """
        try:
            # Create repository for this request
            repository = IAModelsRepository(db)

            # Use repository to create model with SP_CREATE_MODEL
            result = await repository.create_model(
                id_usuario=id_usuario,
                model=model,
                id_model=id_model,
                provider=provider,
                type_id=type_id,
                extra_parameter=extra_parameter
            )

            if result and result.get('ID_TIPO_MENSAJE'):
                logger.info(f"Model created successfully: User={id_usuario}, Model={model}")
            else:
                logger.warning(f"Model creation returned unexpected result: {result}")

            return result

        except Exception as e:
            logger.error(f"Error in create_model service: {e}")
            raise
