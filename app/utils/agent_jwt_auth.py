from typing import Dict, Any, Optional
import jwt
from datetime import datetime, timezone, timedelta
import logging
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


class AgentJWTAuth:
    """JWT Authentication utility for agents"""

    @staticmethod
    def create_agent_jwt_token(agent_data: dict) -> str:
        """
        Create JWT token with agent data for frontend storage

        Args:
            agent_data: Dictionary containing agent verification data with:
                - mensaje: Dict with ID_TIPO_MENSAJE, MENSAJE
                - agente: Dict with ID_AGENTE, ID_EMPRESA
                - rol: Dict with ID_TIPO_ROL, ROL
                - company_areas: List of dicts with ID_EMPRESA, EMPRESA, ID_AREA, AREA

        Returns:
            JWT token string

        Raises:
            Exception if token creation fails
        """
        try:
            # Helper function to convert Decimal objects to int/float
            def convert_decimal(obj):
                from decimal import Decimal
                if isinstance(obj, Decimal):
                    return float(obj) if obj % 1 else int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_decimal(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_decimal(item) for item in obj]
                return obj

            # Convert agent_data to handle Decimal objects
            clean_agent_data = convert_decimal(agent_data)

            # Extract data from SP result sets
            agente = clean_agent_data.get('agente', {})
            rol = clean_agent_data.get('rol', {})
            company_areas = clean_agent_data.get('company_areas', [])

            # Extract fields
            agent_id = agente.get('ID_AGENTE')
            role_id = rol.get('ID_TIPO_ROL', 4)  # Default to 4 (Agente-IA)
            role_name = rol.get('ROL', 'Agente-IA')

            # Create JWT payload with all fields needed by frontend
            payload = {
                'ID_AGENTE': agent_id,
                'ID_TIPO_ROL': role_id,
                'ROL': role_name,
                'company_areas': company_areas,
                'exp': datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiration_minutes),
                'iat': datetime.now(timezone.utc),  # Issued at
                'iss': 'qamaq-rag-api-agent'  # Issuer (different from user tokens)
            }

            # Create JWT token
            token = jwt.encode(payload, settings.jwt_secret_key, algorithm='HS256')

            return token

        except Exception as e:
            logger.error(f"Error creating agent JWT token: {e}")
            raise

    @staticmethod
    def verify_agent_jwt_token(token: str) -> Dict[str, Any]:
        """
        Verify agent JWT token and extract payload

        Args:
            token: JWT token string

        Returns:
            Dictionary with agent information:
            - ID_AGENTE: Agent ID
            - ID_TIPO_ROL: Role type ID
            - ROL: Role name
            - company_areas: List of accessible areas

        Raises:
            HTTPException if token is invalid or expired
        """
        from fastapi import HTTPException

        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret_key,
                algorithms=['HS256']
            )

            # Check expiration
            exp_timestamp = payload.get('exp')
            if exp_timestamp:
                exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
                if datetime.now(timezone.utc) > exp_datetime:
                    raise HTTPException(
                        status_code=401,
                        detail={"result": {"idTipoMensaje": 1, "mensaje": "Token expirado"}}
                    )

            # Extract agent identification
            agent_id = payload.get('ID_AGENTE')
            role_id = payload.get('ID_TIPO_ROL')
            role_name = payload.get('ROL')
            company_areas = payload.get('company_areas', [])

            return {
                'ID_AGENTE': agent_id,
                'ID_TIPO_ROL': role_id,
                'ROL': role_name,
                'company_areas': company_areas
            }

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Token expirado"}}
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=401,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Token inválido"}}
            )
        except Exception as e:
            logger.error(f"Agent JWT verification error: {e}")
            raise HTTPException(
                status_code=401,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Error de autenticación"}}
            )

    @staticmethod
    def validate_agent_company_area_access(token: str, company_id: int, area_id: int) -> bool:
        """
        Validate agent access to company/area based on token data

        Agents (role_id 4) require both company_id and area_id to match (like Users role_id 3)

        Args:
            token: JWT token
            company_id: Requested company ID
            area_id: Requested area ID (required)

        Returns:
            True if access is allowed, False otherwise
        """
        try:
            # Validate input parameters - both required
            if not isinstance(company_id, int) or company_id <= 0:
                return False
            if not isinstance(area_id, int) or area_id <= 0:
                return False

            # Extract full payload
            payload = jwt.decode(
                token,
                settings.jwt_secret_key,
                algorithms=['HS256']
            )

            company_areas = payload.get('company_areas', [])

            # Agents: Validate both company_id and area_id exist in the same row
            return any(
                ca.get('ID_EMPRESA') == company_id and
                ca.get('ID_AREA') == area_id
                for ca in company_areas
            )

        except Exception as e:
            logger.error(f"Error validating agent company/area access: {e}")
            return False


# Dependency function
async def get_current_agent_with_company_area_validation(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """Get current agent and validate company/area access from request by company_id and area_id"""

    if not credentials:
        logger.warning("No agent JWT token found in Authorization header")
        raise HTTPException(
            status_code=401,
            detail={"result": {"idTipoMensaje": 1, "mensaje": "Token de autenticación requerido"}}
        )

    token = credentials.credentials

    # Verify JWT expiration and get agent info
    agent_data = AgentJWTAuth.verify_agent_jwt_token(token)

    # Extract company_id and area_id from request body
    if request.method == "POST":
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            # For JSON requests
            body = await request.json()
            company_id = body.get("company_id")
            area_id = body.get("area_id")
        else:
            raise HTTPException(status_code=400, detail="Unsupported content type")

        # Validate input parameters first
        if company_id is None:
            raise HTTPException(status_code=422, detail={"result": {"idTipoMensaje": 1, "mensaje": "company_id is required"}})
        if area_id is None:
            raise HTTPException(status_code=422, detail={"result": {"idTipoMensaje": 1, "mensaje": "area_id is required"}})

        # Validate agent has access to the requested company_id and area_id
        has_access = AgentJWTAuth.validate_agent_company_area_access(token, company_id, area_id)

        if not has_access:
            logger.warning(f"Access denied for agent {agent_data.get('ID_AGENTE')} to company_id: {company_id}, area_id: {area_id}")
            raise HTTPException(
                status_code=403,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Acceso denegado"}}
            )

    return agent_data
