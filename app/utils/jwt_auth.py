from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
from decimal import Decimal
import jwt
from datetime import datetime, timezone, timedelta
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)  # Don't auto-error, let us handle it


class JWTAuth:
    """JWT Authentication utility"""

    @staticmethod
    def create_jwt_token(user_data: dict) -> str:
        """Create JWT token with user data for frontend storage"""
        try:
            # Helper function to convert Decimal objects to int/float
            def convert_decimal(obj):
                if isinstance(obj, Decimal):
                    return float(obj) if obj % 1 else int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_decimal(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_decimal(item) for item in obj]
                return obj

            # Convert user_data to handle Decimal objects
            clean_user_data = convert_decimal(user_data)

            # Extract role information from roles array
            role_name = 'User'  # Default role
            role_id = 1  # Default role ID
            if clean_user_data.get('roles') and len(clean_user_data['roles']) > 0:
                role_info = clean_user_data['roles'][0]
                role_name = role_info.get('STRING1', 'User')
                role_id = role_info.get('ID_TIPO_ROL', 1)

            # Create JWT payload with all fields needed by frontend
            payload = {
                'ID_USUARIO': clean_user_data.get('ID_USUARIO'),
                'USUARIO': clean_user_data.get('USUARIO'),
                'NOMBRES': clean_user_data.get('NOMBRES'),
                'APELLIDOS': clean_user_data.get('APELLIDOS'),
                'ID_TIPO_ROL': role_id,
                'ROL': role_name,
                'company_areas': clean_user_data.get('company_areas', []),  # Include all available company areas
                'exp': datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiration_minutes),
                'iat': datetime.now(timezone.utc),  # Issued at
                'iss': 'qamaq-rag-api'  # Issuer
            }

            # Create JWT token
            token = jwt.encode(payload, settings.jwt_secret_key, algorithm='HS256')

            return token

        except Exception as e:
            logger.error(f"Error creating JWT token: {e}")
            raise

    @staticmethod
    def create_n8n_jwt_token(user_id: int, jwt_secret: str) -> str:
        """
        Create JWT token for n8n webhook authentication

        Args:
            user_id: User ID from the current user token
            jwt_secret: Secret key for n8n JWT (from N8N_CC_JWT_SECRET env var)

        Returns:
            JWT token string for n8n webhook
        """
        try:
            payload = {
                "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
                "iat": datetime.now(timezone.utc),
                "user_id": user_id
            }

            token = jwt.encode(payload, jwt_secret, algorithm='HS256')
            return token

        except Exception as e:
            logger.error(f"Error creating n8n JWT token: {e}")
            raise

    @staticmethod
    def _extract_payload(token: str) -> Dict[str, Any]:
        """Extract JWT payload without validation"""
        return jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=['HS256']
        )

    @staticmethod
    def verify_jwt_token(token: str) -> Dict[str, Any]:
        """Verify JWT token expiration only"""
        try:
            payload = JWTAuth._extract_payload(token)

            # Check expiration
            exp_timestamp = payload.get('exp')
            if exp_timestamp:
                exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
                if datetime.now(timezone.utc) > exp_datetime:
                    raise HTTPException(
                        status_code=401,
                        detail={"result": {"idTipoMensaje": 1, "mensaje": "Token expirado"}}
                    )

            # Extract user identification
            user_id = payload.get('ID_USUARIO')
            username = payload.get('USUARIO')
            role_id = payload.get('ID_TIPO_ROL')

            return {
                'ID_USUARIO': user_id,
                'USUARIO': username,
                'ID_TIPO_ROL': role_id
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
            logger.error(f"JWT verification error: {e}")
            raise HTTPException(
                status_code=401,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Error de autenticación"}}
            )

    @staticmethod
    def validate_company_access(token: str, company_id: int) -> bool:
        """
        Validate user access to company based on role

        Args:
            token: JWT token
            company_id: Requested company ID (matches ID_EMPRESA field in token)

        Returns:
            True if access is allowed, False otherwise
        """
        try:
            # Validate input parameters
            if not isinstance(company_id, int) or company_id <= 0:
                return False

            # Extract full payload
            payload = JWTAuth._extract_payload(token)

            role_id = payload.get('ID_TIPO_ROL')
            company_areas = payload.get('company_areas', [])

            # SuperAdmin (role_id = 1): Always allow access
            if role_id == 1:
                return True

            # Admin (role_id = 2): Validate company_id exists in any company_areas row
            if role_id == 2:
                return any(ca.get('ID_EMPRESA') == company_id for ca in company_areas)

            # User (role_id = 3): Validate company_id exists in any company_areas row
            if role_id == 3:
                return any(ca.get('ID_EMPRESA') == company_id for ca in company_areas)

            # Unknown role: Deny access
            return False

        except Exception as e:
            logger.error(f"Error validating company access: {e}")
            return False

    @staticmethod
    def validate_company_area_access(token: str, company_id: int, area_id: int = None) -> bool:
        """
        Validate user access to company/area based on role

        Args:
            token: JWT token
            company_id: Requested company ID (matches ID_EMPRESA field in token)
            area_id: Requested area ID (matches ID_AREA field in token, optional)

        Returns:
            True if access is allowed, False otherwise
        """
        try:
            # Validate input parameters
            if not isinstance(company_id, int) or company_id <= 0:
                return False
            if area_id is not None and (not isinstance(area_id, int) or area_id <= 0):
                return False

            # Extract full payload
            payload = JWTAuth._extract_payload(token)

            role_id = payload.get('ID_TIPO_ROL')
            company_areas = payload.get('company_areas', [])

            # SuperAdmin (role_id = 1): Always allow access
            if role_id == 1:
                return True

            # Admin (role_id = 2): Validate company_id exists in any company_areas row
            if role_id == 2:
                return any(ca.get('ID_EMPRESA') == company_id for ca in company_areas)

            # User (role_id = 3 or 4): Validate both company_id and area_id exist in the same row
            if role_id in [3, 4]:
                if area_id is None:
                    return False
                return any(
                    ca.get('ID_EMPRESA') == company_id and
                    ca.get('ID_AREA') == area_id
                    for ca in company_areas
                )

            # Unknown role: Deny access
            return False

        except Exception as e:
            logger.error(f"Error validating company/area access: {e}")
            return False

# Dependency function
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """Get current user from JWT token in Authorization header"""

    if not credentials:
        logger.warning("No JWT token found in Authorization header")
        raise HTTPException(
            status_code=401,
            detail={"result": {"idTipoMensaje": 1, "mensaje": "Token de autenticación requerido"}}
        )

    return JWTAuth.verify_jwt_token(credentials.credentials)

async def get_current_user_with_company_validation(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """Get current user and validate company access from request by ID_EMPRESA"""

    if not credentials:
        logger.warning("No JWT token found in Authorization header")
        raise HTTPException(
            status_code=401,
            detail={"result": {"idTipoMensaje": 1, "mensaje": "Token de autenticación requerido"}}
        )

    token = credentials.credentials

    # Verify JWT expiration and get user info
    user_data = JWTAuth.verify_jwt_token(token)

    # Extract company_id from request body
    if request.method == "POST":
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            # For JSON requests
            body = await request.json()
            company_id = body.get("id_empresa")
        else:
            raise HTTPException(status_code=400, detail="Unsupported content type")

        # Validate input parameters first
        if company_id is None:
            raise HTTPException(status_code=422, detail={"result": {"idTipoMensaje": 1, "mensaje": "id_empresa is required"}})

        # Validate user has access to the requested company_id
        has_access = JWTAuth.validate_company_access(token, company_id)

        if not has_access:
            logger.warning(f"Access denied for user {user_data.get('ID_USUARIO')} to company_id: {company_id}")
            raise HTTPException(
                status_code=403,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Acceso denegado"}}
            )

    return user_data

async def get_current_user_with_company_area_validation(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """Get current user and validate company/area access from request by ID_EMPRESA and ID_AREA"""
    if not credentials:
        logger.warning("No JWT token found in Authorization header")
        raise HTTPException(
            status_code=401,
            detail={"result": {"idTipoMensaje": 1, "mensaje": "Token de autenticación requerido"}}
        )

    token = credentials.credentials

    # Verify JWT expiration and get user info
    user_data = JWTAuth.verify_jwt_token(token)

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

        # Validate user has access to the requested company_id and area_id
        has_access = JWTAuth.validate_company_area_access(token, company_id, area_id)

        if not has_access:
            logger.warning(f"Access denied for user {user_data.get('ID_USUARIO')} to company_id: {company_id}, area_id: {area_id}")
            raise HTTPException(
                status_code=403,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Acceso denegado"}}
            )

    return user_data