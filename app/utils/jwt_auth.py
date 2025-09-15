from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import jwt
from datetime import datetime, timezone
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)  # Don't auto-error, let us handle it


class JWTAuth:
    """JWT Authentication utility"""

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

            return {
                'ID_USUARIO': user_id,
                'USUARIO': username
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
    def validate_company_access(token: str, company_id: str, area: str = None) -> bool:
        """
        Validate user access to company/area based on role

        Args:
            token: JWT token
            company_id: Requested company name (matches EMPRESA field in token)
            area: Requested area name (matches AREA field in token, optional)

        Returns:
            True if access is allowed, False otherwise
        """
        try:
            # Validate input parameters
            if not company_id or not company_id.strip():
                return False
            if area is not None and (not area or not area.strip()):
                return False

            # Extract full payload
            payload = JWTAuth._extract_payload(token)

            role = payload.get('STRING1', '').lower()
            company_areas = payload.get('company_areas', [])

            # SuperAdmin: Always allow access
            if 'super admin' in role:
                return True

            # Admin: Validate company_id exists in any company_areas row
            if role == 'admin':
                return any(ca.get('EMPRESA') == company_id for ca in company_areas)

            # User: Validate both company_id and area exist in the same row
            if role == 'user':
                if not area:
                    return False
                return any(
                    ca.get('EMPRESA') == company_id and
                    ca.get('AREA') == area
                    for ca in company_areas
                )

            # Unknown role: Deny access
            return False

        except Exception as e:
            logger.error(f"Error validating company access: {e}")
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
    """Get current user and validate company/area access from request"""
    from fastapi import Request

    if not credentials:
        logger.warning("No JWT token found in Authorization header")
        raise HTTPException(
            status_code=401,
            detail={"result": {"idTipoMensaje": 1, "mensaje": "Token de autenticación requerido"}}
        )

    token = credentials.credentials

    # Verify JWT expiration and get user info
    user_data = JWTAuth.verify_jwt_token(token)

    # Extract company/area from request body
    if request.method == "POST":
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            # For JSON requests (like chat-streaming)
            body = await request.json()
            company_id = body.get("company_id")
            area = body.get("area")
        elif "multipart/form-data" in content_type:
            # For form requests (like upload)
            form = await request.form()
            company_id = form.get("company_name")
            area = form.get("area_name")
        else:
            raise HTTPException(status_code=400, detail="Unsupported content type")

        # Validate input parameters first
        if company_id is not None and (not company_id or not company_id.strip()):
            raise HTTPException(status_code=422, detail={"result": {"idTipoMensaje": 1, "mensaje": "Company ID cannot be empty"}})
        if area is not None and (not area or not area.strip()):
            raise HTTPException(status_code=422, detail={"result": {"idTipoMensaje": 1, "mensaje": "Area cannot be empty"}})

        # Validate company/area access
        if company_id and not JWTAuth.validate_company_access(token, company_id, area):
            logger.warning(f"Access denied for user {user_data.get('ID_USUARIO')} to company: {company_id}, area: {area}")
            raise HTTPException(
                status_code=403,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Acceso denegado"}}
            )

    return user_data