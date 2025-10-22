from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
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
                from decimal import Decimal
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
    def validate_company_access(token: str, company: str, area: str = None) -> bool:
        """
        Validate user access to company/area based on role

        Args:
            token: JWT token
            company: Requested company name (matches EMPRESA field in token)
            area: Requested area name (matches AREA field in token, optional)

        Returns:
            True if access is allowed, False otherwise
        """
        try:
            # Validate input parameters
            if not company or not company.strip():
                return False
            if area is not None and (not area or not area.strip()):
                return False

            # Extract full payload
            payload = JWTAuth._extract_payload(token)

            role_id = payload.get('ID_TIPO_ROL')
            company_areas = payload.get('company_areas', [])

            # SuperAdmin (role_id = 1): Always allow access
            if role_id == 1:
                return True

            # Admin (role_id = 2): Validate company exists in any company_areas row
            if role_id == 2:
                return any(ca.get('EMPRESA') == company for ca in company_areas)

            # User (role_id = 3): Validate both company and area exist in the same row
            if role_id == 3:
                if not area:
                    return False
                return any(
                    ca.get('EMPRESA') == company and
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
            company = body.get("company")
            area = body.get("area")
        elif "multipart/form-data" in content_type:
            # For form requests (like upload)
            form = await request.form()
            company = form.get("company_name")
            area = form.get("area_name")
        else:
            raise HTTPException(status_code=400, detail="Unsupported content type")

        # Validate input parameters first
        if company is not None and (not company or not company.strip()):
            raise HTTPException(status_code=422, detail={"result": {"idTipoMensaje": 1, "mensaje": "Company ID cannot be empty"}})
        if area is not None and (not area or not area.strip()):
            raise HTTPException(status_code=422, detail={"result": {"idTipoMensaje": 1, "mensaje": "Area cannot be empty"}})

        # Validate company/area access
        if company and not JWTAuth.validate_company_access(token, company, area):
            logger.warning(f"Access denied for user {user_data.get('ID_USUARIO')} to company: {company}, area: {area}")
            raise HTTPException(
                status_code=403,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Acceso denegado"}}
            )

    return user_data