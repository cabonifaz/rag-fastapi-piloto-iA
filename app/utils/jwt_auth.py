from fastapi import HTTPException, Depends
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
    def verify_jwt_token(token: str) -> Dict[str, Any]:
        """Verify JWT token and extract required user data"""
        try:
            payload = jwt.decode(
                token, 
                settings.jwt_secret_key, 
                algorithms=['HS256']
            )
            
            # Extract the required parameters
            user_id = payload.get('ID_USUARIO')
            username = payload.get('USUARIO')
            rol = payload.get('STRING1')  # rol name
            rol_id = payload.get('ID_TIPO_ROL')  # rol id
            company_areas = payload.get('company_areas', [])
            
            # Validate that all required parameters exist
            required_fields = [user_id, username, rol, rol_id]
            field_names = ['ID_USUARIO', 'USUARIO', 'STRING1 (rol)', 'ID_TIPO_ROL']
            
            missing = []
            for field, name in zip(required_fields, field_names):
                if field is None:
                    missing.append(name)
                    
            if missing:
                logger.warning(f"JWT missing required fields: {missing}")
                raise HTTPException(
                    status_code=401,
                    detail={"result": {"idTipoMensaje": 1, "mensaje": f"Token inválido - faltan campos: {missing}"}}
                )
            
            # Check expiration
            exp_timestamp = payload.get('exp')
            if exp_timestamp:
                exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
                if datetime.now(timezone.utc) > exp_datetime:
                    logger.warning(f"Token expired for user: {username}")
                    raise HTTPException(
                        status_code=401,
                        detail={"result": {"idTipoMensaje": 1, "mensaje": "Token expirado"}}
                    )
            
            # Return all validated JWT data
            extracted_data = {
                'ID_USUARIO': user_id,
                'USUARIO': username,
                'STRING1': rol,
                'ID_TIPO_ROL': rol_id,
                'company_areas': company_areas
            }
            
            return extracted_data
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            raise HTTPException(
                status_code=401,
                detail={"result": {"idTipoMensaje": 1, "mensaje": "Token expirado"}}
            )
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
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