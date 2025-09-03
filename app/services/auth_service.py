from sqlalchemy.orm import Session
from sqlalchemy import and_
from app.models.user_models import Usuario, LoginRequest, LoginResponse, UserInfo
from app.core.database import get_db
from app.utils.password_utils import PasswordUtils
from datetime import datetime
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AuthService:
    """Authentication service for user login validation"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def _verify_password(self, stored_password: str, provided_password: str) -> bool:
        """Verify password against stored hash using centralized password utilities"""
        return PasswordUtils.verify_password(stored_password, provided_password)
    
    async def authenticate_user(self, login_request: LoginRequest) -> Optional[LoginResponse]:
        """
        Authenticate user credentials against SQL Server database
        
        Args:
            login_request: LoginRequest containing usuario and clave_acceso
            
        Returns:
            LoginResponse with user details if successful, None if failed
        """
        try:
            # Query user from database
            user = self.db.query(Usuario).filter(
                and_(
                    Usuario.USUARIO == login_request.usuario,
                    Usuario.ID_ESTADO_REGISTRO == 1  # Only active users
                )
            ).first()
            
            if not user:
                logger.warning(f"User not found: {login_request.usuario}")
                return None
            
            # Verify password
            if not self._verify_password(user.CLAVE_ACCESO, login_request.clave_acceso):
                logger.warning(f"Invalid password for user: {login_request.usuario}")
                return None
            
            # Update last login timestamp
            user.ULTIMO_INGRESO = datetime.utcnow()
            user.ID_CONECTADO = True
            self.db.commit()
            
            logger.info(f"User authenticated successfully: {login_request.usuario}")
            
            # Return successful login response
            return LoginResponse(
                user_id=user.ID_USUARIO,
                usuario=user.USUARIO,
                nombres=user.NOMBRES,
                apellidos=user.APELLIDOS,
                email=user.EMAIL,
                id_empresa=user.ID_EMPRESA,
                id_sucursal=user.ID_SUCURSAL,
                ultimo_ingreso=user.ULTIMO_INGRESO,
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Database error during authentication: {e}")
            self.db.rollback()
            return None
    
    async def logout_user(self, user_id: int) -> bool:
        """
        Update user connection status on logout
        
        Args:
            user_id: ID of the user to logout
            
        Returns:
            True if successful, False otherwise
        """
        try:
            user = self.db.query(Usuario).filter(Usuario.ID_USUARIO == user_id).first()
            if user:
                user.ID_CONECTADO = False
                self.db.commit()
                logger.info(f"User logged out: {user.USUARIO}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error during logout for user {user_id}: {e}")
            self.db.rollback()
            return False
    
    async def get_user_info(self, user_id: int) -> Optional[UserInfo]:
        """
        Get user information by ID
        
        Args:
            user_id: ID of the user
            
        Returns:
            UserInfo object if found, None otherwise
        """
        try:
            user = self.db.query(Usuario).filter(
                and_(
                    Usuario.ID_USUARIO == user_id,
                    Usuario.ID_ESTADO_REGISTRO == 1
                )
            ).first()
            
            if user:
                return UserInfo(
                    id_usuario=user.ID_USUARIO,
                    usuario=user.USUARIO,
                    nombres=user.NOMBRES,
                    apellidos=user.APELLIDOS,
                    email=user.EMAIL,
                    id_empresa=user.ID_EMPRESA,
                    id_sucursal=user.ID_SUCURSAL,
                    ultimo_ingreso=user.ULTIMO_INGRESO,
                    id_estado_registro=user.ID_ESTADO_REGISTRO
                )
            return None
        except Exception as e:
            logger.error(f"Error getting user info for ID {user_id}: {e}")
            return None


def get_auth_service(db: Session = next(get_db())) -> AuthService:
    """Dependency injection for AuthService"""
    return AuthService(db)