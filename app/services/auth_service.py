from sqlalchemy.orm import Session
from sqlalchemy import and_, text
from app.models.user_models import Usuario, LoginRequest, LoginResponse, UserInfo
from app.core.database import get_db
from app.core.config import settings
from datetime import datetime, timezone, timedelta
import logging
from typing import Optional
import jwt

logger = logging.getLogger(__name__)


class AuthService:
    """Authentication service for user login validation"""
    
    def __init__(self, db: Session):
        self.db = db
        # JWT configuration from settings
        self.jwt_secret = settings.jwt_secret_key
        self.jwt_expiration_hours = settings.jwt_expiration_hours
        self.jwt_algorithm = 'HS256'
    
    def create_jwt_token(self, user_data: dict) -> str:
        """Create JWT token with user data for frontend cookie storage"""
        try:
            # Extract role information from roles array
            role_name = 'User'  # Default role
            role_id = 1  # Default role ID
            if user_data.get('roles') and len(user_data['roles']) > 0:
                role_info = user_data['roles'][0]
                role_name = role_info.get('STRING1', 'User')
                role_id = role_info.get('ID_TIPO_ROL', 1)
            
            # Create JWT payload with all fields needed by frontend
            payload = {
                'ID_USUARIO': user_data.get('ID_USUARIO'),
                'USUARIO': user_data.get('USUARIO'),
                'NOMBRES': user_data.get('NOMBRES'),
                'APELLIDOS': user_data.get('APELLIDOS'),
                'ID_TIPO_ROL': role_id,
                'STRING1': role_name,
                'ID_EMPRESA': user_data.get('ID_EMPRESA'),
                'exp': datetime.now(timezone.utc) + timedelta(hours=self.jwt_expiration_hours),  # Configurable expiration
                'iat': datetime.now(timezone.utc),  # Issued at
                'iss': 'qamaq-rag-api'  # Issuer
            }
            
            # Create JWT token
            token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
            
            return token
            
        except Exception as e:
            logger.error(f"Error creating JWT token: {e}")
            raise
    
    async def verify_user_password(self, usuario: str, password: str) -> bool:
        """Verify user password using SP_VERIFY_USER_PASS"""
        try:
            query = text("""
                EXEC SP_VERIFY_USER_PASS 
                @Username = :username, 
                @Password = :password
            """)
            
            result = self.db.execute(query, {
                'username': usuario,
                'password': password
            })
            
            status_data = result.fetchone()
            result.close()
            
            if not status_data:
                logger.warning(f"No response from SP_VERIFY_USER_PASS for user: {usuario}")
                return False
            
            status_dict = dict(status_data._mapping) if hasattr(status_data, '_mapping') else dict(zip(result.keys(), status_data))
            auth_status = status_dict.get('Status', 0)

            return auth_status == 1
            
        except Exception as e:
            logger.error(f"Error in verify_user_password: {e}")
            return False
    
    async def get_user_data(self, usuario: str) -> Optional[dict]:
        """Get user data using SP_USUARIO_LOGIN"""
        try:
            query = text("""
                EXEC SP_USUARIO_LOGIN 
                @USUARIO = :usuario
            """)
            
            # Use raw connection to handle multiple result sets
            raw_conn = self.db.connection().connection
            cursor = raw_conn.cursor()
            
            try:
                cursor.execute("EXEC SP_USUARIO_LOGIN @USUARIO = ?", usuario)
                
                user_data = {}
                roles_data = []
                result_set_num = 1
                
                while True:
                    
                    try:
                        # Check if we have columns (indicating data)
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            
                            rows = cursor.fetchall()
                            
                            if result_set_num == 3 and rows:  # User data
                                user_row = rows[0]
                                user_data = dict(zip(columns, user_row))
                            elif result_set_num == 4 and rows:  # Role data
                                for row in rows:
                                    role_dict = dict(zip(columns, row))
                                    roles_data.append(role_dict)
                    
                    except Exception as fetch_error:
                        logger.error(f"Fetch error: {fetch_error}")
                    
                    # Move to next result set
                    try:
                        if not cursor.nextset():
                            break
                    except Exception as nextset_error:
                        logger.error(f"Nextset error: {nextset_error}")
                        break
                    
                    result_set_num += 1
                
                cursor.close()
                
                # Combine user data with roles
                complete_user_data = {
                    **user_data,
                    'roles': roles_data
                }
                
                return complete_user_data
                
            except Exception as cursor_error:
                logger.error(f"Cursor error: {cursor_error}")
                cursor.close()
                raise
            
        except Exception as e:
            logger.error(f"Error in get_user_data: {e}")
            return None
    
    async def authenticate_user(self, login_request: LoginRequest) -> Optional[LoginResponse]:
        """
        Authenticate user credentials using separate stored procedure calls
        
        Args:
            login_request: LoginRequest containing usuario and clave_acceso (plain text)
            
        Returns:
            LoginResponse with user details if successful, None if failed
        """
        try:
            # Step 1: Verify password
            is_valid = await self.verify_user_password(login_request.usuario, login_request.clave_acceso)
            
            if not is_valid:
                logger.warning(f"Password verification FAILED for user: {login_request.usuario}")
                return None
            
            # Step 2: Get user data
            user_data = await self.get_user_data(login_request.usuario)
            
            if not user_data:
                logger.error(f"Failed to get user data for: {login_request.usuario}")
                return None
            
            # Create JWT token
            jwt_token = self.create_jwt_token(user_data)
            
            # Extract role information for frontend display
            role_name = 'User'  # Default role
            role_id = 1  # Default role ID
            if user_data.get('roles') and len(user_data['roles']) > 0:
                role_info = user_data['roles'][0]
                role_name = role_info.get('STRING1', 'User')
                role_id = role_info.get('ID_TIPO_ROL', 1)
            
            # Return complete login response with real user data, JWT, and role info
            return LoginResponse(
                user_id=user_data.get('ID_USUARIO'),
                usuario=user_data.get('USUARIO'),
                nombres=user_data.get('NOMBRES'),
                apellidos=user_data.get('APELLIDOS'),
                email=None,  # Not provided by SP
                id_empresa=user_data.get('ID_EMPRESA'),
                id_sucursal=user_data.get('ID_SUCURSAL'),
                ultimo_ingreso=datetime.now(timezone.utc),
                token=jwt_token,  # JWT token for authentication
                status="success",
                id_tipo_rol=role_id,  # Role ID for permissions
                rol_nombre=role_name  # Role name for display
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