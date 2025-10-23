from sqlalchemy.orm import Session
from app.models.user_models import LoginRequest, LoginResponse, UserInfo
from app.infrastructure.repositories.user_repository import UserRepository
from app.utils.jwt_auth import JWTAuth
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AuthService:
    """Authentication service for user login validation - stateless, singleton."""

    def __init__(self):
        """Initialize stateless AuthService - no db parameter."""
        pass

    async def verify_user_password(self, db: Session, usuario: str, password: str) -> bool:
        """Verify user password using SP_VERIFY_USER_PASS"""
        try:
            user_repo = UserRepository(db)
            auth_status = user_repo.verify_user_password_sp(usuario, password)
            return auth_status == 1

        except Exception as e:
            logger.error(f"Error in verify_user_password: {e}")
            return False
    
    async def get_user_data(self, db: Session, usuario: str) -> Optional[dict]:
        """Get user data using SP_USUARIO_LOGIN"""
        try:
            user_repo = UserRepository(db)
            # Get data from repository (only raw SP call)
            user_data, roles_data, company_areas_data = user_repo.get_user_data_sp(usuario)

            # Combine user data with roles and company areas (business logic in service)
            complete_user_data = {
                **user_data,
                'roles': roles_data,
                'company_areas': company_areas_data
            }

            return complete_user_data

        except Exception as e:
            logger.error(f"Error in get_user_data: {e}")
            return None
    
    async def authenticate_user(self, db: Session, login_request: LoginRequest) -> Optional[LoginResponse]:
        """
        Authenticate user credentials using separate stored procedure calls

        Args:
            db: Database session
            login_request: LoginRequest containing usuario and clave_acceso (plain text)

        Returns:
            LoginResponse with user details if successful, None if failed
        """
        try:
            # Step 1: Verify password
            is_valid = await self.verify_user_password(db, login_request.usuario, login_request.clave_acceso)

            if not is_valid:
                logger.warning(f"Password verification FAILED for user: {login_request.usuario}")
                return None

            # Step 2: Get user data
            user_data = await self.get_user_data(db, login_request.usuario)

            if not user_data:
                logger.error(f"Failed to get user data for: {login_request.usuario}")
                return None

            # Create JWT token
            jwt_token = JWTAuth.create_jwt_token(user_data)

            # Return response with JWT token
            return LoginResponse(
                token=jwt_token,
                status="success"
            )

        except Exception as e:
            logger.error(f"Database error during authentication: {e}")
            db.rollback()
            return None
    
    async def logout_user(self, db: Session, user_id: int) -> bool:
        """
        Update user connection status on logout

        Args:
            db: Database session
            user_id: ID of the user to logout

        Returns:
            True if successful, False otherwise
        """
        try:
            user_repo = UserRepository(db)
            return user_repo.update_connection_status(user_id, connected=False)
        except Exception as e:
            logger.error(f"Error during logout for user {user_id}: {e}")
            return False
    
    async def get_user_info(self, db: Session, user_id: int) -> Optional[UserInfo]:
        """
        Get user information by ID

        Args:
            db: Database session
            user_id: ID of the user

        Returns:
            UserInfo object if found, None otherwise
        """
        try:
            user_repo = UserRepository(db)
            user = user_repo.get_active_user_by_id(user_id)

            if user:
                return UserInfo(
                    id_usuario=user.ID_USUARIO,
                    usuario=user.USUARIO,
                    nombres=user.NOMBRES,
                    apellidos=user.APELLIDOS,
                    email=user.EMAIL,
                    ultimo_ingreso=user.ULTIMO_INGRESO,
                    id_estado_registro=user.ID_ESTADO_REGISTRO
                )
            return None
        except Exception as e:
            logger.error(f"Error getting user info for ID {user_id}: {e}")
            return None


    async def get_user_data_by_id(self, db: Session, user_id: int) -> Optional[dict]:
        """Get user data by user ID (similar to get_user_data but by ID)"""
        try:
            user_repo = UserRepository(db)
            # Get user from database
            user = user_repo.get_active_user_by_id(user_id)

            if not user:
                return None

            # Get user data using the stored procedure with username
            return await self.get_user_data(db, user.USUARIO)

        except Exception as e:
            logger.error(f"Error getting user data by ID {user_id}: {e}")
            return None

    async def get_user_company_areas(self, db: Session, user_id: int, role_id: int) -> Optional[list]:
        """Get user company areas using SP_USUARIO_EMPR_AREA_LST"""
        try:
            user_repo = UserRepository(db)
            return user_repo.get_user_company_areas_sp(user_id, role_id)

        except Exception as e:
            logger.error(f"Error getting user company areas for user ID {user_id}: {e}")
            return None


def get_auth_service() -> AuthService:
    """Get singleton AuthService instance - stateless, no db parameter."""
    return AuthService()