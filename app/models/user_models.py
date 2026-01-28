from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

Base = declarative_base()


class Usuario(Base):
    """SQLAlchemy model for USUARIOS table in SQL Server"""
    __tablename__ = "USUARIOS"
    __table_args__ = {'schema': 'dbo'}

    ID_USUARIO = Column(Integer, primary_key=True, index=True)
    USUARIO = Column(String(100), unique=True, index=True, nullable=False)
    CLAVE_ACCESO = Column(String(255), nullable=False)
    NOMBRES = Column(String(100), nullable=False)
    APELLIDOS = Column(String(100), nullable=False)
    ULTIMO_INGRESO = Column(DateTime, nullable=True)
    ID_CONECTADO = Column(Boolean, default=False)
    USUCRE = Column(String(50), nullable=True)
    FCHCRE = Column(DateTime, default=datetime.utcnow)
    ID_ESTADO_REGISTRO = Column(Integer, default=1)
    EMAIL = Column(String(150), nullable=True)


class LoginRequest(BaseModel):
    """Request model for login endpoint"""
    usuario: str
    clave_acceso: str
    ref: str



class LoginResponse(BaseModel):
    """Response model for successful login"""
    token: str  # JWT token containing user information
    status: str = "success"


class UserInfo(BaseModel):
    """User information model"""
    id_usuario: int
    usuario: str
    nombres: str
    apellidos: str
    email: Optional[str] = None
    ultimo_ingreso: Optional[datetime] = None
    id_estado_registro: int


class CreateUserRequest(BaseModel):
    """Request model for creating a new user"""
    nuevo_usuario: str
    password: str
    nombres: str
    apellidos: str
    codigo_pais: str
    telefono: str
    nuevo_rol: int
    id_empresa: int
    areas_string: str


class UpdateUserRequest(BaseModel):
    """Request model for updating user data"""
    id_usuario: int
    usuario: str
    nombres: str
    apellidos: str
    telefono: Optional[str] = None


class UpdateUserStatusRequest(BaseModel):
    """Request model for updating user status"""
    id_usuario: int
    status: int


class UpdateUserPasswordRequest(BaseModel):
    """Request model for updating user password"""
    id_usuario: int
    clave_acceso: str


class UpdateUserAccessRequest(BaseModel):
    """Request model for updating user role and areas access"""
    id_usuario: int
    nuevo_rol: int
    areas_string: str
    id_empresa: int