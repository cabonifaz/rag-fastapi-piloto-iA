from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base
from pydantic import BaseModel
from typing import Optional
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
    ID_SUCURSAL = Column(Integer, nullable=True)
    EMAIL = Column(String(150), nullable=True)
    ID_EMPRESA = Column(Integer, nullable=False)


class LoginRequest(BaseModel):
    """Request model for login endpoint"""
    usuario: str
    clave_acceso: str
    

class LoginResponse(BaseModel):
    """Response model for successful login"""
    user_id: int
    usuario: str
    nombres: str
    apellidos: str
    email: Optional[str] = None
    id_empresa: int
    id_sucursal: Optional[int] = None
    ultimo_ingreso: Optional[datetime] = None
    token: Optional[str] = None  # JWT token
    status: str = "success"
    # Role information for frontend display
    id_tipo_rol: int
    rol_nombre: str  # STRING1 from the role SP


class UserInfo(BaseModel):
    """User information model"""
    id_usuario: int
    usuario: str
    nombres: str
    apellidos: str
    email: Optional[str] = None
    id_empresa: int
    id_sucursal: Optional[int] = None
    ultimo_ingreso: Optional[datetime] = None
    id_estado_registro: int