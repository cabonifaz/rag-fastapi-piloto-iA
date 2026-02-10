from pydantic import BaseModel
from typing import Optional


class CreateAgentRequest(BaseModel):
    """Request model for creating a new agent"""
    numero_telf: str
    codigo_pais: str
    id_tipo_agente: int
    id_empresa: int
    acceso_general: int
    areas_string: str


class UpdateAgentRequest(BaseModel):
    """Request model for updating agent data"""
    id_agente: int
    numero_telf: str
    codigo_pais: str
    id_tipo_agente: int
    acceso_general: int


class UpdateAgentStatusRequest(BaseModel):
    """Request model for updating agent status"""
    id_agente: int
    status: int


class UpdateAgentOperativoRequest(BaseModel):
    """Request model for updating agent operative status"""
    id_agente: int
    operativo: int


class UpdateAgentSecretKeyRequest(BaseModel):
    """Request model for updating agent secret key"""
    id_agente: int


class UpdateAgentAccessRequest(BaseModel):
    """Request model for updating agent area access"""
    id_agente: int
    id_empresa: int
    areas_string: str


class AgentLoginRequest(BaseModel):
    """Request model for agent login"""
    numero_telf: str
    secret_key: str


class AgentLoginResponse(BaseModel):
    """Response model for agent login"""
    token: Optional[str] = None
    id_tipo_mensaje: int
    mensaje: str