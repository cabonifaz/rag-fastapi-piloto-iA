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


class AgentLoginRequest(BaseModel):
    """Request model for agent login"""
    numero_telf: str
    secret_key: str


class AgentLoginResponse(BaseModel):
    """Response model for agent login"""
    token: Optional[str] = None
    id_tipo_mensaje: int
    mensaje: str