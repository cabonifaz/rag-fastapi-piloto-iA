from pydantic import BaseModel
from typing import Optional


class CreateAgentRequest(BaseModel):
    """Request model for creating a new agent"""
    numero_telf: str
    id_tipo_agente: int
    id_empresa: int
    acceso_general: int
    areas_string: str