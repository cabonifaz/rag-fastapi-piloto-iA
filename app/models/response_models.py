from pydantic import BaseModel
from typing import Optional, Any, Dict


class MensajeResponse(BaseModel):
    """Standard response message format expected by frontend"""
    idTipoMensaje: int  # 1 = error/warning, 2 = success
    mensaje: str


class StandardResponse(BaseModel):
    """Base response wrapper that includes MensajeResponse result"""
    result: MensajeResponse
    

class StandardResponseWithData(BaseModel):
    """Response wrapper that includes both result and additional data"""
    result: MensajeResponse
    data: Optional[Dict[str, Any]] = None


def create_success_response(mensaje: str = "Operación exitosa") -> MensajeResponse:
    """Helper to create success response"""
    return MensajeResponse(idTipoMensaje=2, mensaje=mensaje)


def create_error_response(mensaje: str) -> MensajeResponse:
    """Helper to create error response"""
    return MensajeResponse(idTipoMensaje=1, mensaje=mensaje)


def create_warning_response(mensaje: str) -> MensajeResponse:
    """Helper to create warning response"""
    return MensajeResponse(idTipoMensaje=1, mensaje=mensaje)