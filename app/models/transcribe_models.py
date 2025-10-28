"""Transcription models for WebSocket-based speech-to-text."""

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime


# =============================================
# Transcription Configuration Models
# =============================================

class TranscribeConfig(BaseModel):
    """
    Configuración para iniciar una sesión de transcripción.
    Enviado por el cliente al establecer la conexión WebSocket.
    """
    model_config = ConfigDict(from_attributes=True)

    language_code: str = Field(
        default="es-ES",
        description="Código de idioma (es-ES, en-US, pt-BR, etc.)"
    )
    sample_rate: int = Field(
        default=16000,
        description="Frecuencia de muestreo en Hz (8000, 16000, 44100, 48000)"
    )
    media_encoding: str = Field(
        default="pcm",
        description="Formato de codificación de audio (pcm, ogg-opus, flac)"
    )
    vocabulary_name: Optional[str] = Field(
        default=None,
        description="Nombre de vocabulario personalizado de AWS Transcribe"
    )
    enable_partial_results: bool = Field(
        default=True,
        description="Habilitar resultados parciales durante la transcripción"
    )
    show_speaker_label: bool = Field(
        default=False,
        description="Habilitar identificación de hablantes (speaker diarization)"
    )
    enable_channel_identification: bool = Field(
        default=False,
        description="Habilitar identificación de canales de audio"
    )
    number_of_channels: Optional[int] = Field(
        default=None,
        description="Número de canales de audio (1 o 2)"
    )

    @field_validator('sample_rate')
    @classmethod
    def validate_sample_rate(cls, v):
        """Validar que la frecuencia de muestreo sea soportada."""
        valid_rates = [8000, 16000, 44100, 48000]
        if v not in valid_rates:
            raise ValueError(f"Sample rate must be one of {valid_rates}")
        return v

    @field_validator('media_encoding')
    @classmethod
    def validate_media_encoding(cls, v):
        """Validar que el formato de codificación sea soportado."""
        valid_encodings = ['pcm', 'ogg-opus', 'flac']
        if v not in valid_encodings:
            raise ValueError(f"Media encoding must be one of {valid_encodings}")
        return v

    @field_validator('language_code')
    @classmethod
    def validate_language_code(cls, v):
        """Validar que el código de idioma tenga el formato correcto."""
        # Verificar formato básico xx-XX
        if not v or len(v) < 5 or v[2] != '-':
            raise ValueError("Language code must be in format 'xx-XX' (e.g., 'es-ES', 'en-US')")
        return v


# =============================================
# Transcription Result Models
# =============================================

class TranscriptAlternative(BaseModel):
    """Alternativa de transcripción con su confianza."""
    model_config = ConfigDict(from_attributes=True)

    transcript: str = Field(description="Texto transcrito")
    confidence: Optional[float] = Field(
        default=None,
        description="Nivel de confianza (0.0-1.0)"
    )
    items: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Items individuales con timestamps"
    )


class TranscriptResult(BaseModel):
    """
    Resultado de transcripción procesado.
    Enviado desde el servidor al cliente vía WebSocket.
    """
    model_config = ConfigDict(from_attributes=True)

    transcript: str = Field(description="Texto transcrito")
    is_partial: bool = Field(description="True si es resultado parcial, False si es final")
    start_time: float = Field(description="Tiempo de inicio en segundos")
    end_time: float = Field(description="Tiempo de fin en segundos")
    confidence: Optional[float] = Field(
        default=None,
        description="Nivel de confianza promedio (0.0-1.0), solo en resultados finales"
    )
    alternatives: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Transcripciones alternativas"
    )
    speaker_label: Optional[str] = Field(
        default=None,
        description="Identificador del hablante si está habilitado"
    )


# =============================================
# WebSocket Message Models
# =============================================

class TranscribeWebSocketMessage(BaseModel):
    """
    Mensaje del cliente al servidor vía WebSocket.
    Define el protocolo de comunicación.
    """
    model_config = ConfigDict(from_attributes=True)

    type: str = Field(
        description="Tipo de mensaje: 'config', 'audio', 'stop'"
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Datos del mensaje (para tipo 'config')"
    )

    @field_validator('type')
    @classmethod
    def validate_message_type(cls, v):
        """Validar que el tipo de mensaje sea válido."""
        valid_types = ['config', 'audio', 'stop']
        if v not in valid_types:
            raise ValueError(f"Message type must be one of {valid_types}")
        return v


class TranscribeResponse(BaseModel):
    """
    Respuesta del servidor al cliente vía WebSocket.
    Puede contener resultados, errores o mensajes de estado.
    """
    model_config = ConfigDict(from_attributes=True)

    type: str = Field(
        description="Tipo de respuesta: 'partial', 'final', 'error', 'status'"
    )
    result: Optional[TranscriptResult] = Field(
        default=None,
        description="Resultado de transcripción (si type='partial' o 'final')"
    )
    error: Optional[str] = Field(
        default=None,
        description="Mensaje de error (si type='error')"
    )
    status: Optional[str] = Field(
        default=None,
        description="Mensaje de estado (si type='status')"
    )

    @field_validator('type')
    @classmethod
    def validate_response_type(cls, v):
        """Validar que el tipo de respuesta sea válido."""
        valid_types = ['partial', 'final', 'error', 'status']
        if v not in valid_types:
            raise ValueError(f"Response type must be one of {valid_types}")
        return v


# =============================================
# Optional: Transcript Storage Models (DynamoDB)
# =============================================

class TranscriptSession(BaseModel):
    """
    Modelo para almacenar sesiones de transcripción en DynamoDB (opcional).
    Útil para histórico y análisis.
    """
    model_config = ConfigDict(from_attributes=True)

    user_id: int = Field(description="ID del usuario")
    session_id: str = Field(description="ID único de la sesión de transcripción")
    transcript: str = Field(description="Texto completo transcrito")
    language_code: str = Field(description="Idioma usado en la transcripción")
    duration_seconds: float = Field(description="Duración de la sesión en segundos")
    average_confidence: Optional[float] = Field(
        default=None,
        description="Confianza promedio de la transcripción"
    )
    word_count: int = Field(description="Número de palabras transcritas")
    created_at: str = Field(description="Timestamp de creación (ISO format)")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadatos adicionales de la sesión"
    )


# =============================================
# Helper Functions
# =============================================

def create_status_response(status: str) -> TranscribeResponse:
    """Helper para crear respuesta de estado."""
    return TranscribeResponse(type="status", status=status)


def create_error_response(error: str) -> TranscribeResponse:
    """Helper para crear respuesta de error."""
    return TranscribeResponse(type="error", error=error)


def create_partial_response(result: TranscriptResult) -> TranscribeResponse:
    """Helper para crear respuesta parcial."""
    return TranscribeResponse(type="partial", result=result)


def create_final_response(result: TranscriptResult) -> TranscribeResponse:
    """Helper para crear respuesta final."""
    return TranscribeResponse(type="final", result=result)
