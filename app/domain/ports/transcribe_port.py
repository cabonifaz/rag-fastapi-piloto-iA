from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional


class TranscribePort(ABC):
    """
    Puerto (interfaz) para servicios de transcripción de voz a texto.
    Define cómo la aplicación interactúa con cualquier proveedor de transcripción.

    🚨 IMPORTANTE: Cada instancia de implementación = UNA sesión de usuario
        → Crear NUEVA instancia por cada conexión WebSocket
        → NO reutilizar instancias entre diferentes usuarios
    """

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language_code: str = "es-ES",
        sample_rate: int = 16000,
        media_encoding: str = "pcm",
        show_speaker_label: bool = False,
        enable_channel_identification: bool = False,
        number_of_channels: Optional[int] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Transcribe audio stream en tiempo real.

        Args:
            audio_stream: Generador asíncrono que produce bytes de audio (formato PCM)
            language_code: Código de idioma (ej: 'en-US', 'es-US', 'pt-BR')
            sample_rate: Frecuencia de muestreo en Hz (8000 o 16000)
            media_encoding: Formato de codificación de audio ('pcm', 'ogg-opus', 'flac')
            show_speaker_label: Habilitar identificación de hablantes
            enable_channel_identification: Habilitar identificación de canales
            number_of_channels: Número de canales de audio (1 o 2)

        Yields:
            dict: Resultados de transcripción con formato:
                {
                    "transcript": str,              # Texto transcrito
                    "is_partial": bool,             # True si es resultado parcial
                    "start_time": float,            # Tiempo de inicio en segundos
                    "end_time": float,              # Tiempo de fin en segundos
                    "confidence": Optional[float],  # Puntuación de confianza (0.0-1.0)
                    "speaker_label": Optional[str], # ID del hablante si está habilitado
                    "alternatives": List[dict]      # Transcripciones alternativas
                }
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Cierra el stream de transcripción y limpia recursos.
        DEBE ser llamado cuando la sesión de transcripción termina.
        """
        pass
