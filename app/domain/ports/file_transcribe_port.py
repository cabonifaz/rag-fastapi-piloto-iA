from abc import ABC, abstractmethod
from typing import Dict, Any


class FileTranscribePort(ABC):
    """
    Puerto (interfaz) para servicios de transcripción de archivos de audio.

    Esta interfaz es para proveedores de transcripción BATCH (no streaming)
    que procesan archivos completos de audio, como OpenAI gpt-4o-mini-transcribe.

    Diferencias con TranscribePort:
    - Recibe archivo completo (bytes) en lugar de stream
    - Retorna UN solo resultado final (no resultados parciales)
    - No requiere WebSocket
    - Optimizado para archivos subidos desde frontend
    """

    @abstractmethod
    async def transcribe_file(
        self,
        audio_data: bytes,
        language_code: str = "es-ES",
        filename: str = "audio.wav"
    ) -> Dict[str, Any]:
        """
        Transcribe un archivo de audio completo.

        Args:
            audio_data: Bytes del archivo de audio
            language_code: Código de idioma (ej: 'es-ES', 'en-US', 'pt-BR')
            filename: Nombre del archivo con extensión (para determinar formato)

        Returns:
            dict: Resultado de transcripción con formato:
                {
                    "transcript": str,              # Texto transcrito
                    "language": str,                # Idioma detectado
                    "duration": float,              # Duración en segundos
                    "confidence": Optional[float],  # Confianza (si está disponible)
                    "model": str                    # Modelo utilizado
                }
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Cierra el cliente y limpia recursos.
        DEBE ser llamado cuando ya no se necesita el proveedor.
        """
        pass
