"""Service for file-based transcription (OpenAI)."""

from typing import Optional, Dict, Any
import logging
from datetime import datetime

from app.domain.ports.file_transcribe_port import FileTranscribePort

logger = logging.getLogger(__name__)


class FileTranscribeService:
    """
    Service for file-based audio transcription (OpenAI).

    This service handles transcription of complete audio files uploaded
    from the frontend, using batch providers like OpenAI gpt-4o-mini-transcribe.

    Differences from TranscribeService:
    - Works with complete files, not audio streams
    - Uses FileTranscribePort (not TranscribePort)
    - Returns single result (no partial results)
    - No WebSocket required
    """

    def __init__(self, file_transcribe_port: FileTranscribePort):
        """
        Initialize file transcribe service with a file transcription provider.

        Args:
            file_transcribe_port: File transcription provider (e.g., OpenAI)
        """
        self.file_transcribe_port = file_transcribe_port

    async def transcribe_file(
        self,
        audio_file: bytes,
        language_code: str = "es",
        filename: str = "audio.wav",
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.

        Args:
            audio_file: Audio file bytes
            language_code: Language code (e.g., 'es', 'en', 'de')
            filename: Filename with extension (determines audio format)
            user_id: ID del usuario (opcional, para logging)

        Returns:
            Dict with transcription result:
                {
                    "transcript": str,
                    "language": str,
                    "duration": float,
                    "confidence": Optional[float],
                    "model": str
                }
        """
        try:
            start_time = datetime.now()
            logger.info(
                f"Starting file transcription: "
                f"user_id={user_id}, "
                f"filename={filename}, "
                f"language={language_code}, "
                f"size={len(audio_file)} bytes"
            )

            # Validate file size
            if len(audio_file) == 0:
                raise ValueError("Audio file is empty")

            if len(audio_file) > 25 * 1024 * 1024:
                raise ValueError(f"Audio file too large: {len(audio_file)} bytes (max 25MB)")

            # Use the file transcribe port to process the file
            result = await self.file_transcribe_port.transcribe_file(
                audio_data=audio_file,
                language_code=language_code,
                filename=filename
            )

            if not result:
                raise ValueError("No transcription result received from provider")

            if not result.get('transcript'):
                raise ValueError("Empty transcription result received")

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"File transcription complete: "
                f"user_id={user_id}, "
                f"duration={duration:.2f}s, "
                f"chars={len(result['transcript'])}, "
                f"detected_language={result.get('language')}"
            )

            return result

        except ValueError as e:
            logger.error(f"Validation error in file transcription: {e}")
            raise

        except ConnectionError as e:
            logger.error(f"Connection error in file transcription: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error in file transcription: {e}", exc_info=True)
            raise ConnectionError(f"File transcription error: {str(e)}")

    async def close(self) -> None:
        """
        Close transcription provider and cleanup resources.
        """
        try:
            await self.file_transcribe_port.close()
            logger.info("File transcribe provider closed")
        except Exception as e:
            logger.error(f"Error closing file transcribe provider: {e}")
