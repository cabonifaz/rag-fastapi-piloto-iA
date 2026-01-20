"""
OpenAI GPT-4o-mini-transcribe provider for file-based audio transcription.

Uses OpenAI's Audio API (REST endpoint) with gpt-4o-mini-transcribe model.
This is a NON-STREAMING implementation for batch file transcription.
"""

import logging
import asyncio
import io
from typing import Optional, AsyncGenerator, Dict, Any, BinaryIO
import httpx
from app.domain.ports.file_transcribe_port import FileTranscribePort
from app.core.config import settings

logger = logging.getLogger(__name__)


class OpenAITranscribe(FileTranscribePort):
    """
    OpenAI GPT-4o-mini-transcribe provider using REST API for file-based transcription.

    This implementation:
    - Uses /v1/audio/transcriptions endpoint (REST API)
    - Does NOT use WebSocket (batch processing only)
    - Accumulates audio chunks into a file and transcribes when complete
    - Supports audio files up to 25MB

    Note: This is designed for non-realtime use cases where you have
    complete audio files or can accumulate chunks before transcription.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini-transcribe",
        timeout: int = 60,
    ):
        """
        Initialize OpenAI Transcribe client.

        Args:
            api_key: OpenAI API key (defaults to settings.openai_api_key)
            base_url: OpenAI API base URL
            model: Model to use (gpt-4o-mini-transcribe or gpt-4o-transcribe)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or getattr(settings, 'openai_api_key', None)
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in environment.")

        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}"
            }
        )

        logger.info(f"OpenAI Transcribe initialized with model: {model}")

    async def transcribe_file(
        self,
        audio_data: bytes,
        language_code: str = "es",
        filename: str = "audio.wav"
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using OpenAI API.

        Args:
            audio_data: Audio file bytes
            language_code: Language code (e.g., 'es', 'en', 'de')
            filename: Filename with extension (determines audio format)

        Returns:
            Dict with transcription result:
                {
                    "transcript": str,
                    "language": str,
                    "duration": float,
                    "confidence": None,
                    "model": str
                }
        """
        try:
            logger.info(f"Transcribing file: {filename}, size: {len(audio_data)} bytes, language: {language_code}")

            # Check size limit (25MB for OpenAI)
            if len(audio_data) > 25 * 1024 * 1024:
                raise ValueError(f"Audio file too large: {len(audio_data)} bytes (max 25MB)")

            # Detect MIME type from filename
            mime_type = self._get_mime_type(filename)

            # Prepare multipart form data
            files = {
                'file': (filename, audio_data, mime_type)
            }

            data = {
                'model': self.model,
                'response_format': 'json',  # Get detailed response with language and duration
                'language': language_code  # Language code (e.g., 'es', 'en', 'de')
            }

            # Make API request
            response = await self.client.post(
                f"{self.base_url}/audio/transcriptions",
                files=files,
                data=data
            )

            response.raise_for_status()
            result = response.json()

            # Parse OpenAI response
            transcript = result.get('text', '')
            detected_language = result.get('language', language_code or 'unknown')
            duration = result.get('duration', 0.0)

            logger.info(f"Transcription complete: {len(transcript)} chars, duration: {duration}s, language: {detected_language}")

            # Return result matching FileTranscribePort interface
            return {
                "transcript": transcript,
                "language": detected_language,
                "duration": duration,
                "confidence": None,  # OpenAI doesn't provide confidence scores
                "model": self.model
            }

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_detail = e.response.text

            logger.error(f"OpenAI API error {status_code}: {error_detail}")

            if status_code == 400:
                raise ValueError(f"Invalid audio format or parameters: {error_detail}")
            elif status_code == 401:
                raise ConnectionError("Invalid OpenAI API key")
            elif status_code == 413:
                raise ValueError("Audio file too large (max 25MB)")
            elif status_code == 429:
                raise ConnectionError("OpenAI rate limit exceeded")
            elif status_code >= 500:
                raise ConnectionError(f"OpenAI service error: {status_code}")
            else:
                raise ConnectionError(f"OpenAI API error {status_code}: {error_detail}")

        except httpx.TimeoutException as e:
            logger.error(f"OpenAI API timeout: {e}")
            raise ConnectionError(f"Transcription request timed out after {self.timeout}s")

        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI API: {e}", exc_info=True)
            raise ConnectionError(f"Transcription service error: {str(e)}")

    def _get_mime_type(self, filename: str) -> str:
        """
        Get MIME type from filename extension.

        Args:
            filename: Filename with extension

        Returns:
            MIME type string
        """
        extension = filename.lower().split('.')[-1] if '.' in filename else ''

        mime_types = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'mp4': 'audio/mp4',
            'mpeg': 'audio/mpeg',
            'mpga': 'audio/mpeg',
            'm4a': 'audio/mp4',
            'ogg': 'audio/ogg',
            'webm': 'audio/webm',
            'flac': 'audio/flac',
        }

        return mime_types.get(extension, 'audio/wav')  # Default to wav

    def _pcm_to_wav(self, pcm_data: bytes, sample_rate: int, channels: int) -> bytes:
        """
        Convert raw PCM audio to WAV format.

        Args:
            pcm_data: Raw PCM audio bytes (16-bit)
            sample_rate: Sample rate in Hz
            channels: Number of audio channels

        Returns:
            WAV file bytes
        """
        import struct

        # WAV file header
        byte_rate = sample_rate * channels * 2  # 16-bit = 2 bytes per sample
        block_align = channels * 2
        data_size = len(pcm_data)

        # Build WAV header (44 bytes)
        wav_header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            data_size + 36,  # File size - 8
            b'WAVE',
            b'fmt ',
            16,  # PCM format chunk size
            1,   # PCM format code
            channels,
            sample_rate,
            byte_rate,
            block_align,
            16,  # Bits per sample
            b'data',
            data_size
        )

        return wav_header + pcm_data

    async def close(self) -> None:
        """
        Close HTTP client and cleanup resources.
        """
        try:
            await self.client.aclose()
            logger.info("OpenAI Transcribe client closed")
        except Exception as e:
            logger.error(f"Error closing OpenAI client: {e}")
