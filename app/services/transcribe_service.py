"""Service for managing transcription sessions."""

from typing import Optional, AsyncGenerator, Dict, Any
import logging
import io
from datetime import datetime

from app.domain.ports.transcribe_port import TranscribePort
from app.models.transcribe_models import (
    TranscribeConfig,
    TranscriptResult
)

logger = logging.getLogger(__name__)


class TranscribeService:
    """
    Service for transcription operations.
    Handles business logic for real-time audio transcription.
    """

    def __init__(self, transcribe_port: TranscribePort):
        """
        Initialize transcribe service with a transcription provider.

        Args:
            transcribe_port: Transcription provider (e.g., AWS Transcribe)
        """
        self.transcribe_port = transcribe_port
        self.session_start_time = None
        self.total_words = 0
        self.transcription_buffer = []

    async def process_audio_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        config: TranscribeConfig,
        user_id: Optional[int] = None
    ) -> AsyncGenerator[TranscriptResult, None]:
        """
        Process audio stream and yield transcription results.

        Esta función orquesta el flujo de transcripción:
        1. Valida la configuración
        2. Inicia el stream de transcripción
        3. Procesa y valida cada resultado
        4. Mantiene métricas de la sesión

        Args:
            audio_stream: Generador asíncrono de chunks de audio
            config: Configuración de transcripción
            user_id: ID del usuario (opcional, para logging)

        Yields:
            TranscriptResult: Resultados de transcripción procesados
        """
        try:
            # Validar configuración
            self._validate_config(config)

            # Registrar inicio de sesión
            self.session_start_time = datetime.now()
            logger.info(
                f"Starting transcription session: "
                f"user_id={user_id}, "
                f"language={config.language_code}, "
                f"sample_rate={config.sample_rate}"
            )

            # Iniciar stream de transcripción
            async for result in self.transcribe_port.transcribe_stream(
                audio_stream=audio_stream,
                language_code=config.language_code,
                sample_rate=config.sample_rate,
                media_encoding=config.media_encoding,
                vocabulary_name=config.vocabulary_name,
                show_speaker_label=config.show_speaker_label,
                enable_channel_identification=config.enable_channel_identification,
                number_of_channels=config.number_of_channels
            ):
                # Validar y procesar resultado
                processed_result = self._process_result(result, config)

                if processed_result:
                    # Actualizar métricas de sesión
                    if not processed_result.is_partial:
                        self._update_session_metrics(processed_result)

                    yield processed_result

        except ValueError as e:
            logger.error(f"Configuration error in transcription: {e}")
            raise

        except ConnectionError as e:
            logger.error(f"Connection error in transcription: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error in transcription stream: {e}", exc_info=True)
            raise ConnectionError(f"Transcription service error: {str(e)}")

        finally:
            # Log session summary
            if self.session_start_time:
                duration = (datetime.now() - self.session_start_time).total_seconds()
                logger.info(
                    f"Transcription session ended: "
                    f"user_id={user_id}, "
                    f"duration={duration:.2f}s, "
                    f"words={self.total_words}"
                )

    async def close_session(self) -> Dict[str, Any]:
        """
        Close transcription session and return summary.

        Returns:
            Dict con resumen de la sesión (duración, palabras, etc.)
        """
        try:
            # Cerrar el puerto de transcripción
            await self.transcribe_port.close()

            # Calcular métricas finales
            session_summary = self._get_session_summary()

            logger.info(f"Transcription session closed: {session_summary}")

            return session_summary

        except Exception as e:
            logger.error(f"Error closing transcription session: {e}")
            return {
                "error": str(e),
                "duration_seconds": 0,
                "total_words": self.total_words
            }

    def _validate_config(self, config: TranscribeConfig) -> None:
        """
        Validate transcription configuration.

        Args:
            config: Configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # Validar sample rate
        valid_sample_rates = [8000, 16000, 44100, 48000]
        if config.sample_rate not in valid_sample_rates:
            raise ValueError(
                f"Invalid sample rate: {config.sample_rate}. "
                f"Must be one of {valid_sample_rates}"
            )

        # Validar media encoding
        valid_encodings = ['pcm', 'ogg-opus', 'flac']
        if config.media_encoding not in valid_encodings:
            raise ValueError(
                f"Invalid media encoding: {config.media_encoding}. "
                f"Must be one of {valid_encodings}"
            )

        # Validar language code format
        if not config.language_code or len(config.language_code) < 5 or '-' not in config.language_code:
            raise ValueError(
                f"Invalid language code: {config.language_code}. "
                f"Must be in format 'xx-XX' (e.g., 'es-ES', 'en-US')"
            )

        # Validar channel configuration
        if config.enable_channel_identification and not config.number_of_channels:
            raise ValueError(
                "number_of_channels is required when enable_channel_identification is True"
            )

        logger.debug(f"Configuration validated: {config.model_dump()}")

    def _process_result(
        self,
        result: Dict[str, Any],
        config: TranscribeConfig
    ) -> Optional[TranscriptResult]:
        """
        Process and validate transcription result.

        Args:
            result: Raw result from transcribe port
            config: Transcription configuration

        Returns:
            TranscriptResult if valid, None if should be filtered
        """
        try:
            # Filtrar resultados vacíos
            if not result.get('transcript') or not result['transcript'].strip():
                return None

            # Filtrar resultados parciales si están deshabilitados
            if result.get('is_partial') and not config.enable_partial_results:
                return None

            # Crear TranscriptResult con validación Pydantic
            transcript_result = TranscriptResult(
                transcript=result['transcript'],
                is_partial=result.get('is_partial', False),
                start_time=result.get('start_time', 0.0),
                end_time=result.get('end_time', 0.0),
                confidence=result.get('confidence'),
                alternatives=result.get('alternatives', []),
                speaker_label=result.get('speaker_label')
            )

            return transcript_result

        except Exception as e:
            logger.error(f"Error processing transcription result: {e}")
            return None

    def _update_session_metrics(self, result: TranscriptResult) -> None:
        """
        Update session metrics with new result.

        Args:
            result: Transcription result to process
        """
        try:
            # Contar palabras (solo en resultados finales)
            if not result.is_partial and result.transcript:
                words = len(result.transcript.split())
                self.total_words += words

                # Guardar en buffer para posible persistencia
                self.transcription_buffer.append({
                    'transcript': result.transcript,
                    'start_time': result.start_time,
                    'end_time': result.end_time,
                    'confidence': result.confidence
                })

        except Exception as e:
            logger.error(f"Error updating session metrics: {e}")

    def _get_session_summary(self) -> Dict[str, Any]:
        """
        Get session summary with metrics.

        Returns:
            Dict with session metrics
        """
        try:
            duration = 0.0
            if self.session_start_time:
                duration = (datetime.now() - self.session_start_time).total_seconds()

            # Calcular confianza promedio
            confidences = [
                item['confidence']
                for item in self.transcription_buffer
                if item.get('confidence') is not None
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else None

            # Concatenar transcripción completa
            full_transcript = " ".join([
                item['transcript']
                for item in self.transcription_buffer
            ])

            return {
                'duration_seconds': round(duration, 2),
                'total_words': self.total_words,
                'average_confidence': round(avg_confidence, 3) if avg_confidence else None,
                'full_transcript': full_transcript,
                'segment_count': len(self.transcription_buffer)
            }

        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            return {
                'error': str(e),
                'duration_seconds': 0,
                'total_words': self.total_words
            }


