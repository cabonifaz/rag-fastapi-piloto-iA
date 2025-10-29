import logging
import os
import asyncio
import boto3
from typing import Optional, AsyncGenerator, Dict, Any
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream
from botocore.exceptions import ClientError, NoCredentialsError
from app.domain.ports.transcribe_port import TranscribePort

# Configure logging
logger = logging.getLogger(__name__)


class MyEventHandler(TranscriptResultStreamHandler):
    """
    Handler personalizado para procesar eventos de transcripción.

    Este handler captura los resultados de transcripción (parciales y finales)
    y los pone en una cola para que puedan ser consumidos por el generador asíncrono.
    """

    def __init__(self, transcript_result_stream: TranscriptResultStream):
        super().__init__(transcript_result_stream)
        self.result_queue = asyncio.Queue()
        self.closed = False

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        """
        Maneja eventos de transcripción y los encola.

        Args:
            transcript_event: Evento de transcripción de AWS
        """
        try:
            results = transcript_event.transcript.results

            for result in results:
                if not result.alternatives:
                    continue

                # Obtener la mejor alternativa (primera)
                best_alternative = result.alternatives[0]
                transcript_text = best_alternative.transcript

                if not transcript_text:
                    continue

                # Extraer información adicional
                is_partial = result.is_partial
                start_time = result.start_time if hasattr(result, 'start_time') else 0.0
                end_time = result.end_time if hasattr(result, 'end_time') else 0.0

                # Confidence solo disponible en resultados finales
                confidence = None
                if not is_partial and hasattr(best_alternative, 'items'):
                    items = best_alternative.items
                    if items:
                        confidences = [
                            float(item.confidence)
                            for item in items
                            if hasattr(item, 'confidence') and item.confidence is not None
                        ]
                        if confidences:
                            confidence = sum(confidences) / len(confidences)

                # Convert AWS Alternative objects to dictionaries for JSON serialization
                alternatives_list = []
                try:
                    for alt in result.alternatives:
                        # Convert items (AWS Item objects) to dictionaries
                        items_list = []
                        if hasattr(alt, 'items') and alt.items:
                            try:
                                for item in alt.items:
                                    item_dict = {
                                        "type": item.type if hasattr(item, 'type') else None,
                                        "value": item.value if hasattr(item, 'value') else None,
                                        "start_time": float(item.start_time) if hasattr(item, 'start_time') and item.start_time else None,
                                        "end_time": float(item.end_time) if hasattr(item, 'end_time') and item.end_time else None,
                                        "confidence": float(item.confidence) if hasattr(item, 'confidence') and item.confidence else None,
                                        "stable": item.stable if hasattr(item, 'stable') else None
                                    }
                                    items_list.append(item_dict)
                            except Exception as e:
                                logger.debug(f"Error converting items: {e}")

                        alt_dict = {
                            "transcript": alt.transcript if hasattr(alt, 'transcript') else "",
                            "confidence": float(alt.confidence) if hasattr(alt, 'confidence') and alt.confidence else None,
                            "items": items_list if items_list else None
                        }
                        alternatives_list.append(alt_dict)
                except Exception as e:
                    logger.debug(f"Error converting alternatives: {e}")
                    alternatives_list = []

                # Crear resultado procesado
                processed_result = {
                    "transcript": transcript_text,
                    "is_partial": is_partial,
                    "start_time": start_time,
                    "end_time": end_time,
                    "confidence": confidence,
                    "alternatives": alternatives_list,
                    "speaker_label": None  # TODO: Agregar soporte para speaker diarization
                }

                # Encolar resultado
                await self.result_queue.put(processed_result)

                logger.debug(f"Transcription: {transcript_text[:50]}... (partial={is_partial})")

        except Exception as e:
            logger.error(f"Error handling transcript event: {e}", exc_info=True)

    async def signal_done(self):
        """Señala que no habrá más resultados."""
        await self.result_queue.put(None)
        self.closed = True


class AWSTranscribeStreaming(TranscribePort):
    """
    AWS Transcribe Streaming provider usando amazon-transcribe library.

    🚨 IMPORTANTE: Cada instancia = UNA sesión de transcripción para UN usuario
        → Crear NUEVA instancia por cada conexión WebSocket
        → NO reutilizar instancias entre usuarios
        → Cada instancia gestiona su propio ciclo de vida de cliente AWS

    Esta clase implementa el puerto TranscribePort para proporcionar
    transcripción de voz a texto en tiempo real usando AWS Transcribe Streaming API.
    """

    def __init__(
        self,
        region: str,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        """
        Inicializa AWS Transcribe Streaming client.

        🎯 Esta instancia es para UNA sesión de usuario únicamente.

        Args:
            region: Región AWS
            profile_name: Nombre del perfil AWS (opcional)
            aws_access_key_id: AWS access key ID (opcional)
            aws_secret_access_key: AWS secret access key (opcional)
        """
        self.region = region
        self.client = None
        self.event_handler = None

        # Load AWS credentials and set them as environment variables
        # amazon-transcribe library requires credentials in environment variables
        try:
            if aws_access_key_id and aws_secret_access_key:
                # Use provided credentials
                os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
                os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
                logger.info(f"Using provided AWS credentials for region: {region}")
            elif profile_name and os.getenv('ENVIRONMENT', '').lower() != 'production':
                # Load credentials from AWS profile using boto3
                session = boto3.Session(profile_name=profile_name, region_name=region)
                credentials = session.get_credentials()

                if credentials:
                    os.environ['AWS_ACCESS_KEY_ID'] = credentials.access_key
                    os.environ['AWS_SECRET_ACCESS_KEY'] = credentials.secret_key
                    if credentials.token:
                        os.environ['AWS_SESSION_TOKEN'] = credentials.token
                    logger.info(f"Loaded credentials from AWS profile '{profile_name}' for region: {region}")
                else:
                    logger.warning(f"No credentials found in profile '{profile_name}'")
            else:
                logger.info(f"Using default AWS credential chain for region: {region}")
        except Exception as e:
            logger.error(f"Error loading AWS credentials: {e}")
            raise

        logger.info(f"AWSTranscribeStreaming instance created for region: {region}")

    async def _audio_stream_generator(
        self,
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[bytes, None]:
        """
        Pasa el audio stream directamente al cliente de transcripción.

        Args:
            audio_stream: Generador de chunks de audio en bytes

        Yields:
            Audio chunks en bytes
        """
        try:
            async for audio_chunk in audio_stream:
                if audio_chunk and len(audio_chunk) > 0:
                    yield audio_chunk
        except Exception as e:
            logger.error(f"Error in audio stream generator: {e}")
            raise

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language_code: str = "es-ES",
        sample_rate: int = 16000,
        media_encoding: str = "pcm",
        vocabulary_name: Optional[str] = None,
        show_speaker_label: bool = False,
        enable_channel_identification: bool = False,
        number_of_channels: Optional[int] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Inicia stream de transcripción en tiempo real para ESTE usuario.

        🎯 Crea una NUEVA sesión de AWS Transcribe dedicada para esta llamada.

        Args:
            audio_stream: Generador asíncrono de chunks de audio
            language_code: Código de idioma (ej: 'es-ES', 'en-US', 'pt-BR')
            sample_rate: Frecuencia de muestreo en Hz (8000, 16000, 44100, 48000)
            media_encoding: Formato de audio ('pcm', 'ogg-opus', 'flac')
            vocabulary_name: Vocabulario personalizado (opcional)
            show_speaker_label: Habilitar identificación de hablantes
            enable_channel_identification: Habilitar identificación de canales
            number_of_channels: Número de canales (1 o 2)

        Yields:
            Dict con resultados de transcripción procesados
        """
        try:
            # CRITICAL FIX: Client sends raw PCM Int16 bytes, not ogg-opus
            # If client requests ogg-opus but we're receiving raw PCM, convert to pcm
            actual_encoding = media_encoding
            if media_encoding == "ogg-opus":
                # Client requests ogg-opus but we're receiving raw PCM, so convert
                actual_encoding = "pcm"

            # CRITICAL FIX: AWS Transcribe with PCM expects specific sample rates
            # Support common rates: 8000, 16000, 44100, 48000
            actual_sample_rate = sample_rate
            if actual_encoding == "pcm" and sample_rate not in [8000, 16000, 44100, 48000]:
                # AWS PCM requires specific sample rates, convert unsupported rates
                actual_sample_rate = 16000

            logger.info(f"Starting transcription stream: language={language_code}, "
                       f"sample_rate={actual_sample_rate}, encoding={actual_encoding}")

            # Crear cliente de transcripción
            # El cliente usa automáticamente las credenciales de las variables de entorno
            self.client = TranscribeStreamingClient(region=self.region)

            # Iniciar stream de transcripción
            stream = await self.client.start_stream_transcription(
                language_code=language_code,
                media_sample_rate_hz=actual_sample_rate,
                media_encoding=actual_encoding,
            )

            # Crear event handler
            self.event_handler = MyEventHandler(stream.output_stream)

            # Tarea para manejar eventos de transcripción
            async def handle_events():
                try:
                    await self.event_handler.handle_events()
                except Exception as e:
                    error_message = str(e)
                    # Check if it's the expected AWS timeout (normal end of stream)
                    if "timed out because no new audio was received" in error_message.lower():
                        logger.info(f"AWS Transcribe timeout (normal end of stream): {e}")
                    else:
                        logger.error(f"Error handling events: {e}", exc_info=True)
                finally:
                    await self.event_handler.signal_done()

            # Tarea para enviar audio
            async def send_audio():
                try:
                    logger.info("Starting audio send task")
                    audio_chunk_count = 0

                    async for audio_chunk in audio_stream:
                        if audio_chunk and len(audio_chunk) > 0:
                            try:
                                result = await asyncio.wait_for(
                                    stream.input_stream.send_audio_event(audio_chunk=audio_chunk),
                                    timeout=5
                                )
                                audio_chunk_count += 1
                                if audio_chunk_count % 10 == 0:
                                    logger.debug(f"Sent {audio_chunk_count} audio chunks to AWS")
                            except asyncio.TimeoutError:
                                logger.error("Timeout sending audio event to AWS")
                                break
                            except Exception as e:
                                logger.error(f"Error in send_audio_event: {type(e).__name__}: {e}")
                                break

                    # Audio stream ended, signal end to AWS
                    logger.info(f"Audio stream ended after {audio_chunk_count} chunks, sending end_stream signal to AWS")
                    try:
                        await asyncio.wait_for(stream.input_stream.end_stream(), timeout=5)
                        logger.info("end_stream signal sent successfully")
                    except asyncio.TimeoutError:
                        logger.error("Timeout sending end_stream to AWS")
                except Exception as e:
                    logger.error(f"Error sending audio: {e}", exc_info=True)

            # Iniciar ambas tareas concurrentemente
            event_task = asyncio.create_task(handle_events())
            audio_task = asyncio.create_task(send_audio())

            try:
                # Consumir resultados de la cola mientras se procesan audio y eventos
                logger.info("Starting to consume transcription results from queue")
                while True:
                    result = await self.event_handler.result_queue.get()

                    if result is None:
                        # No más resultados
                        logger.info("Received end-of-stream signal from event handler")
                        break

                    yield result

            finally:
                # Esperar a que terminen ambas tareas concurrentemente
                logger.info("Waiting for audio_task and event_task to complete")
                await asyncio.gather(audio_task, event_task, return_exceptions=True)
                logger.info("Transcription stream completed")

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"AWS ClientError in transcribe_stream: {error_code} - {error_message}")

            if error_code == 'BadRequestException':
                raise ValueError(f"Invalid transcription parameters: {error_message}")
            elif error_code == 'LimitExceededException':
                raise ConnectionError(f"Transcription limit exceeded: {error_message}")
            elif error_code == 'ConflictException':
                raise ConnectionError(f"Transcription conflict: {error_message}")
            elif error_code == 'ServiceUnavailableException':
                raise ConnectionError("AWS Transcribe service temporarily unavailable")
            else:
                raise ConnectionError(f"AWS Transcribe error: {error_code} - {error_message}")

        except NoCredentialsError as e:
            logger.error(f"AWS credentials error in transcribe_stream: {e}")
            raise ConnectionError("AWS credentials not configured or invalid")

        except asyncio.TimeoutError as e:
            logger.error(f"Timeout error in transcribe_stream: {e}")
            raise ConnectionError("AWS Transcribe request timed out - network may be slow or unstable")

        except Exception as e:
            # Check if it's a timeout exception
            error_message = str(e)
            if "timed out" in error_message.lower() or "timeout" in error_message.lower():
                logger.error(f"Timeout error in transcribe_stream: {e}")
                raise ConnectionError(f"Transcription timed out: {error_message}")

            logger.error(f"Unexpected error in transcribe_stream: {e}", exc_info=True)
            raise ConnectionError(f"Transcription service error: {error_message}")

    async def close(self) -> None:
        """
        Cierra la sesión de transcripción y limpia recursos.

        Este método debe ser llamado cuando el WebSocket se cierra para
        asegurar que los recursos de AWS se liberen correctamente.
        """
        try:
            # Cerrar event handler si existe
            if self.event_handler and not self.event_handler.closed:
                await self.event_handler.signal_done()

            # Limpiar referencias
            self.client = None
            self.event_handler = None

            logger.info("Transcription session closed and resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing transcription session: {e}")
