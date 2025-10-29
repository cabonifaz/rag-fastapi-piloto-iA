"""WebSocket utility functions for handling connections and streaming."""

from fastapi import WebSocket, WebSocketDisconnect
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


async def audio_stream_from_websocket(websocket: WebSocket) -> AsyncGenerator[bytes, None]:
    """
    Create async generator from WebSocket audio messages.

    Esta función convierte mensajes binarios de WebSocket en un generador
    asíncrono que puede ser consumido por servicios de transcripción.

    Procesa:
    - Mensajes binarios: Yield audio chunks directamente a AWS Transcribe
    - Mensajes de texto con "stop": Termina el stream y cierra la conexión

    Args:
        websocket: WebSocket connection

    Returns:
        AsyncGenerator yielding audio bytes

    Yields:
        bytes: Audio chunks received from WebSocket
    """
    audio_chunk_count = 0
    print(f"DEBUG: audio_stream_from_websocket started")
    logger.info("Audio stream generator started, waiting for audio chunks from WebSocket")

    try:
        # Use low-level receive() to handle both binary and text frames
        # This is the correct way to handle mixed message types
        while True:
            try:
                msg = await websocket.receive()

                # Handle binary audio chunks
                if "bytes" in msg:
                    data = msg["bytes"]
                    if data and len(data) > 0:
                        audio_chunk_count += 1
                        logger.debug(f"Received audio chunk {audio_chunk_count}: {len(data)} bytes")
                        yield data

                # Handle text messages (control signals like stop)
                elif "text" in msg:
                    text_data = msg["text"]
                    print(f"DEBUG: Received text message: {text_data}")
                    logger.debug(f"Received text message: {text_data}")
                    # Check for stop signal
                    if "stop" in text_data.lower():
                        print(f"DEBUG: Stop signal received after {audio_chunk_count} audio chunks, ending audio stream")
                        logger.info(f"Stop signal received after {audio_chunk_count} audio chunks, stopping audio capture")
                        print(f"DEBUG: AWS will now process remaining audio and generate final summary")
                        logger.info(f"AWS Transcribe will now process remaining audio chunks and generate session summary")
                        break

            except Exception as e:
                logger.error(f"Error receiving message: {type(e).__name__}: {e}")
                break
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected during audio receive after {audio_chunk_count} chunks")
    except Exception as e:
        logger.error(f"Error receiving audio from WebSocket: {type(e).__name__}: {e}", exc_info=True)
