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
    logger.info("Audio stream generator started, waiting for audio chunks from WebSocket")

    try:
        # Use low-level receive() to handle both binary and text frames
        while True:
            try:
                # Try to receive binary data
                data = await websocket.receive_bytes()
                if data and len(data) > 0:
                    audio_chunk_count += 1
                    logger.debug(f"Received audio chunk {audio_chunk_count}: {len(data)} bytes")
                    yield data
            except RuntimeError as e:
                # receive_bytes() raises RuntimeError if it's a text frame
                logger.debug(f"RuntimeError receiving bytes: {e}, trying receive_text()")
                try:
                    text_data = await websocket.receive_text()
                    logger.debug(f"Received text message: {text_data[:100]}")
                    # Check for stop signal
                    if "stop" in text_data.lower():
                        logger.info(f"Stop signal received after {audio_chunk_count} audio chunks, ending audio stream")
                        break
                except Exception as text_error:
                    logger.error(f"Error receiving text: {type(text_error).__name__}: {text_error}")
                    break
            except KeyError as ke:
                # KeyError means the message type is not what we expect
                logger.debug(f"KeyError receiving message: {ke}, getting raw message")
                try:
                    msg = await websocket.receive()
                    if "bytes" in msg:
                        data = msg["bytes"]
                        if data and len(data) > 0:
                            audio_chunk_count += 1
                            logger.debug(f"Received audio chunk {audio_chunk_count} via raw message: {len(data)} bytes")
                            yield data
                    elif "text" in msg:
                        logger.debug(f"Received text message via raw message: {msg['text'][:100]}")
                        if "stop" in msg["text"].lower():
                            logger.info(f"Stop signal received after {audio_chunk_count} audio chunks, ending audio stream")
                            break
                except Exception as raw_error:
                    logger.error(f"Error getting raw message: {type(raw_error).__name__}: {raw_error}")
                    break
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected during audio receive after {audio_chunk_count} chunks")
    except Exception as e:
        logger.error(f"Error receiving audio from WebSocket: {type(e).__name__}: {e}", exc_info=True)
