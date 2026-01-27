"""WebSocket utility functions for handling connections and streaming."""

from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import logging
import time
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

# Maximum time to stay muted before auto-closing (60 seconds)
MAX_MUTE_DURATION_SECONDS = 60

# Silence audio chunk - 16-bit PCM silence (zeros) at 16kHz
# Small chunk to keep AWS stream alive without significant cost
# 160 samples = 10ms of audio at 16kHz, 320 bytes (16-bit = 2 bytes per sample)
SILENCE_CHUNK = bytes(320)


async def audio_stream_from_websocket(websocket: WebSocket) -> AsyncGenerator[bytes, None]:
    """
    Create async generator from WebSocket audio messages.

    Esta función convierte mensajes binarios de WebSocket en un generador
    asíncrono que puede ser consumido por servicios de transcripción.

    Procesa:
    - Mensajes binarios: Yield audio chunks directamente a AWS Transcribe
    - Mensajes de texto con "stop": Termina el stream y cierra la conexión
    - Mensajes de texto con "mute": Pausa el envío de audio a AWS (no se cobra)
    - Mensajes de texto con "unmute": Reanuda el envío de audio a AWS

    Args:
        websocket: WebSocket connection

    Returns:
        AsyncGenerator yielding audio bytes

    Yields:
        bytes: Audio chunks received from WebSocket
    """
    audio_chunk_count = 0
    is_muted = False
    mute_start_time = None
    muted_chunks_skipped = 0

    logger.info("Audio stream generator started, waiting for audio chunks from WebSocket")

    try:
        # Use low-level receive() to handle both binary and text frames
        # This is the correct way to handle mixed message types
        while True:
            try:
                # Check mute timeout
                if is_muted and mute_start_time:
                    mute_duration = time.time() - mute_start_time
                    if mute_duration >= MAX_MUTE_DURATION_SECONDS:
                        logger.warning(f"Mute duration exceeded {MAX_MUTE_DURATION_SECONDS}s, auto-stopping stream")
                        # Send timeout notification to client
                        try:
                            await websocket.send_json({
                                "type": "status",
                                "status": "mute_timeout",
                                "message": f"Muted for too long ({MAX_MUTE_DURATION_SECONDS}s), stopping stream"
                            })
                        except:
                            pass
                        break

                # Use timeout to periodically check mute status
                try:
                    msg = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                except asyncio.TimeoutError:
                    # No message received - if muted, yield silence to keep AWS stream alive
                    if is_muted:
                        yield SILENCE_CHUNK
                    continue

                # Handle binary audio chunks
                if "bytes" in msg:
                    data = msg["bytes"]
                    if data and len(data) > 0:
                        if is_muted:
                            # While muted, send silence to keep AWS stream alive
                            # This prevents AWS timeout while minimizing charges
                            muted_chunks_skipped += 1
                            if muted_chunks_skipped % 50 == 0:
                                logger.debug(f"Muted: replaced {muted_chunks_skipped} audio chunks with silence")
                            # Yield silence to keep the stream alive
                            yield SILENCE_CHUNK
                        else:
                            audio_chunk_count += 1
                            logger.debug(f"Received audio chunk {audio_chunk_count}: {len(data)} bytes")
                            yield data

                # Handle text messages (control signals)
                elif "text" in msg:
                    text_data = msg["text"]
                    logger.debug(f"Received text message: {text_data}")

                    # Try to parse as JSON first
                    try:
                        json_msg = json.loads(text_data)
                        msg_type = json_msg.get("type", "").lower()
                    except json.JSONDecodeError:
                        # Not JSON, check raw text
                        msg_type = text_data.lower().strip()

                    # Handle stop signal first (highest priority)
                    if msg_type == "stop":
                        logger.info(f"Stop signal received after {audio_chunk_count} audio chunks, stopping audio capture")
                        logger.info(f"AWS Transcribe will now process remaining audio chunks and generate session summary")
                        break

                    # Handle unmute signal (check before mute since "mute" is substring of "unmute")
                    elif msg_type == "unmute":
                        if is_muted:
                            mute_duration = time.time() - mute_start_time if mute_start_time else 0
                            is_muted = False
                            mute_start_time = None
                            logger.info(f"Unmute signal received - resuming audio to AWS (was muted for {mute_duration:.1f}s, replaced {muted_chunks_skipped} chunks with silence)")
                            # Notify client that unmute is active
                            try:
                                await websocket.send_json({
                                    "type": "status",
                                    "status": "unmuted",
                                    "message": "Audio resumed"
                                })
                            except:
                                pass

                    # Handle mute signal
                    elif msg_type == "mute":
                        if not is_muted:
                            is_muted = True
                            mute_start_time = time.time()
                            muted_chunks_skipped = 0
                            logger.info(f"Mute signal received after {audio_chunk_count} chunks - replacing audio with silence")
                            # Notify client that mute is active
                            try:
                                await websocket.send_json({
                                    "type": "status",
                                    "status": "muted",
                                    "message": "Audio muted - sending silence to keep stream alive"
                                })
                            except:
                                pass

                    # Fallback for raw text messages (legacy support)
                    elif "stop" in text_data.lower():
                        logger.info(f"Stop signal received after {audio_chunk_count} audio chunks, stopping audio capture")
                        logger.info(f"AWS Transcribe will now process remaining audio chunks and generate session summary")
                        break

            except Exception as e:
                logger.error(f"Error receiving message: {type(e).__name__}: {e}")
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected during audio receive after {audio_chunk_count} chunks")
    except Exception as e:
        logger.error(f"Error receiving audio from WebSocket: {type(e).__name__}: {e}", exc_info=True)
