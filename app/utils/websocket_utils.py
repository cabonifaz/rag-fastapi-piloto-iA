"""WebSocket utility functions for handling connections and streaming."""

from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


async def audio_stream_from_websocket(websocket: WebSocket) -> AsyncGenerator[bytes, None]:
    """
    Create async generator from WebSocket audio messages.

    Esta función convierte mensajes binarios de WebSocket en un generador
    asíncrono que puede ser consumido por servicios de transcripción.

    Args:
        websocket: WebSocket connection

    Returns:
        AsyncGenerator yielding audio bytes

    Yields:
        bytes: Audio chunks received from WebSocket
    """
    queue = asyncio.Queue()

    async def receive_audio():
        """Receive audio from WebSocket and put in queue."""
        try:
            while True:
                # Receive bytes (audio chunks) from client
                data = await websocket.receive_bytes()
                await queue.put(data)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected during audio receive")
            await queue.put(None)  # Signal end of stream
        except Exception as e:
            logger.error(f"Error receiving audio from WebSocket: {e}")
            await queue.put(None)

    # Start receiving task
    asyncio.create_task(receive_audio())

    # Generator that yields from queue
    async def generator():
        while True:
            data = await queue.get()
            if data is None:
                break
            yield data

    return generator()
