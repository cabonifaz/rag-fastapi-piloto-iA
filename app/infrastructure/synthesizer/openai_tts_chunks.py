"""
OpenAI GPT-4o-mini-tts provider for chunk-based Text-to-Speech.

- Uses OpenAI Speech API with gpt-4o-mini-tts
- STREAMING implementation (low-latency)
- Audio is generated and yielded in chunks
- Optimized for realtime playback using PCM or WAV

For fastest response times:
- response_format = "pcm" (lowest latency)
- or "wav" (slightly more latency, better compatibility)
"""

import logging
import asyncio
from typing import Optional, AsyncGenerator

from openai import AsyncOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class OpenAITTSChunks:
    """
    OpenAI TTS provider for realtime / chunk-based audio generation.

    Designed for:
    - RAG + LLM streaming responses
    - Reading text as it is generated
    - Minimal latency audio playback
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini-tts",
        voice: str = "aloy",
        response_format: str = "pcm",  # pcm = fastest, wav = compatible
    ):
        self.api_key = api_key or getattr(settings, "openai_api_key", None)
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")

        self.model = model
        self.voice = voice
        self.response_format = response_format

        self.client = AsyncOpenAI(api_key=self.api_key)

        logger.info(
            f"OpenAI TTS initialized | model={model} voice={voice} format={response_format}"
        )

    async def synthesize_chunks(
        self,
        text: str,
        instructions: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate speech audio in streaming chunks.

        Args:
            text: Text to convert to speech
            instructions: Optional voice/style instructions

        Yields:
            Audio chunks (raw PCM or WAV bytes depending on response_format)
        """

        logger.debug(f"Generating TTS audio | chars={len(text)}")

        async with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
            instructions=instructions,
            response_format=self.response_format,  # pcm or wav
        ) as response:

            async for chunk in response.iter_bytes():
                if chunk:
                    yield chunk

        logger.debug("TTS streaming completed")

    async def close(self) -> None:
        """
        Cleanup resources.
        """
        try:
            await self.client.close()
            logger.info("OpenAI TTS client closed")
        except Exception as e:
            logger.error(f"Error closing OpenAI TTS client: {e}")
