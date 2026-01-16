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
    _semaphore = asyncio.Semaphore(3)

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini-tts",
        voice: str = "aloy",
        response_format: str = "pcm",
    ):
        self.api_key = api_key or getattr(settings, "openai_api_key", None)
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")

        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def synthesize_chunks(
        self,
        text: str,
        instructions: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:

        if not text or len(text.strip()) < 3:
            return

        instructions = instructions or (
            "Speak naturally, clearly, and at a conversational pace. "
            "Pause slightly at punctuation."
        )

        async with self._semaphore:
            async with self.client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=self.voice,
                input=text,
                instructions=instructions,
                response_format=self.response_format,
            ) as response:
                async for chunk in response.iter_bytes():
                    if chunk:
                        yield chunk

    async def close(self) -> None:
        await self.client.close()
