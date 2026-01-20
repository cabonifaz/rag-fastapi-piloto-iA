"""
TTS Streaming Workflow

Handles real-time text-to-speech generation with parallel processing and ordered delivery.
Cleans markdown/technical content for natural speech synthesis.

Usage:
    # Option 1: Per-request (creates new provider each time)
    async for event in stream_with_tts(text_stream):
        yield event

    # Option 2: Injected provider (recommended for production)
    tts_provider = OpenAITTSChunks()  # Initialize at startup
    async for event in stream_with_tts(text_stream, tts_provider=tts_provider):
        yield event
"""

import base64
import re
import logging
from typing import Dict, Any, AsyncGenerator, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# TTS Provider Protocol
# ─────────────────────────────────────────────────────────────

@runtime_checkable
class TTSProviderProtocol(Protocol):
    async def synthesize_chunks(self, text: str) -> AsyncGenerator[bytes, None]:
        ...

    async def close(self) -> None:
        ...


# ─────────────────────────────────────────────────────────────
# Text normalization (language-agnostic)
# ─────────────────────────────────────────────────────────────

def normalize_text_for_tts(text: str) -> str:
    """
    Normalize text for TTS without changing language or meaning.
    - Removes markdown syntax
    - Cleans tables
    - Normalizes whitespace and punctuation
    """

    # Markdown tables
    text = re.sub(r'\|[-:]+\|[-:|\s]+\|?', ' ', text)
    text = re.sub(r'\|[-:]+\|', ' ', text)
    text = re.sub(r'\s*\|\s*', '. ', text)

    # Markdown emphasis
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # Inline code (keep content)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Newlines & spaces
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text)

    # Punctuation cleanup
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\.\s*\.', '.', text)

    # Empty brackets
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\{\s*\}', '', text)

    return text.strip()


def normalize_math(text: str) -> str:
    """
    Normalize mathematical notation for TTS readability.
    Does NOT translate symbols to any spoken language.
    """

    replacements = {
        '≥': ' >= ',
        '≤': ' <= ',
        '≠': ' != ',
        '≈': ' ~ ',
        '×': ' * ',
        '÷': ' / ',
        '−': '-',
        '–': '-',
        '—': '-',
    }

    for symbol, replacement in replacements.items():
        text = text.replace(symbol, replacement)

    # Subscripts
    text = text.translate(str.maketrans({
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
    }))

    # Superscripts
    text = text.translate(str.maketrans({
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    }))

    # Remove combining accents / overlines
    text = re.sub(r'[\u0300-\u036f]', '', text)

    return text


def prepare_text_for_tts(text: str) -> str:
    text = normalize_text_for_tts(text)
    text = normalize_math(text)
    return text


# ─────────────────────────────────────────────────────────────
# Sentence boundary detection
# ─────────────────────────────────────────────────────────────

def is_sentence_end(text: str) -> bool:
    cleaned = text.rstrip()

    if not cleaned:
        return False

    last = cleaned[-1]

    if last in "!?;:":
        return True

    if last == ".":
        if len(cleaned) >= 2:
            before = cleaned[-2]

            # Decimal numbers: 3.14
            if before.isdigit():
                return False

            # Single-letter abbreviations: E.
            if before.isalpha() and (len(cleaned) < 3 or not cleaned[-3].isalpha()):
                return False

        return True

    return False


# ─────────────────────────────────────────────────────────────
# Stream processor
# ─────────────────────────────────────────────────────────────

class TTSStreamProcessor:

    def __init__(
        self,
        tts_provider: Optional[TTSProviderProtocol] = None,
        debug: bool = True
    ):
        self.debug = debug
        self._injected_provider = tts_provider
        self._owns_provider = tts_provider is None
        self.tts_provider: Optional[TTSProviderProtocol] = None
        self.buffer = ""

    async def process_stream(
        self,
        text_stream: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[Dict[str, Any], None]:

        if self._injected_provider is not None:
            self.tts_provider = self._injected_provider
        else:
            from app.infrastructure.synthesizer.openai_tts_chunks import OpenAITTSChunks
            self.tts_provider = OpenAITTSChunks()

        try:
            async for event in text_stream:
                yield event

                if event.get("type") != "chunk":
                    continue

                self.buffer += event["content"]

                prepared = prepare_text_for_tts(self.buffer)

                if is_sentence_end(prepared):
                    self.buffer = ""

                    async for audio in self.tts_provider.synthesize_chunks(prepared):
                        yield {
                            "type": "audio_chunk",
                            "content": base64.b64encode(audio).decode("utf-8")
                        }

            # Flush final buffer
            if self.buffer.strip():
                prepared = prepare_text_for_tts(self.buffer)

                async for audio in self.tts_provider.synthesize_chunks(prepared):
                    yield {
                        "type": "audio_chunk",
                        "content": base64.b64encode(audio).decode("utf-8")
                    }

        finally:
            if self.tts_provider and self._owns_provider:
                await self.tts_provider.close()


# ─────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────

async def stream_with_tts(
    text_stream: AsyncGenerator[Dict[str, Any], None],
    tts_provider: Optional[TTSProviderProtocol] = None,
    debug: bool = True
) -> AsyncGenerator[Dict[str, Any], None]:

    processor = TTSStreamProcessor(
        tts_provider=tts_provider,
        debug=debug
    )

    async for event in processor.process_stream(text_stream):
        yield event
