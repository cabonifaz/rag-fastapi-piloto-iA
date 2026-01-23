"""
TTS Streaming Workflow

Handles real-time text-to-speech generation with parallel processing and ordered delivery.
Cleans markdown/technical content for natural speech synthesis.

Architecture:
    - LLM reading and TTS processing run in SEPARATE async tasks
    - Uses asyncio.Queue to decouple streams with different latencies
    - SSE generator only reads from output queue (never blocks on TTS)
    - Text streams fast while audio can lag without blocking LLM

Usage:
    # Option 1: Per-request (creates new provider each time)
    async for event in stream_with_tts(text_stream):
        yield event

    # Option 2: Injected provider (recommended for production)
    tts_provider = OpenAITTSChunks()  # Initialize at startup
    async for event in stream_with_tts(text_stream, tts_provider=tts_provider):
        yield event
"""

import asyncio
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

    # Markdown headings (remove # symbols)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

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
# Sentinel for queue termination
# ─────────────────────────────────────────────────────────────

class _QueueSentinel:
    """Marker to signal end of queue processing."""
    pass

_QUEUE_END = _QueueSentinel()


# ─────────────────────────────────────────────────────────────
# Decoupled TTS Stream Processor
# ─────────────────────────────────────────────────────────────

class TTSStreamProcessor:
    """
    Processes LLM text stream and generates TTS audio without blocking.

    Architecture:
        - Task A (LLM reader): Reads LLM stream, yields text immediately,
          and enqueues sentences for TTS processing.
        - Task B (TTS worker): Reads from TTS queue, generates audio,
          and enqueues audio events for output.
        - SSE generator: Only reads from output queue and yields events.

    This ensures LLM reading never waits for TTS, preventing backpressure.
    """

    def __init__(
        self,
        tts_provider: Optional[TTSProviderProtocol] = None,
        debug: bool = True
    ):
        self.debug = debug
        self._injected_provider = tts_provider
        self._owns_provider = tts_provider is None
        self.tts_provider: Optional[TTSProviderProtocol] = None

    async def process_stream(
        self,
        text_stream: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process text stream with decoupled TTS generation.

        Yields text events immediately while TTS runs in background.
        Audio events are yielded as they become available without blocking text.
        """
        # Initialize TTS provider
        if self._injected_provider is not None:
            self.tts_provider = self._injected_provider
        else:
            from app.infrastructure.synthesizer.openai_tts_chunks import OpenAITTSChunks
            self.tts_provider = OpenAITTSChunks()

        # Queues for decoupled processing
        tts_input_queue: asyncio.Queue[str | _QueueSentinel] = asyncio.Queue()
        output_queue: asyncio.Queue[Dict[str, Any] | _QueueSentinel] = asyncio.Queue()

        # Track errors from background tasks
        llm_error: Optional[Exception] = None
        tts_error: Optional[Exception] = None

        async def llm_reader_task():
            """
            Task A: Read LLM stream, enqueue text events and sentences for TTS.
            This task NEVER waits for TTS - it just reads and enqueues.
            """
            nonlocal llm_error
            buffer = ""

            try:
                async for event in text_stream:
                    # Immediately enqueue ALL events for output (text streams fast)
                    await output_queue.put(event)

                    # Only process chunk events for TTS
                    if event.get("type") != "chunk":
                        continue

                    buffer += event["content"]
                    prepared = prepare_text_for_tts(buffer)

                    # When sentence ends, enqueue for TTS (non-blocking)
                    if is_sentence_end(prepared):
                        buffer = ""
                        await tts_input_queue.put(prepared)

                # Flush remaining buffer
                if buffer.strip():
                    prepared = prepare_text_for_tts(buffer)
                    await tts_input_queue.put(prepared)

            except Exception as e:
                llm_error = e
                logger.error(f"Error in LLM reader task: {e}")
            finally:
                # Signal TTS task that no more input is coming
                await tts_input_queue.put(_QUEUE_END)

        async def tts_worker_task():
            """
            Task B: Read sentences from queue, generate TTS, enqueue audio events.
            Runs independently - LLM reader never waits for this.
            """
            nonlocal tts_error

            try:
                while True:
                    item = await tts_input_queue.get()

                    if isinstance(item, _QueueSentinel):
                        break

                    text = item
                    try:
                        async for audio in self.tts_provider.synthesize_chunks(text):
                            await output_queue.put({
                                "type": "audio_chunk",
                                "content": base64.b64encode(audio).decode("utf-8")
                            })
                    except Exception as e:
                        logger.error(f"TTS synthesis error for text '{text[:50]}...': {e}")
                        # Continue processing other sentences even if one fails

            except Exception as e:
                tts_error = e
                logger.error(f"Error in TTS worker task: {e}")
            finally:
                # Signal output that TTS is done
                await output_queue.put(_QUEUE_END)

        # Start both tasks concurrently
        llm_task = asyncio.create_task(llm_reader_task())
        tts_task = asyncio.create_task(tts_worker_task())

        try:
            # Track completion of both tasks
            llm_done = False
            tts_done = False

            while not (llm_done and tts_done):
                try:
                    # Use timeout to periodically check task status
                    item = await asyncio.wait_for(output_queue.get(), timeout=0.1)

                    if isinstance(item, _QueueSentinel):
                        # TTS task signaled completion
                        tts_done = True
                        continue

                    yield item

                except asyncio.TimeoutError:
                    # Check if LLM task finished (even if queue is empty)
                    if llm_task.done() and not llm_done:
                        llm_done = True
                    # Continue waiting for more events
                    continue

            # Drain any remaining items in output queue
            while not output_queue.empty():
                item = await output_queue.get()
                if not isinstance(item, _QueueSentinel):
                    yield item

        finally:
            # Ensure tasks are cleaned up
            if not llm_task.done():
                llm_task.cancel()
                try:
                    await llm_task
                except asyncio.CancelledError:
                    pass

            if not tts_task.done():
                tts_task.cancel()
                try:
                    await tts_task
                except asyncio.CancelledError:
                    pass

            # Close TTS provider if we own it
            if self.tts_provider and self._owns_provider:
                await self.tts_provider.close()

            # Re-raise errors if any occurred
            if llm_error:
                raise llm_error
            if tts_error:
                raise tts_error


# ─────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────

async def stream_with_tts(
    text_stream: AsyncGenerator[Dict[str, Any], None],
    tts_provider: Optional[TTSProviderProtocol] = None,
    debug: bool = True
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream text with parallel TTS audio generation.

    Architecture ensures:
        - Text events are yielded immediately (no blocking on TTS)
        - TTS runs in background task with its own queue
        - Audio events are yielded as they become available
        - LLM reading never stalls waiting for slow TTS network calls

    Args:
        text_stream: Async generator yielding text chunk events
        tts_provider: Optional TTS provider (creates new one if not provided)
        debug: Enable debug logging

    Yields:
        Events from text_stream (immediately) and audio_chunk events (as ready)
    """
    processor = TTSStreamProcessor(
        tts_provider=tts_provider,
        debug=debug
    )

    async for event in processor.process_stream(text_stream):
        yield event
