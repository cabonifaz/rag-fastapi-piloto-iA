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

import asyncio
import base64
import re
import time
import logging
from typing import Dict, Any, AsyncGenerator, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class TTSProviderProtocol(Protocol):
    """Protocol for TTS providers - allows dependency injection."""

    async def synthesize_chunks(self, text: str) -> AsyncGenerator[bytes, None]:
        """Synthesize text to audio chunks."""
        ...

    async def close(self) -> None:
        """Close any open connections."""
        ...


# Configuration constants
TTS_MIN_BUFFER_SIZE = 150  # Minimum chars before considering split
TTS_MAX_BUFFER_SIZE = 800  # Force split if buffer gets too large
TTS_TIMEOUT = 2.5  # Seconds before forcing a split


def clean_text_for_tts(text: str) -> str:
    """
    Clean text to make it TTS-friendly.
    Removes markdown, converts symbols to speakable text (Spanish).

    Args:
        text: Raw text potentially containing markdown and special characters

    Returns:
        Cleaned text suitable for TTS synthesis
    """
    # Remove markdown table separators (|---|---|)
    text = re.sub(r'\|[-:]+\|[-:|\s]+\|?', ' ', text)
    text = re.sub(r'\|[-:]+\|', ' ', text)

    # Replace table cell separators with pauses
    text = re.sub(r'\s*\|\s*', '. ', text)

    # Remove markdown bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold** → bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic* → italic
    text = re.sub(r'__([^_]+)__', r'\1', text)      # __bold__ → bold
    text = re.sub(r'_([^_]+)_', r'\1', text)        # _italic_ → italic

    # Convert mathematical/comparison symbols to Spanish words
    text = text.replace(' > ', ' mayor que ')
    text = text.replace(' < ', ' menor que ')
    text = text.replace(' ≥ ', ' mayor o igual que ')
    text = text.replace(' ≤ ', ' menor o igual que ')
    text = text.replace(' = ', ' igual a ')
    text = text.replace(' – ', ' a ')  # en-dash as range
    text = text.replace('–', ' a ')

    # Handle standalone comparison at start
    text = re.sub(r'^>\s*', 'mayor que ', text)
    text = re.sub(r'^<\s*', 'menor que ', text)

    # Convert common units
    text = text.replace('m/s', 'metros por segundo')
    text = text.replace('kPa', 'kilopascales')

    # Convert subscript/superscript Unicode to readable form
    text = text.replace('₀', '0')
    text = text.replace('₁', '1')
    text = text.replace('₂', '2')
    text = text.replace('₃', '3')
    text = text.replace('₄', '4')
    text = text.replace('₆₀', '60')
    text = text.replace('ₛ', 's')
    text = text.replace('ᵤ', 'u')
    text = text.replace('ᵢ', 'i')

    # Handle overline characters (mathematical notation)
    text = text.replace('𝑉̅', 'V promedio')
    text = text.replace('𝑁̅', 'N promedio')
    text = text.replace('𝑆̅', 'S promedio')
    text = text.replace('̅', '')  # Remove any remaining combining overlines

    # Clean up multiple spaces and newlines
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text)

    # Clean up multiple periods
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\.\s*\.', '.', text)

    # Remove empty parentheses or brackets
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)

    return text.strip()


def is_sentence_end(text: str) -> bool:
    """
    Check if text ends at a natural sentence boundary.
    Uses cleaned text for better detection.
    Avoids splitting after decimals (3.14), codes (E.030), abbreviations (Dr.), etc.

    Args:
        text: Text buffer to check

    Returns:
        True if text ends at a good breaking point
    """
    # Clean the text first to remove markdown artifacts
    cleaned = clean_text_for_tts(text)
    cleaned = cleaned.rstrip()

    if not cleaned:
        return False

    last_char = cleaned[-1]

    # These always indicate sentence end
    if last_char in "!?;":
        return True

    # Colon: good break point for headers/lists
    if last_char == ":":
        return True

    # Period: need to be careful
    if last_char == ".":
        if len(cleaned) >= 2:
            before_dot = cleaned[-2]
            # Decimal number: "3." might not be sentence end
            if before_dot.isdigit():
                return False
            # Single letter abbreviation: "E." "a." etc.
            if before_dot.isalpha() and (len(cleaned) < 3 or not cleaned[-3].isalpha()):
                return False
            # Check for common abbreviations
            lower_text = cleaned.lower()
            abbrevs = [
                'dr.', 'sr.', 'sra.', 'mr.', 'mrs.', 'ms.', 'vs.',
                'etc.', 'e.g.', 'i.e.', 'no.', 'vol.', 'inc.', 'ltd.', 'corp.'
            ]
            for abbrev in abbrevs:
                if lower_text.endswith(abbrev):
                    return False
        return True

    # Table row ends as good break points
    if text.rstrip().endswith('\n') or text.rstrip().endswith('|'):
        return True

    return False


class TTSStreamProcessor:
    """
    Processes text chunks and generates TTS audio in parallel with ordered delivery.

    Usage:
        # Option 1: Let processor create provider (closed after each stream)
        processor = TTSStreamProcessor()
        async for event in processor.process_stream(text_stream):
            yield event

        # Option 2: Inject shared provider (NOT closed after stream - you manage lifecycle)
        shared_provider = OpenAITTSChunks()  # Initialize at startup
        processor = TTSStreamProcessor(tts_provider=shared_provider)
        async for event in processor.process_stream(text_stream):
            yield event
    """

    def __init__(
        self,
        tts_provider: Optional[TTSProviderProtocol] = None,
        min_buffer_size: int = TTS_MIN_BUFFER_SIZE,
        max_buffer_size: int = TTS_MAX_BUFFER_SIZE,
        timeout: float = TTS_TIMEOUT,
        debug: bool = True
    ):
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        self.timeout = timeout
        self.debug = debug

        # TTS provider - can be injected or created per-request
        self._injected_provider = tts_provider
        self._owns_provider = tts_provider is None  # Only close if we created it
        self.tts_provider: Optional[TTSProviderProtocol] = None

        # State (reset for each stream)
        self.buffer = ""
        self.last_tts_time = 0.0
        self.total_llm_text = ""

        # Ordered delivery system
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.tts_tasks: List[asyncio.Task] = []
        self.audio_buffer: Dict[int, list] = {}
        self.completed_sequences: set = set()
        self.next_sequence_to_yield = 0
        self.total_sequences = 0

        # Debug tracking
        self.debug_log: List[Dict] = []

    async def _tts_worker(self, original_text: str, sequence: int):
        """Generate TTS audio in background, push chunks with sequence number."""
        try:
            clean_text = clean_text_for_tts(original_text)
            logger.info(f"[TTS] Sequence {sequence} cleaned text: '{clean_text[:100]}...'")

            if not clean_text.strip():
                logger.warning(f"[TTS] Sequence {sequence} has empty text after cleaning, skipping")
                await self.audio_queue.put((sequence, []))
                return

            chunks = []
            async for audio_chunk in self.tts_provider.synthesize_chunks(clean_text):
                chunks.append({
                    "type": "audio_chunk",
                    "content": base64.b64encode(audio_chunk).decode("utf-8")
                })

            await self.audio_queue.put((sequence, chunks))
            logger.info(f"[TTS] Sequence {sequence} completed: {len(chunks)} audio chunks")
        except Exception as e:
            logger.error(f"[TTS] Worker error for sequence {sequence}: {e}")
            await self.audio_queue.put((sequence, []))

    def _collect_ready_audio(self):
        """Collect completed sequences from queue (non-blocking)."""
        while not self.audio_queue.empty():
            try:
                seq, chunks = self.audio_queue.get_nowait()
                self.audio_buffer[seq] = chunks
                self.completed_sequences.add(seq)
            except asyncio.QueueEmpty:
                break

    def _get_ordered_audio(self):
        """Yield audio chunks in correct order."""
        while self.next_sequence_to_yield in self.completed_sequences:
            chunks = self.audio_buffer.pop(self.next_sequence_to_yield, [])
            logger.info(f"[TTS] Yielding sequence {self.next_sequence_to_yield}: {len(chunks)} audio chunks")
            for audio_event in chunks:
                yield audio_event
            self.completed_sequences.discard(self.next_sequence_to_yield)
            self.next_sequence_to_yield += 1

    def _should_generate_tts(self, current_time: float) -> tuple[bool, str]:
        """
        Check if we should trigger TTS generation.

        Returns:
            Tuple of (should_generate, reason)
        """
        buffer_len = len(self.buffer)

        if buffer_len >= self.max_buffer_size:
            return True, "max_buffer"

        if buffer_len >= self.min_buffer_size and is_sentence_end(self.buffer):
            return True, "sentence_end"

        if buffer_len >= self.min_buffer_size and current_time - self.last_tts_time >= self.timeout:
            return True, "timeout"

        return False, ""

    def _queue_tts_task(self, text: str, reason: str):
        """Queue a TTS generation task."""
        logger.info(f"[TTS] Queuing sequence {self.total_sequences} ({reason}): '{text[:80]}...' ({len(text)} chars)")

        self.debug_log.append({
            "sequence": self.total_sequences,
            "text": text,
            "char_count": len(text),
            "split_reason": reason
        })

        task = asyncio.create_task(self._tts_worker(text, self.total_sequences))
        self.tts_tasks.append(task)
        self.total_sequences += 1

    def _generate_debug_event(self) -> Dict[str, Any]:
        """Generate debug summary event."""
        full_text_combined = "".join(entry["text"] for entry in self.debug_log)

        return {
            "type": "tts_debug",
            "total_sequences_queued": self.total_sequences,
            "sequences_yielded": self.next_sequence_to_yield,
            "remaining_in_buffer": len(self.audio_buffer),
            "pending_sequences": list(self.completed_sequences),
            "llm_text_length": len(self.total_llm_text),
            "tts_text_length": len(full_text_combined),
            "text_match": len(self.total_llm_text) == len(full_text_combined),
            "missing_chars": len(self.total_llm_text) - len(full_text_combined),
            "llm_full_text": self.total_llm_text,
            "tts_combined_text": full_text_combined,
            "segments": [
                {
                    "sequence": entry["sequence"],
                    "char_count": entry["char_count"],
                    "text": entry["text"],
                    "split_reason": entry.get("split_reason", "final_flush")
                }
                for entry in self.debug_log
            ]
        }

    def _log_summary(self):
        """Log final TTS summary."""
        logger.info("[TTS] === SUMMARY ===")
        logger.info(f"[TTS] Total sequences queued: {self.total_sequences}")
        logger.info(f"[TTS] Sequences yielded: {self.next_sequence_to_yield}")
        logger.info(f"[TTS] Remaining in buffer: {len(self.audio_buffer)} sequences")
        logger.info(f"[TTS] Pending sequences: {self.completed_sequences}")

        full_text_combined = ""
        for entry in self.debug_log:
            logger.info(f"[TTS] Seq {entry['sequence']}: {entry['char_count']} chars")
            logger.info(f"[TTS] Seq {entry['sequence']} FULL TEXT START >>>")
            logger.info(f"{entry['text']}")
            logger.info(f"[TTS] Seq {entry['sequence']} FULL TEXT END <<<")
            full_text_combined += entry['text']

        logger.info(f"[TTS] === COMBINED TEXT LENGTH: {len(full_text_combined)} chars ===")
        logger.info(f"[TTS] === TOTAL LLM TEXT LENGTH: {len(self.total_llm_text)} chars ===")

        if len(self.total_llm_text) != len(full_text_combined):
            logger.warning(
                f"[TTS] ⚠️ TEXT MISMATCH! LLM: {len(self.total_llm_text)} chars, "
                f"TTS: {len(full_text_combined)} chars, "
                f"DIFF: {len(self.total_llm_text) - len(full_text_combined)} chars"
            )
            logger.warning(f"[TTS] LLM TEXT START >>>\n{self.total_llm_text}\n<<< LLM TEXT END")
            logger.warning(f"[TTS] TTS TEXT START >>>\n{full_text_combined}\n<<< TTS TEXT END")
        else:
            logger.info("[TTS] ✅ Text lengths match perfectly")

    async def process_stream(
        self,
        text_stream: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a stream of text chunks and generate TTS audio in parallel.

        Args:
            text_stream: Async generator yielding {"type": "chunk", "content": "..."} events

        Yields:
            Text chunks (passed through) and audio chunks ({"type": "audio_chunk", "content": "base64..."})
        """
        # Use injected provider or create new one
        if self._injected_provider is not None:
            self.tts_provider = self._injected_provider
        else:
            # Import here to avoid circular imports and allow injection
            from app.infrastructure.synthesizer.openai_tts_chunks import OpenAITTSChunks
            self.tts_provider = OpenAITTSChunks()

        self.last_tts_time = time.time()

        try:
            async for chunk_event in text_stream:
                # Yield text chunk immediately
                yield chunk_event

                # Collect any ready audio and yield in order
                self._collect_ready_audio()
                for audio_event in self._get_ordered_audio():
                    yield audio_event

                if chunk_event.get("type") == "chunk":
                    content = chunk_event.get("content", "")
                    self.buffer += content
                    self.total_llm_text += content
                    current_time = time.time()

                    should_generate, reason = self._should_generate_tts(current_time)

                    if should_generate and self.buffer.strip():
                        self._queue_tts_task(self.buffer, reason)
                        self.buffer = ""
                        self.last_tts_time = current_time

            # Flush remaining buffer
            if self.buffer.strip():
                logger.info(f"[TTS] Queuing final sequence {self.total_sequences}: '{self.buffer[:80]}...' ({len(self.buffer)} chars)")
                self.debug_log.append({
                    "sequence": self.total_sequences,
                    "text": self.buffer,
                    "char_count": len(self.buffer),
                    "split_reason": "final_flush"
                })
                task = asyncio.create_task(self._tts_worker(self.buffer, self.total_sequences))
                self.tts_tasks.append(task)
                self.total_sequences += 1

            # Wait for all TTS tasks to complete
            if self.tts_tasks:
                await asyncio.gather(*self.tts_tasks, return_exceptions=True)

            # Yield all remaining audio in order
            self._collect_ready_audio()
            for audio_event in self._get_ordered_audio():
                yield audio_event

            # Log summary
            if self.debug:
                self._log_summary()

            # Yield debug event
            if self.debug:
                yield self._generate_debug_event()

        finally:
            # Only close provider if we created it (not if it was injected)
            if self.tts_provider and self._owns_provider:
                await self.tts_provider.close()


async def stream_with_tts(
    text_stream: AsyncGenerator[Dict[str, Any], None],
    tts_provider: Optional[TTSProviderProtocol] = None,
    min_buffer_size: int = TTS_MIN_BUFFER_SIZE,
    max_buffer_size: int = TTS_MAX_BUFFER_SIZE,
    timeout: float = TTS_TIMEOUT,
    debug: bool = True
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Convenience function to stream text with TTS generation.

    Args:
        text_stream: Async generator yielding text chunk events
        tts_provider: Optional TTS provider instance (for connection reuse).
                     If not provided, creates a new one per call.
                     If provided, caller is responsible for closing it.
        min_buffer_size: Minimum characters before considering a split
        max_buffer_size: Maximum characters before forcing a split
        timeout: Seconds before forcing a split
        debug: Whether to log debug information

    Yields:
        Text chunks and audio chunks

    Example:
        # Per-request (simple, creates new connection each time)
        async for event in stream_with_tts(text_stream):
            yield event

        # With shared provider (recommended for production)
        # Initialize once at startup:
        tts_provider = OpenAITTSChunks()

        # Use in requests:
        async for event in stream_with_tts(text_stream, tts_provider=tts_provider):
            yield event

        # Close at shutdown:
        await tts_provider.close()
    """
    processor = TTSStreamProcessor(
        tts_provider=tts_provider,
        min_buffer_size=min_buffer_size,
        max_buffer_size=max_buffer_size,
        timeout=timeout,
        debug=debug
    )

    async for event in processor.process_stream(text_stream):
        yield event
