"""
Model saturation tracking service for circuit breaker pattern.
Tracks which models are saturated and prevents request cascades.
"""

from datetime import datetime, timedelta
from typing import Dict, Set
import asyncio
import logging

logger = logging.getLogger(__name__)


class ModelSaturationTracker:
    """
    Tracks which models are currently saturated (rate limited/quota exceeded).

    Once a model is marked as saturated, it's skipped for all requests
    until the timeout expires, preventing cascading 429 errors.

    Example:
        tracker = ModelSaturationTracker(saturation_timeout_minutes=30)

        # Mark a model as saturated after a 429 error
        await tracker.mark_saturated("anthropic.claude-3-5-sonnet-20241022-v2:0")

        # Check if a model is currently saturated
        is_saturated = await tracker.is_saturated("anthropic.claude-3-5-sonnet-20241022-v2:0")

        # Get status of all saturated models
        status = await tracker.get_status()
    """

    def __init__(self, saturation_timeout_minutes: int = 60):
        """
        Initialize the saturation tracker.

        Args:
            saturation_timeout_minutes: How long to mark a model as saturated
                                       before trying again (default: 60 minutes / 1 hour)
                                       Must be > 0
        """
        if saturation_timeout_minutes <= 0:
            raise ValueError("saturation_timeout_minutes must be greater than 0")

        self.saturation_timeout = timedelta(minutes=saturation_timeout_minutes)
        self.timeout_minutes = saturation_timeout_minutes
        self.saturated_models: Dict[str, datetime] = {}  # model_id -> time_marked_bad
        self._lock = asyncio.Lock()

    async def mark_saturated(self, model_id: str) -> None:
        """
        Mark a model as saturated (rate limited or quota exceeded).

        Once marked, the model will be skipped by all requests until the
        timeout expires. This prevents cascading 429 errors to the same model.

        Args:
            model_id: AWS Bedrock model ID (e.g., "anthropic.claude-3-5-sonnet...")
        """
        async with self._lock:
            now = datetime.now()
            self.saturated_models[model_id] = now
            recovery_time = now + self.saturation_timeout

            logger.warning(
                f"🚫 Model {model_id} marked as SATURATED. "
                f"Will retry after {recovery_time.strftime('%H:%M:%S')} "
                f"({self.timeout_minutes} minute timeout)"
            )

    async def is_saturated(self, model_id: str) -> bool:
        """
        Check if a model is currently marked as saturated.

        If the saturation timeout has expired, the model is automatically
        unmarked and this returns False.

        Args:
            model_id: AWS Bedrock model ID

        Returns:
            True if model is currently saturated, False otherwise
        """
        async with self._lock:
            if model_id not in self.saturated_models:
                return False

            marked_time = self.saturated_models[model_id]
            elapsed = datetime.now() - marked_time

            if elapsed > self.saturation_timeout:
                # Timeout expired, unmark it and try again
                del self.saturated_models[model_id]
                logger.info(
                    f"✅ Model {model_id} saturation timeout expired. "
                    f"Attempting to use again."
                )
                return False

            remaining = self.saturation_timeout - elapsed
            logger.debug(
                f"⏳ Model {model_id} still marked saturated. "
                f"Remaining timeout: {remaining.total_seconds():.0f}s"
            )
            return True

    async def get_active_saturated_models(self) -> Set[str]:
        """
        Get set of currently saturated models (cleanup expired ones).

        Automatically removes models whose timeout has expired.

        Returns:
            Set of model IDs that are currently saturated
        """
        async with self._lock:
            now = datetime.now()
            expired = []

            for model_id, marked_time in self.saturated_models.items():
                if now - marked_time > self.saturation_timeout:
                    expired.append(model_id)

            for model_id in expired:
                del self.saturated_models[model_id]
                logger.debug(f"Expired saturation timeout for {model_id}")

            return set(self.saturated_models.keys())

    async def get_status(self) -> Dict[str, Dict]:
        """
        Get detailed status of all saturated models.

        Shows which models are saturated and how long until they recover.

        Returns:
            Dict mapping model IDs to their status:
                - marked_at: ISO timestamp when marked as saturated
                - remaining_seconds: Seconds until timeout expires
                - expires_at: ISO timestamp when saturation timeout expires

            Example:
                {
                    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
                        "marked_at": "2025-01-15T14:30:45.123456",
                        "remaining_seconds": 1234.5,
                        "expires_at": "2025-01-15T15:00:45.123456"
                    }
                }
        """
        async with self._lock:
            status = {}
            now = datetime.now()

            for model_id, marked_time in self.saturated_models.items():
                elapsed = now - marked_time
                remaining = self.saturation_timeout - elapsed

                # Only include if not expired
                if remaining.total_seconds() > 0:
                    status[model_id] = {
                        "marked_at": marked_time.isoformat(),
                        "remaining_seconds": remaining.total_seconds(),
                        "expires_at": (marked_time + self.saturation_timeout).isoformat()
                    }

            return status

    async def clear_saturated(self, model_id: str) -> bool:
        """
        Manually clear saturation status for a model.

        Useful for admin operations or testing.

        Args:
            model_id: AWS Bedrock model ID to clear

        Returns:
            True if model was saturated and cleared, False if wasn't saturated
        """
        async with self._lock:
            if model_id in self.saturated_models:
                del self.saturated_models[model_id]
                logger.info(f"Manually cleared saturation for {model_id}")
                return True
            return False

    async def clear_all_saturated(self) -> int:
        """
        Manually clear saturation status for all models.

        Useful for testing or manual recovery operations.

        Returns:
            Number of models that were cleared
        """
        async with self._lock:
            count = len(self.saturated_models)
            if count > 0:
                logger.info(f"Clearing saturation for {count} models")
                self.saturated_models.clear()
            return count

    def get_saturated_models_sync(self) -> Set[str]:
        """
        Get set of currently saturated models (synchronous version).

        WARNING: This is a non-async version for use in sync contexts.
        Does not acquire lock, so may have race conditions.

        Use async version (get_active_saturated_models) when possible.

        Returns:
            Set of model IDs that appear to be saturated
        """
        now = datetime.now()
        saturated = set()

        for model_id, marked_time in self.saturated_models.items():
            if now - marked_time <= self.saturation_timeout:
                saturated.add(model_id)

        return saturated

    async def get_recovery_time(self, model_id: str) -> datetime:
        """
        Get the time when a saturated model will recover.

        Args:
            model_id: AWS Bedrock model ID

        Returns:
            datetime when the model will recover, or None if not saturated
        """
        async with self._lock:
            if model_id not in self.saturated_models:
                return None

            marked_time = self.saturated_models[model_id]
            recovery_time = marked_time + self.saturation_timeout

            # Check if already expired
            if datetime.now() > recovery_time:
                return None

            return recovery_time
