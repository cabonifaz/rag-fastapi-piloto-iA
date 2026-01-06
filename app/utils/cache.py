"""Simple in-memory cache utility with TTL support."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SimpleCache:
    """Simple in-memory cache with TTL (Time To Live)."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str, ttl_seconds: int = 300) -> Optional[Any]:
        """
        Get value from cache if it exists and hasn't expired.

        Args:
            key: Cache key
            ttl_seconds: Time to live in seconds (default: 300 = 5 minutes)

        Returns:
            Cached value if valid, None otherwise
        """
        if key in self._cache:
            cached_data = self._cache[key]
            expiry_time = cached_data['expiry']

            # Check if cache is still valid
            if datetime.now() < expiry_time:
                logger.info(f"Cache HIT for key: {key}")
                return cached_data['value']
            else:
                # Cache expired, remove it
                logger.info(f"Cache EXPIRED for key: {key}")
                del self._cache[key]

        logger.info(f"Cache MISS for key: {key}")
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """
        Set value in cache with expiry time.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds (default: 300 = 5 minutes)
        """
        expiry_time = datetime.now() + timedelta(seconds=ttl_seconds)
        self._cache[key] = {
            'value': value,
            'expiry': expiry_time
        }
        logger.info(f"Cache SET for key: {key}, TTL: {ttl_seconds}s, expires at: {expiry_time}")

    def clear(self, key: Optional[str] = None):
        """
        Clear cache entry or entire cache.

        Args:
            key: Specific key to clear, or None to clear all
        """
        if key:
            if key in self._cache:
                del self._cache[key]
                logger.info(f"Cache CLEARED for key: {key}")
        else:
            self._cache.clear()
            logger.info("Cache CLEARED (all entries)")

    def cleanup_expired(self):
        """Remove all expired entries from cache."""
        now = datetime.now()
        expired_keys = [
            key for key, data in self._cache.items()
            if now >= data['expiry']
        ]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.info(f"Cache CLEANUP: removed {len(expired_keys)} expired entries")
