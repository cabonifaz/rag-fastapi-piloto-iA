from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def format_timestamp_with_timezone(timestamp: int, tz_name: str) -> tuple[str, str]:
    """
    Convert a timestamp (in milliseconds) to formatted UTC and local timezone strings.

    Args:
        timestamp: Unix timestamp in milliseconds
        tz_name: Timezone name (e.g., "America/Lima", "Europe/London")

    Returns:
        Tuple of (utc_formatted, local_formatted)

    Example:
        >>> utc_formatted, local_formatted = format_timestamp_with_timezone(1767715545781, "America/Lima")
        >>> utc_formatted
        '2026-01-06T16:25:45.781000+00:00'
        >>> local_formatted
        '2026-01-06T11:25:45.781000-05:00'
    """
    # Convert milliseconds to seconds
    timestamp_seconds = timestamp / 1000.0

    # Create UTC datetime
    utc_dt = datetime.fromtimestamp(timestamp_seconds, tz=timezone.utc)

    # Convert to local timezone
    local_tz = ZoneInfo(tz_name)
    local_dt = utc_dt.astimezone(local_tz)

    # Format output
    utc_formatted = utc_dt.isoformat()
    local_formatted = local_dt.isoformat()

    return utc_formatted, local_formatted
