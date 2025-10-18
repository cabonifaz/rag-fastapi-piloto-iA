"""Utility functions for query processing."""

import re


def clean_user_query(message: str) -> str:
    """
    Clean user query by removing special quotes, normalizing whitespace, and handling line breaks.

    Args:
        message: Raw user input message

    Returns:
        Cleaned query string
    """
    # First strip leading/trailing whitespace and line breaks
    user_query = message.strip()

    # Replace newlines/line breaks in the middle with spaces
    user_query = user_query.replace('\n', ' ').replace('\r', ' ')

    # Remove special quote characters from anywhere in the string
    special_quotes = ['"', '“', '”', "'"]
    for quote in special_quotes:
        user_query = user_query.replace(quote, '')

    # Normalize multiple spaces into single space and trim again
    user_query = re.sub(r'\s+', ' ', user_query).strip()

    return user_query
