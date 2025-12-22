"""Chat request models for RAG endpoints."""

from pydantic import BaseModel
from typing import Optional


class UnifiedRequest(BaseModel):
    """Request Schema for endpoints that need company-specific search."""
    message: str
    user_id: int                            # Required, user ID for database operations
    user: str                               # Required, username for logging/display
    company_id: int                         # Required, company ID for database operations
    area_id: int                            # Required, area ID for database operations
    created_at: str                         # Message date
    chat_id: Optional[int] = None           # Optional, chat id
    request_timezone: Optional[str] = None   # Optional, user's timezone for time-aware responses


class N8NRequest(BaseModel):
    """Request Schema for n8n non-streaming endpoint."""
    message: str
    user_id: int                            # Required, user ID for database operations
    user: str                               # Required, username for logging/display
    company_id: int                         # Required, company ID for database operations
    area_id: int                            # Required, area ID for database operations
    created_at: str                         # Required, message timestamp
    chat_id: int                            # Required, chat ID
    request_timezone: Optional[str] = None   # Optional, user's timezone for time-aware responses


class AgentStreamingRequest(BaseModel):
    """Request Schema for agent streaming endpoint with external token."""
    message: str
    user_id: int                            # Required, user ID for database operations
    user: str                               # Required, username for logging/display
    company_id: int                         # Required, company ID for database operations
    area_id: int                            # Required, area ID for database operations
    created_at: str                         # Message date
    external_token: str                     # Required, external system authentication token
