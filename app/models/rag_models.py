"""Chat request models for RAG endpoints."""

from pydantic import BaseModel
from typing import Optional, List


class UnifiedRequest(BaseModel):
    """Request Schema for endpoints that need company-specific search."""
    message: str
    user_id: int                            # Required, user ID for database operations
    company_id: int                         # Required, company ID for database operations
    area_id: int                            # Required, area ID for database operations
    created_at: str                         # Message date
    chat_id: Optional[int] = None           # Optional, chat id
    request_timezone: Optional[str] = None   # Optional, user's timezone for time-aware responses
    tts: bool = False                        # Optional, enable text-to-speech streaming


class N8NRequest(BaseModel):
    """Request Schema for n8n non-streaming endpoint."""
    message: str
    user_id: int                            # Required, user ID for database operations
    company_id: int                         # Required, company ID for database operations
    area_id: int                            # Required, area ID for database operations
    created_at: str                         # Required, message timestamp
    chat_id: int                            # Required, chat ID
    request_timezone: Optional[str] = None   # Optional, user's timezone for time-aware responses


class N8NAnonymousRequest(BaseModel):
    """Request Schema for n8n non-streaming anonymous chat endpoint."""
    message: str                            # Required, message to be stored
    rag_query: str                          # Required, query to be processed by RAG
    user_anonymous_id: int                  # Required, anonymous user ID for database operations
    company_id: int                         # Required, company ID for database operations
    area_id: int                            # Required, area ID for database operations
    created_at: str                         # Required, message timestamp
    chat_anonymous_id: int                  # Required, anonymous chat ID
    request_timezone: Optional[str] = None   # Optional, user's timezone for time-aware responses


class N8NLLMOnlyRequest(BaseModel):
    """Request Schema for n8n non-streaming LLM-only endpoint (no area required)."""
    message: str
    user_id: int                            # Required, user ID for database operations
    company_id: int                         # Required, company ID for database operations
    created_at: str                         # Required, message timestamp
    chat_id: int                            # Required, chat ID
    system_behavior: Optional[str] = None    # Optional, custom system behavior/role for the LLM
    custom_llm: Optional[str] = None          # Optional, custom LLM model ID to use
    request_timezone: Optional[str] = None   # Optional, user's timezone for time-aware responses


class N8NLLMOnlyAnonymousRequest(BaseModel):
    """Request Schema for n8n non-streaming LLM-only anonymous endpoint (no area required)."""
    message: str
    user_anonymous_id: int                  # Required, anonymous user ID for database operations
    company_id: int                         # Required, company ID for database operations
    created_at: str                         # Required, message timestamp
    chat_anonymous_id: int                  # Required, anonymous chat ID
    system_behavior: Optional[str] = None    # Optional, custom system behavior/role for the LLM
    custom_llm: Optional[str] = None          # Optional, custom LLM model ID to use
    request_timezone: Optional[str] = None   # Optional, user's timezone for time-aware responses


class AgentStreamingRequest(BaseModel):
    """Request Schema for agent streaming endpoint with external token."""
    message: str
    user_id: int                            # Required, user ID for database operations
    company_id: int                         # Required, company ID for database operations
    area_id: int                            # Required, area ID for database operations
    created_at: str                         # Message date
    external_token: str                     # Required, external system authentication token


class VLMRequest(BaseModel):
    """Request Schema for VLM streaming endpoint."""
    message: str
    user_id: int
    company_id: int
    area_id: int
    created_at: str
    filenames: List[str]
    chat_id: Optional[int] = None
    request_timezone: Optional[str] = None
