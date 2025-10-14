"""Chat request models for RAG endpoints."""

from pydantic import BaseModel
from typing import Optional


class UnifiedRequest(BaseModel):
    """Request Schema for endpoints that need company-specific search."""
    message: str
    user_id: int                            # Required, user ID for database operations
    user: str                               # Required, username for logging/display
    company_id: int                         # Required, company ID for database operations
    company: str                            # Required, company name for vector search filtering
    area_id: int                            # Required, area ID for database operations
    area: str                               # Required, area name for vector search filtering
    id_ia_area: int                         # Required, ID from actual_company_area
    created_at: str                         # Message date
    chat_id: Optional[int] = None           # Optional, chat id
    top_k: Optional[int] = None             # Optional, defaults to env config
    similarity_threshold: Optional[float] = None  # Optional, defaults to env config
    alpha: Optional[float] = None           # Optional, hybrid search alpha (0.0=keyword, 1.0=vector), defaults to env config
    temperature: Optional[float] = None      # Optional, defaults to env config
    max_tokens: Optional[int] = None         # Optional, defaults to env config


class AgentStreamingRequest(BaseModel):
    """Request Schema for agent streaming endpoint with external token."""
    message: str
    user_id: int                            # Required, user ID for database operations
    user: str                               # Required, username for logging/display
    company_id: int                         # Required, company ID for database operations
    company: str                            # Required, company name for vector search filtering
    area_id: int                            # Required, area ID for database operations
    area: str                               # Required, area name for vector search filtering
    id_ia_area: int                         # Required, ID from actual_company_area
    created_at: str                         # Message date
    external_token: str                     # Required, external system authentication token
    top_k: Optional[int] = None             # Optional, defaults to env config
    similarity_threshold: Optional[float] = None  # Optional, defaults to env config
    alpha: Optional[float] = None           # Optional, hybrid search alpha (0.0=keyword, 1.0=vector), defaults to env config
    temperature: Optional[float] = None      # Optional, defaults to env config
    max_tokens: Optional[int] = None         # Optional, defaults to env config
