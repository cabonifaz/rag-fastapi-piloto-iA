"""Chat request models for RAG endpoints."""

from pydantic import BaseModel
from typing import Optional


class UnifiedRequest(BaseModel):
    """Request Schema for endpoints that need company-specific search."""
    user_id: str
    message: str
    company_id: str                         # Required, for company-specific search
    area: str                               # Required, for area-specific filtering and user role validation
    id_ia_area: int                         # Required, ID from actual_company_area
    top_k: Optional[int] = None             # Optional, defaults to env config
    similarity_threshold: Optional[float] = None  # Optional, defaults to env config
    alpha: Optional[float] = None           # Optional, hybrid search alpha (0.0=keyword, 1.0=vector), defaults to env config
    temperature: Optional[float] = None      # Optional, defaults to env config
    max_tokens: Optional[int] = None         # Optional, defaults to env config


class AgentStreamingRequest(BaseModel):
    """Request Schema for agent streaming endpoint with external token."""
    user_id: str
    message: str
    company_id: str                         # Required, for company-specific search
    area: str                               # Required, for area-specific filtering and user role validation
    id_ia_area: int                         # Required, ID from actual_company_area
    top_k: Optional[int] = None             # Optional, defaults to env config
    similarity_threshold: Optional[float] = None  # Optional, defaults to env config
    alpha: Optional[float] = None           # Optional, hybrid search alpha (0.0=keyword, 1.0=vector), defaults to env config
    temperature: Optional[float] = None      # Optional, defaults to env config
    max_tokens: Optional[int] = None         # Optional, defaults to env config
    external_token: str                      # Required, external system authentication token
