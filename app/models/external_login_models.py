"""External login request and response models."""

from pydantic import BaseModel
from typing import Dict, Any


class ExternalLoginRequest(BaseModel):
    """External login request model."""
    username: str
    password: str


class ExternalLoginResponse(BaseModel):
    """External login response model."""
    token: str
    result: Dict[str, Any]
