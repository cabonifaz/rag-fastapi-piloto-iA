from typing import Any, Dict

from pydantic import BaseModel, Field


class MCPChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    company_id: int = Field(..., gt=0)
    area_id: int = Field(..., gt=0)


class MCPChatResponse(BaseModel):
    response: str
    result: Dict[str, Any]


class MCPAvailabilityRequest(BaseModel):
    company_id: int = Field(..., gt=0)
    area_id: int = Field(..., gt=0)


class MCPAvailabilityResponse(BaseModel):
    enabled: bool
