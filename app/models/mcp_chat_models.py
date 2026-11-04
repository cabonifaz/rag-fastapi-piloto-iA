from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class MCPHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class MCPChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    company_id: int = Field(..., gt=0)
    area_id: int = Field(..., gt=0)
    history: List[MCPHistoryMessage] = Field(default_factory=list)


class MCPChatResponse(BaseModel):
    response: str
    result: Dict[str, Any]


class MCPAvailabilityRequest(BaseModel):
    company_id: int = Field(..., gt=0)
    area_id: int = Field(..., gt=0)


class MCPAvailabilityResponse(BaseModel):
    enabled: bool
