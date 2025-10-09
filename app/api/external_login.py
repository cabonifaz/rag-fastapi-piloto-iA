"""External login API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

from app.services.external_login_service import ExternalLoginService
from app.utils.jwt_auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


class ExternalLoginRequest(BaseModel):
    """External login request model."""
    username: str
    password: str


class ExternalLoginResponse(BaseModel):
    """External login response model."""
    token: str
    result: Dict[str, Any]


# @router.post("/external/login", response_model=Dict[str, Any])
# async def external_login(
#      request: ExternalLoginRequest,
#      current_user: Dict[str, Any] = Depends(get_current_user)
#  ) -> Dict[str, Any]:
#      """
#      External system login endpoint.
#
#      Calls the external login API and returns the raw response.
#      Requires authentication with the main system.
#      """
#      try:
#          # Create external login service instance
#          external_service = ExternalLoginService()
#
#          # Call external login service
#          result = await external_service.login(
#              username=request.username,
#              password=request.password
#          )
#
#          return result
#
#      except Exception as e:
#          logger.error(f"External login endpoint error: {e}")
#          raise HTTPException(
#              status_code=500,
#              detail=f"External login failed: {str(e)}"
#          )