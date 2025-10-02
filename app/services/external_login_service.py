"""External login service for handling third-party API authentication."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ExternalLoginService:
    """Service for handling external system login operations."""

    async def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Login to external system using httpx client.
        Returns raw response from external API.
        """
        try:
            from app.infrastructure.api_clients.api_client import httpx_test_login

            # Call external login API and return raw response
            result = await httpx_test_login(username, password)

            if result.get("success"):
                # Return the actual login response data
                return result.get("data", {})
            else:
                # Return error info
                return {"error": result.get("error", "Login failed")}

        except Exception as e:
            logger.error(f"External login error: {e}")
            return {"error": f"Login service error: {str(e)}"}