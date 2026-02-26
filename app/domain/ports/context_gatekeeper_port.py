"""
Port (interface) for context gatekeeper services.
Defines the contract for classifying whether a query requires prior context.
"""

from abc import ABC, abstractmethod
from typing import Dict


class ContextGatekeeperPort(ABC):
    """
    Abstract interface for query context classification.

    Implementations analyze the original user query to determine:
    1. Whether the query requires prior conversation context to be searched effectively.
    2. Whether the user is asking for a recap of the previous conversation.
    """

    @abstractmethod
    async def comparate_query(
        self,
        original_query: str,
    ) -> Dict:
        """
        Classify the original query.

        Args:
            original_query: The user's original query.

        Returns:
            Dictionary with:
                - needs_context: bool — True if the query requires prior context.
                - is_summary: bool — True if the user is asking for a conversation recap.
            Returns None if the classification fails.
        """
        pass
