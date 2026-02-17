"""
Port (interface) for query comparison services.
Defines the contract for comparing an original query against its recontextualized version.
"""

from abc import ABC, abstractmethod
from typing import Dict


class ComparatorPort(ABC):
    """
    Abstract interface for query comparison.

    Implementations compare the original user query with the recontextualized one to determine:
    1. Whether the original query alone is sufficient for an accurate vector search.
    2. Whether the user is asking for a recap of the previous conversation.
    """

    @abstractmethod
    async def comparate_query(
        self,
        original_query: str,
        recontextualized_query: str,
    ) -> Dict:
        """
        Compare the original query against its recontextualized version.

        Args:
            original_query: The user's original query.
            recontextualized_query: The recontextualized version of the query.

        Returns:
            Dictionary with:
                - same_info: bool — True if the original query is sufficient for vector search.
                - asks_for_summary: bool — True if the user is asking for a conversation recap.
            Returns None if the comparison fails.
        """
        pass
