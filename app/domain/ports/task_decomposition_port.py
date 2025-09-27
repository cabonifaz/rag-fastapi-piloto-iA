from abc import ABC, abstractmethod
from typing import Dict, Any


class QueryAnalysisPort(ABC):
    """
    Port interface for query analysis services.
    Responsible for analyzing user queries and returning structured workflow requirements.
    """

    @abstractmethod
    async def analyze_query(
        self,
        user_query: str,
        available_apis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze user query and return structured workflow requirements.

        Args:
            user_query: The user's question or request
            available_apis: Available API endpoints and their schemas

        Returns:
            Structured analysis with workflow requirements

        Example:
            {
                "needs_context": False,
                "context_messages": 0,
                "needs_system_data": True,
                "system_calls": [
                    {
                        "entity": "switches",
                        "endpoint": "/network/switches",
                        "method": "GET",
                        "params": {},
                        "missing_required_params": []
                    }
                ],
                "needs_external_knowledge": False,
                "semantic_query": "",
                "format": "table",
                "query_clean": "What network switches do we have?"
            }
        """
        pass