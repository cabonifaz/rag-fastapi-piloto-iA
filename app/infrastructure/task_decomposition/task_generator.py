"""Task generator for creating executable tasks from orchestrator analysis."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class TaskGenerator:
    """Generates executable tasks from orchestrator analysis output."""

    @staticmethod
    def generate_tasks_from_analysis(analysis: Dict[str, Any], available_apis: List[Dict[str, Any]], user_query: str) -> List[Dict[str, Any]]:
        """Generate executable tasks from orchestrator analysis."""

        tasks = []

        # Task 1: Context retrieval if needed
        if analysis.get("needs_context", False):
            context_messages = analysis.get("context_messages")
            if context_messages and context_messages > 0:
                tasks.append({
                    "action": "get_context",
                    "messages": context_messages
                })

        # Task 2: Embedding generation for external knowledge
        if analysis.get("needs_external_knowledge", False):
            semantic_query = analysis.get("semantic_query", analysis.get("query_clean", user_query))
            tasks.append({
                "action": "embedding",
                "input": semantic_query
            })

        # Task 3: Vector retrieval if external knowledge needed
        if analysis.get("needs_external_knowledge", False):
            tasks.append({
                "action": "retrieval"
            })

        # Task 4: System data calls if needed
        if analysis.get("needs_system_data", False) and available_apis:
            system_calls = analysis.get("system_calls", [])
            for call in system_calls:
                # Find API definition
                api_def = TaskGenerator._find_api_definition(call, available_apis)
                if not api_def:
                    continue

                # Extract parameter definitions
                params_def = api_def.get("params", {})
                required_params = [p for p, meta in params_def.items() if meta.get("required")]
                allowed_params = set(params_def.keys())

                # Check if actually required params are missing
                # Only ignore orchestrator missing params that actually have values
                provided_params = call.get("params", {})
                orchestrator_missing = call.get("missing_required_params", [])

                # Filter out from orchestrator's missing list any params that actually have values
                corrected_missing = [
                    p for p in orchestrator_missing
                    if p not in provided_params or provided_params[p] in [None, "", []]
                ]

                # Also check for any required params not flagged by orchestrator but actually missing
                actually_missing_required = [
                    p for p in required_params
                    if p not in provided_params or provided_params[p] in [None, "", []]
                ]

                # Combine both checks - skip if any required params are truly missing
                if corrected_missing or actually_missing_required:
                    continue

                # Sanitize parameters
                call_params = TaskGenerator._sanitize_parameters(
                    call.get("params", {}), allowed_params
                )

                # Validate parameters
                if not TaskGenerator._validate_api_parameters(call_params, params_def, required_params):
                    continue

                # Create valid API call task
                tasks.append({
                    "action": "api_call",
                    "method": call.get("method", "GET"),
                    "endpoint": call.get("endpoint", ""),
                    "params": call_params
                })

        # Si aún no hay tasks, forzar embeddings + retrieval
        if not tasks:
            semantic_query = analysis.get("semantic_query", analysis.get("query_clean", user_query))
            tasks.append({
                "action": "embedding",
                "input": semantic_query
            })
            tasks.append({
                "action": "retrieval"
            })

        # Task 5: LLM response generation
        format_type = analysis.get("format")
        if format_type:
            tasks.append({
                "action": "llm_response",
                "format": format_type
            })
        else:
            tasks.append({
                "action": "llm_response"
            })

        return tasks

    @staticmethod
    def _find_api_definition(call: Dict[str, Any], available_apis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find API definition matching the call's endpoint and method."""
        call_endpoint = call.get("endpoint", "")
        call_method = call.get("method", "")

        return next(
            (api for api in available_apis
             if (api.get("endpoint") == call_endpoint or call_endpoint in api.get("endpoint", ""))
             and api.get("method") == call_method),
            None
        )

    @staticmethod
    def _sanitize_parameters(call_params: Dict[str, Any], allowed_params: set) -> Dict[str, Any]:
        """Clean invalid or empty parameters from call params."""
        return {
            k: v for k, v in call_params.items()
            if k in allowed_params and v not in [None, "", []]
        }

    @staticmethod
    def _validate_api_parameters(call_params: Dict[str, Any], params_def: Dict[str, Dict[str, Any]],
                               required_params: List[str]) -> bool:
        """Validate API parameters against definition requirements."""
        # Check all required parameters are present
        if not all(r in call_params for r in required_params):
            return False

        # Validate parameter types
        for k, v in call_params.items():
            expected_type = params_def.get(k, {}).get("type")
            if expected_type == "integer" and not isinstance(v, int):
                return False
            if expected_type == "string" and not isinstance(v, str):
                return False

        return True