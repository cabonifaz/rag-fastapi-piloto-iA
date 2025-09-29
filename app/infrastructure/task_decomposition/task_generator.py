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
                missing = call.get("missing_required_params", [])
                if missing:  # si faltan parámetros requeridos, se salta
                    continue

                # Buscar en available_apis la definición de ese endpoint y método
                api_def = next(
                    (api for api in available_apis if api.get("endpoint") == call.get("endpoint") and api.get("method") == call.get("method")),
                    None
                )
                if not api_def:
                    continue  # si no está definida en available_apis, la ignoramos

                # Extraer definición de parámetros
                params_def = api_def.get("params", {})
                required_params = [p for p, meta in params_def.items() if meta.get("required")]
                allowed_params = set(params_def.keys())

                # Limpiar parámetros inválidos o vacíos ("")
                call_params = {
                    k: v for k, v in call.get("params", {}).items()
                    if k in allowed_params and v != ""
                }

                # Verificar que todos los obligatorios estén presentes
                if not all(r in call_params for r in required_params):
                    continue  # falta algún parámetro obligatorio, no se agrega

                # Validar tipos de parámetros según la definición
                valid = True
                for k, v in call_params.items():
                    expected_type = params_def.get(k, {}).get("type")
                    if expected_type == "integer" and not isinstance(v, int):
                        valid = False
                        break
                    if expected_type == "string" and not isinstance(v, str):
                        valid = False
                        break
                if not valid:
                    continue  # si algún parámetro no respeta el tipo → se descarta

                # Si pasa todas las validaciones, se agrega la tarea
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
        return next(
            (api for api in available_apis
             if api.get("endpoint") == call.get("endpoint") and api.get("method") == call.get("method")),
            None
        )

    @staticmethod
    def _sanitize_parameters(call_params: Dict[str, Any], allowed_params: set) -> Dict[str, Any]:
        """Clean invalid or empty parameters from call params."""
        return {
            k: v for k, v in call_params.items()
            if k in allowed_params and v != ""
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