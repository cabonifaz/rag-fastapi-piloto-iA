"""
Conversation-related nodes for workflows.
Handles conversation history, recontextualization, and comparison.
"""
import logging
from app.workflows.states import RAGState

logger = logging.getLogger(__name__)


def create_get_conversation_history_node(message_service, max_ctx: int = 16):
    """Factory function to create get_conversation_history node

    Args:
        message_service: Message service for DynamoDB
        max_ctx: Maximum context window (default 16 for RAG, use 8 for LLM-only)
    """
    async def get_conversation_history(state) -> dict:
        """Fetch conversation history from DynamoDB"""
        conversation_history = []
        chat_id = state.get("chat_id")

        if chat_id is not None:
            try:
                messages = await message_service.get_last_n_messages(
                    chat_id=f"chat-{chat_id}",
                    n=20
                )

                if messages:
                    # Build and collapse in one pass
                    collapsed = []
                    for msg in messages:
                        role = "user" if msg.sender == 0 else "assistant"
                        if not collapsed or collapsed[-1]["role"] != role:
                            collapsed.append({"role": role, "content": msg.message})
                        else:
                            collapsed[-1]["content"] = msg.message

                    # Find boundaries
                    start = next((i for i, m in enumerate(collapsed) if m["role"] == "user"), -1)
                    end = next((i for i in range(len(collapsed)-1, -1, -1)
                               if collapsed[i]["role"] == "assistant"), -1)

                    # Validate and slice
                    if start >= 0 and end > start:
                        segment = collapsed[start:end+1]

                        # Quick alternating check
                        if all(segment[i]["role"] != segment[i+1]["role"]
                               for i in range(len(segment)-1)):

                            # Trim to context window (keep newest)
                            if len(segment) > max_ctx:
                                trim = len(segment) - max_ctx
                                if segment[trim]["role"] == "assistant":
                                    trim -= 1
                                segment = segment[max(0, trim):]

                            # Final check
                            if segment and segment[0]["role"] == "user":
                                conversation_history = segment

            except Exception as e:
                logger.warning(f"Failed to retrieve chat history: {e}")

        state["conversation_history"] = conversation_history
        return state

    return get_conversation_history


def create_context_gatekeeper_node(context_gatekeeper):
    """Factory function to create context_gatekeeper node"""
    async def context_gatekeeper_node(state: RAGState) -> RAGState:
        """Classify the original query to determine if it needs prior context"""
        gatekeeper_result = None

        try:
            gatekeeper_result = await context_gatekeeper.gatekeep_query(
                original_query=state["cleaned_message"]
            )
            logger.info(
                f"Gatekeeper result: needs_context={gatekeeper_result.get('needs_context')}, "
                f"is_summary={gatekeeper_result.get('is_summary')}"
            )
        except Exception as e:
            logger.warning(f"Failed to classify query in context gatekeeper: {e}")

        state["gatekeeper_result"] = gatekeeper_result
        return state

    return context_gatekeeper_node


def create_recontextualize_query_node(recontextualizer):
    """Factory function to create recontextualize_query node"""
    async def recontextualize_query(state: RAGState) -> RAGState:
        """Recontextualize the user query using conversation history"""
        recontextualized_query = None
        chat_id = state.get("chat_id")
        conversation_history = state.get("conversation_history", [])

        if chat_id is not None and len(conversation_history) >= 2:
            try:
                # Send last 6 messages (user + assistant) for full context
                # Trim assistant messages to 250 characters to reduce token usage
                last_messages = [
                    {**m, "content": m["content"][:250] + "..."} if m["role"] == "assistant" else m
                    for m in conversation_history[-6:]
                ]

                result = await recontextualizer.recontextualize_query(
                    user_query=state["cleaned_message"],
                    conversation_history=last_messages
                )

                if result and isinstance(result, str):
                    recontextualized_query = result
                    logger.info(
                        f"Query recontextualized:\n"
                        f"  Original:  {state['cleaned_message']}\n"
                        f"  Rewritten: {recontextualized_query}"
                    )
            except Exception as e:
                logger.warning(f"Failed to recontextualize query: {e}")

        # Fall back to the cleaned message if recontextualization didn't produce a result
        state["recontextualized_query"] = recontextualized_query or state["cleaned_message"]
        return state

    return recontextualize_query


def create_compare_query_node(comparator):
    """Factory function to create compare_query node"""
    async def compare_query(state: RAGState) -> RAGState:
        """Compare original and recontextualized queries"""
        comparator_result = None
        recontextualized_query = state.get("recontextualized_query")

        try:
            comparator_result = await comparator.comparate_query(
                original_query=state["cleaned_message"],
                recontextualized_query=recontextualized_query
            )
            logger.info(
                f"Comparator result: same_info={comparator_result.get('same_info')}, "
                f"asks_for_summary={comparator_result.get('asks_for_summary')}"
            )
        except Exception as e:
            logger.warning(f"Failed to compare queries: {e}")

        state["comparator_result"] = comparator_result
        return state

    return compare_query
