"""
RAG-specific nodes for workflows.
Handles query preparation, embedding generation, vector search, and context building.
"""
import logging
from app.workflows.states import RAGState
from app.utils.search_utils import build_context_from_search_results

logger = logging.getLogger(__name__)


def create_determine_query_for_search_node():
    """Factory function to create determine_query_for_search node"""
    async def determine_query_for_search(state: RAGState) -> RAGState:
        """Determine which query to use for embedding and search"""
        state["query_for_search"] = state["recontextualized_query"]
        return state

    return determine_query_for_search


def create_generate_embedding_node(embeddings_provider):
    """Factory function to create generate_embedding node"""
    async def generate_embedding(state: RAGState) -> RAGState:
        """Generate embedding for the query"""
        query_embedding = await embeddings_provider.embed(state["query_for_search"])
        state["query_embedding"] = query_embedding
        return state

    return generate_embedding


def create_search_vector_db_node(vectorstore):
    """Factory function to create search_vector_db node"""
    async def search_vector_db(state: RAGState) -> RAGState:
        """Search vector database using hybrid search"""
        rag_config = state["rag_config"]

        search_results = await vectorstore.search_in_collection_hybrid(
            company_id=state["company_id"],
            area_id=state["area_id"],
            query_text=state["query_for_search"],
            query_vector=state["query_embedding"],
            top_k=rag_config['config']['RAG_TOP_K_RESULTS'],
            similarity_threshold=rag_config['config']['RAG_SIMILARITY_THRESHOLD'],
            alpha=rag_config['config']['RAG_ALPHA'],
            general_area=rag_config.get('general_area')
        )

        state["search_results"] = search_results
        return state

    return search_vector_db


def create_build_context_node():
    """Factory function to create build_context node"""
    async def build_context(state: RAGState) -> RAGState:
        """Build context text from search results"""
        context_text = build_context_from_search_results(state["search_results"])
        state["context_text"] = context_text
        return state

    return build_context
