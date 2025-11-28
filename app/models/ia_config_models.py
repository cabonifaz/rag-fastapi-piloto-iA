"""Pydantic models for IA area configuration."""

from pydantic import BaseModel
from decimal import Decimal


class GetIaAreaConfigRequest(BaseModel):
    """Request model for getting IA area configuration."""
    id_empresa: int
    id_area: int


class UpdateIaAreaConfigRequest(BaseModel):
    """Request model for updating IA area configuration."""
    id_empresa: int
    id_area: int
    id_embeddings: int
    id_llm: int
    embeddings_dimensions: int
    llm_max_tokens: int
    llm_temperature: Decimal
    llm_top_p: Decimal
    rag_top_k_results: int
    rag_similarity_threshold: Decimal
    rag_alpha: Decimal
    role_behavior: str


class GetIaAreaConfigResponse(BaseModel):
    """Response model for getting IA area configuration."""
    id_ia_area: int
    id_area: int
    id_embeddings: int
    id_llm: int
    embeddings_dimensions: int
    llm_max_tokens: int
    llm_temperature: Decimal
    llm_top_p: Decimal
    rag_top_k_results: int
    rag_similarity_threshold: Decimal
    rag_alpha: Decimal
    role_behavior: str
