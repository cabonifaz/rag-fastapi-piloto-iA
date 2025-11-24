from pydantic import BaseModel
from typing import Optional, List


class GetKnowledgeRequest(BaseModel):
    """Request model for getting knowledge/documents by company"""
    id_empresa: int
    id_area: Optional[int] = None


class UpdateKnowledgeStateRequest(BaseModel):
    """Request model for updating knowledge process state"""
    id_carga: int
    id_estado_proceso: int


class BatchUploadKnowledgeRequest(BaseModel):
    """Request model for batch uploading knowledge/documents"""
    id_empresa: int
    id_area: int
    pdf_keys: List[str]
    id_modelo_embedding: Optional[str] = "4"
