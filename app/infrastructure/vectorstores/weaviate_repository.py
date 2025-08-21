from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TypedDict

import asyncio
import weaviate
from weaviate.auth import AuthApiKey

from app.domain.ports.vectorstore_port import VectorStorePort

class VectorSearchResult(TypedDict, total=False):
    id: str
    properties: Dict[str, Any]
    distance: Optional[float]

class WeaviateRepository(VectorStorePort):
    """
    Adaptador de salida para Weaviate (solo lectura/búsqueda).

    - No embebe textos (BYO vectors): recibes `vector` desde tu servicio de embeddings.
    - No asume nombres de colección ni de vectores: se pasan por parámetro.
    - Compatible con multitenancy (parámetro `tenant`).
    - Cliente v4 (collections + near_vector).

    Uso básico:
        repo = WeaviateRepository(
            url=WEAVIATE_URL,
            api_key=WEAVIATE_API_KEY,
            init_timeout_s=10,
        )
        results = await repo.search_by_vector(
            class_name="MyCollection",
            vector=query_vec,
            top_k=5,
            return_properties=["title", "text", "source"],
        )
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        *,
        init_timeout_s: float = 10,
        extra_headers: Optional[Dict[str, str]] = None,
        skip_init_checks: bool = False,
    ) -> None:
        """
        Parámetros:
        - url: Endpoint del cluster (Weaviate Cloud/managed).
        - api_key: API key de Weaviate (Admin/Client según tu cluster).
        - init_timeout_s: timeout para checks iniciales.
        - extra_headers: headers opcionales.
        - skip_init_checks: salta los health checks iniciales si tu red es lenta.
        """
        auth_config = AuthApiKey(api_key=api_key)
        
        # Use proper HTTPS URL format
        if not url.startswith('https://'):
            url = f"https://{url}"
            
        self._client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=auth_config,
            headers=extra_headers or {},
            skip_init_checks=skip_init_checks
        )

    async def search_by_vector(
        self,
        class_name: str,
        vector: Sequence[float],
        top_k: int = 5,
        return_properties: Optional[Sequence[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        target_vector: Optional[str] = None,
        tenant: Optional[str] = None,
        include_distance: bool = True,
    ) -> List[VectorSearchResult]:
        """
        Búsqueda por vector (similaridad) en una colección.

        Args:
            class_name: nombre de la colección en Weaviate.
            vector: embedding de consulta (mismo espacio que tus datos).
            top_k: máximo de objetos a recuperar.
            return_properties: lista de propiedades a devolver (p. ej. ["title","text","url"]).
            filters: filtro opcional.
            target_vector: si usas 'named vectors', especifica cuál buscar.
            tenant: identificador del tenant si la colección es multi-tenant.
            include_distance: si True, pide distancia en metadatos.

        Returns:
            Lista de dicts con: id (uuid), properties (dict) y distance (float|None).
        """

        def _query_sync() -> List[VectorSearchResult]:
            collection = self._client.collections.get(class_name)
            
            response = collection.query.near_vector(
                near_vector=list(vector),
                limit=top_k,
                return_metadata=["distance"] if include_distance else [],
                return_properties=list(return_properties) if return_properties else None
            )
            
            results: List[VectorSearchResult] = []
            for obj in response.objects:
                item: VectorSearchResult = {
                    "id": str(obj.uuid),
                    "properties": dict(obj.properties) if obj.properties else {},
                }
                if include_distance and obj.metadata and obj.metadata.distance is not None:
                    item["distance"] = obj.metadata.distance
                results.append(item)
            
            return results

        return await asyncio.to_thread(_query_sync)

    async def search(
        self, 
        query_vector: List[float], 
        top_k: int = 5,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors using the default class name from settings."""
        from app.core.config import settings
        
        return await self.search_in_collection(
            collection_name=settings.weaviate_class_name,
            query_vector=query_vector,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

    async def search_in_collection(
        self,
        collection_name: str,
        query_vector: List[float], 
        top_k: int = 5,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in a specific collection."""
        
        results = await self.search_by_vector(
            class_name=collection_name,
            vector=query_vector,
            top_k=top_k,
            return_properties=["text", "source"],
            include_distance=True
        )
        
        if similarity_threshold is not None:
            results = [r for r in results if r.get("distance", 1.0) <= similarity_threshold]
        
        return [
            {
                "id": r["id"],
                "content": r["properties"].get("text", ""),
                "metadata": {
                    "source": r["properties"].get("source", ""),
                    "distance": r.get("distance")
                }
            }
            for r in results
        ]

    async def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        return await self.health()

    async def health(self) -> bool:
        """Ping simple del cluster."""
        def _is_ready() -> bool:
            try:
                return self._client.is_ready()
            except Exception:
                return False

        return await asyncio.to_thread(_is_ready)

    def close(self) -> None:
        """Cierra conexiones gRPC/HTTP."""
        if getattr(self, "_client", None) is not None:
            self._client.close()

    def __enter__(self) -> "WeaviateRepository":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
