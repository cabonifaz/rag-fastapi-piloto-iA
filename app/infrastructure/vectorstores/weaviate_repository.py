# app/infrastructure/vectorstores/weaviate_repository.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TypedDict

import asyncio

import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate.classes.query import MetadataQuery
from weaviate.classes.filters import Filter as WeaviateFilter  # opcional, para filtros avanzados


# ===== Puertos (importa el real si ya existe en tu proyecto) ==================
try:
    from app.domain.ports.vector_store_port import VectorStorePort, VectorSearchResult
except Exception:
    class VectorSearchResult(TypedDict, total=False):
        id: str
        properties: Dict[str, Any]
        distance: Optional[float]

    class VectorStorePort:  # Protocolo mínimo de referencia
        async def search_by_vector(  # type: ignore[override]
            self,
            class_name: str,
            vector: Sequence[float],
            top_k: int = 5,
            return_properties: Optional[Sequence[str]] = None,
            filters: Optional[WeaviateFilter] = None,
            target_vector: Optional[str] = None,
            tenant: Optional[str] = None,
            include_distance: bool = True,
        ) -> List[VectorSearchResult]:
            ...


# ===== Repositorio Weaviate ===================================================
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
        - init_timeout_s: timeout para checks iniciales (gRPC puede requerir ajuste).
        - extra_headers: headers opcionales (p. ej. claves de terceros si el cluster las requiere).
        - skip_init_checks: salta los health checks iniciales si tu red es lenta.
        """
        additional_config = AdditionalConfig(timeout=Timeout(init=init_timeout_s))

        self._client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key),
            additional_config=additional_config if not skip_init_checks else None,
            skip_init_checks=skip_init_checks,
            headers=extra_headers,
        )

    # ---------- API pública (puerto) ----------
    async def search_by_vector(
        self,
        class_name: str,
        vector: Sequence[float],
        top_k: int = 5,
        return_properties: Optional[Sequence[str]] = None,
        filters: Optional[WeaviateFilter] = None,
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
            filters: filtro opcional (weaviate.classes.filters.Filter.*).
            target_vector: si usas 'named vectors', especifica cuál buscar.
            tenant: identificador del tenant si la colección es multi-tenant.
            include_distance: si True, pide distancia en metadatos.

        Returns:
            Lista de dicts con: id (uuid), properties (dict) y distance (float|None).
        """

        def _query_sync() -> List[VectorSearchResult]:
            collection = self._client.collections.get(class_name, tenant=tenant)

            # Metadata opcional (distancia)
            metadata = MetadataQuery(distance=True) if include_distance else None

            resp = collection.query.near_vector(
                near_vector=vector,
                limit=top_k,
                return_properties=list(return_properties) if return_properties else [],
                return_metadata=metadata,
                filters=filters,
                target_vector=target_vector,  # requerido si usas named vectors
            )

            results: List[VectorSearchResult] = []
            for obj in resp.objects or []:
                results.append(
                    {
                        "id": str(obj.uuid),
                        "properties": obj.properties or {},
                        "distance": getattr(obj.metadata, "distance", None) if include_distance else None,
                    }
                )
            return results

        return await asyncio.to_thread(_query_sync)

    async def health(self) -> bool:
        """Ping simple del cluster."""
        def _is_ready() -> bool:
            return bool(self._client.is_ready())

        return await asyncio.to_thread(_is_ready)

    def close(self) -> None:
        """Cierra conexiones gRPC/HTTP."""
        if getattr(self, "_client", None) is not None:
            self._client.close()

    # Soporte de context manager
    def __enter__(self) -> "WeaviateRepository":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # fallback por si olvidan cerrar
        try:
            self.close()
        except Exception:
            pass
