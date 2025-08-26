from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TypedDict

import asyncio
import logging
import weaviate
from weaviate.exceptions import WeaviateBaseError

from app.domain.ports.vectorstore_port import VectorStorePort

# Configure logging
logger = logging.getLogger(__name__)

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
        # Use proper HTTPS URL format
        if not url.startswith('https://'):
            url = f"https://{url}"
            
        # Check what weaviate version is available and use appropriate connection
        try:
            # Try v4 connection (weaviate-client >= 4.0)
            if hasattr(weaviate, 'connect_to_weaviate_cloud'):
                self._client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url,
                    auth_credentials=weaviate.auth.AuthApiKey(api_key),
                    headers=extra_headers or {},
                    skip_init_checks=skip_init_checks
                )
            else:
                raise AttributeError("v4 method not found")
        except (AttributeError, ModuleNotFoundError):
            # Fallback to v3 connection (weaviate-client < 4.0)
            self._client = weaviate.Client(
                url=url,
                auth_client_secret=weaviate.AuthApiKey(api_key=api_key),
                additional_headers=extra_headers or {},
                timeout_config=(5, 15)
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
            try:
                # v4 client method
                collection = self._client.collections.get(class_name)
                
                # First try a simple get to see if collection has any data
                simple_response = collection.query.fetch_objects(limit=1)
                
                # Build query with optional filters
                query_kwargs = {
                    "near_vector": list(vector),
                    "limit": top_k,
                    "return_metadata": ["distance"] if include_distance else [],
                }
                
                # Only specify properties if they're provided
                if return_properties:
                    query_kwargs["return_properties"] = list(return_properties)
                
                # Add filters if provided
                if filters:
                    query_kwargs["where"] = filters
                
                response = collection.query.near_vector(**query_kwargs)
                
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
                
            except WeaviateBaseError as e:
                logger.error(f"Weaviate error in search_by_vector: {e}")
                error_msg = str(e).lower()
                if "unauthorized" in error_msg or "authentication" in error_msg:
                    raise ConnectionError("Weaviate authentication failed")
                elif "not found" in error_msg or "does not exist" in error_msg:
                    raise ValueError(f"Collection '{class_name}' not found in Weaviate")
                elif "timeout" in error_msg or "timed out" in error_msg:
                    raise TimeoutError("Weaviate query timeout")
                elif "connection" in error_msg or "network" in error_msg:
                    raise ConnectionError("Cannot connect to Weaviate service")
                else:
                    raise ConnectionError(f"Weaviate service error: {str(e)}")
                    
            except ConnectionError:
                raise  # Re-raise connection errors
                
            except TimeoutError:
                raise  # Re-raise timeout errors
                
            except ValueError as e:
                logger.error(f"Value error in search_by_vector: {e}")
                if "collection" not in str(e).lower():
                    raise ValueError(f"Invalid search parameters: {str(e)}")
                raise  # Re-raise collection not found errors
                
            except Exception as e:
                logger.error(f"Unexpected error in search_by_vector: {e}")
                raise ConnectionError(f"Vector search service error: {str(e)}")

        return await asyncio.to_thread(_query_sync)

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in Weaviate."""
        def _check_collection() -> bool:
            try:
                # Try v4 client method
                if hasattr(self._client, 'collections'):
                    return self._client.collections.exists(collection_name)
                # Try v3 client method
                else:
                    schema = self._client.schema.get()
                    class_names = [cls["class"] for cls in schema.get("classes", [])]
                    return collection_name in class_names
            except WeaviateBaseError as e:
                logger.error(f"Weaviate error checking collection existence: {e}")
                error_msg = str(e).lower()
                if "unauthorized" in error_msg or "authentication" in error_msg:
                    raise ConnectionError("Weaviate authentication failed")
                elif "connection" in error_msg or "network" in error_msg:
                    raise ConnectionError("Cannot connect to Weaviate service")
                else:
                    return False  # Collection doesn't exist or other non-critical error
                    
            except Exception as e:
                logger.error(f"Unexpected error checking collection existence: {e}")
                raise ConnectionError(f"Unable to check collection existence: {str(e)}")
        
        return await asyncio.to_thread(_check_collection)

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
        similarity_threshold: Optional[float] = None,
        company_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in a specific collection with optional company filtering."""
        
        # Company filtering disabled for now
        filters = None
        # if company_id:
        #     filters = {
        #         "path": ["company_id"],
        #         "operator": "Equal",
        #         "valueText": company_id
        #     }
        
        # Return all actual properties from the database
        results = await self.search_by_vector(
            class_name=collection_name,
            vector=query_vector,
            top_k=top_k,
            return_properties=["text", "company_id", "doc_id", "chunk_id", "page_start", "page_end", "char_start", "char_end", "token_count"],
            filters=filters,
            include_distance=True
        )
        
        
        if similarity_threshold is not None:
            results = [r for r in results if r.get("distance", 1.0) <= similarity_threshold]
        
        formatted_results = []
        for r in results:
            formatted_result = {
                "id": r["id"],
                "content": r["properties"].get("text", ""),
                "metadata": {
                    # All stored database parameters (based on actual CargaConocimiento_iA schema)
                    "company_id": r["properties"].get("company_id", ""),
                    "doc_id": r["properties"].get("doc_id", ""),
                    "chunk_id": r["properties"].get("chunk_id", ""),
                    "page_start": r["properties"].get("page_start"),
                    "page_end": r["properties"].get("page_end"),
                    "char_start": r["properties"].get("char_start"),
                    "char_end": r["properties"].get("char_end"),
                    "token_count": r["properties"].get("token_count"),
                    # Search metadata
                    "distance": r.get("distance"),
                    "relevance_score": 1.0 - r.get("distance", 0.0) if r.get("distance") is not None else None
                }
            }
            formatted_results.append(formatted_result)
        
        return formatted_results

    async def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        return await self.health()

    async def health(self) -> bool:
        """Ping simple del cluster."""
        def _is_ready() -> bool:
            try:
                # Try v4 client method
                if hasattr(self._client, 'is_ready'):
                    return self._client.is_ready()
                # Try v3 client method
                elif hasattr(self._client, 'is_live'):
                    return self._client.is_live()
                else:
                    # Fallback test query
                    self._client.schema.get()
                    return True
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
