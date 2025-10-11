from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TypedDict

import asyncio
import logging
import weaviate
from weaviate.exceptions import WeaviateBaseError
from weaviate.classes.query import Filter

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

    def _search_vector(self, collection, vector: List[float], actual_top_k: int,
                       return_properties: Optional[List[str]], filters: Optional[Any],
                       include_distance: bool):
        """Vector similarity search using collection's configured distance metric."""
        query_kwargs = {
            "near_vector": list(vector),
            "limit": actual_top_k,
            "return_metadata": ["distance"] if include_distance else [],
        }

        if return_properties:
            query_kwargs["return_properties"] = list(return_properties)

        if filters:
            return collection.query.near_vector(
                near_vector=list(vector),
                limit=actual_top_k,
                return_metadata=["distance"] if include_distance else [],
                return_properties=list(return_properties) if return_properties else None,
                filters=filters
            )
        else:
            return collection.query.near_vector(**query_kwargs)

    def _search_hybrid(self, collection, query_text: str, vector: List[float],
                       actual_top_k: int, return_properties: Optional[List[str]],
                       filters: Optional[Any], include_distance: bool, alpha: float = 0.5):
        """Hybrid search combining vector similarity and BM25 keyword search.

        Args:
            collection: Weaviate collection object
            query_text: Text query for BM25 keyword search
            vector: Query embedding for vector similarity
            actual_top_k: Number of results to return
            return_properties: Properties to return in results
            filters: Optional filters to apply
            include_distance: Whether to include distance scores
            alpha: Balance between vector (1.0) and keyword (0.0) search. Default 0.5 for balanced hybrid.
        """
        query_kwargs = {
            "query": query_text,
            "vector": list(vector),
            "alpha": alpha,
            "limit": actual_top_k,
            "return_metadata": ["score"] if include_distance else [],
        }

        if return_properties:
            query_kwargs["return_properties"] = list(return_properties)

        if filters:
            return collection.query.hybrid(
                query=query_text,
                vector=list(vector),
                alpha=alpha,
                limit=actual_top_k,
                return_metadata=["score"] if include_distance else [],
                return_properties=list(return_properties) if return_properties else None,
                filters=filters
            )
        else:
            return collection.query.hybrid(**query_kwargs)

    async def search_by_vector(
        self,
        class_name: str,
        vector: Sequence[float],
        top_k: Optional[int] = None,
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

        # Use environment default if not provided
        from app.core.config import settings
        actual_top_k = top_k if top_k is not None else settings.rag_top_k_results

        def _query_sync() -> List[VectorSearchResult]:
            try:
                # v4 client method
                collection = self._client.collections.get(class_name)
                
                # First try a simple get to see if collection has any data
                simple_response = collection.query.fetch_objects(limit=1)

                # Perform vector search
                response = self._search_vector(collection, vector, actual_top_k, return_properties, filters, include_distance)

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

    async def search_hybrid(
        self,
        class_name: str,
        query_text: str,
        vector: Sequence[float],
        top_k: Optional[int] = None,
        return_properties: Optional[Sequence[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
        target_vector: Optional[str] = None,
        tenant: Optional[str] = None,
        include_distance: bool = True,
    ) -> List[VectorSearchResult]:
        """
        Hybrid search combining vector similarity and BM25 keyword search.

        Args:
            class_name: nombre de la colección en Weaviate.
            query_text: texto de consulta para BM25.
            vector: embedding de consulta para búsqueda vectorial.
            top_k: máximo de objetos a recuperar.
            return_properties: lista de propiedades a devolver.
            filters: filtro opcional.
            alpha: balance entre vector (1.0) y keyword (0.0). None = usar settings.rag_hybrid_alpha.
            target_vector: si usas 'named vectors', especifica cuál buscar.
            tenant: identificador del tenant si la colección es multi-tenant.
            include_distance: si True, pide score en metadatos.

        Returns:
            Lista de dicts con: id (uuid), properties (dict) y score (float|None).
        """

        from app.core.config import settings
        actual_top_k = top_k if top_k is not None else settings.rag_top_k_results
        actual_alpha = alpha if alpha is not None else settings.rag_hybrid_alpha

        def _query_sync() -> List[VectorSearchResult]:
            try:
                collection = self._client.collections.get(class_name)

                # Perform hybrid search
                response = self._search_hybrid(
                    collection, query_text, vector, actual_top_k,
                    return_properties, filters, include_distance, actual_alpha
                )

                results: List[VectorSearchResult] = []
                for obj in response.objects:
                    item: VectorSearchResult = {
                        "id": str(obj.uuid),
                        "properties": dict(obj.properties) if obj.properties else {},
                    }
                    if include_distance and obj.metadata and hasattr(obj.metadata, 'score'):
                        # Hybrid search returns 'score' (higher = better), not 'distance'
                        # Store score as-is without conversion
                        item["score"] = obj.metadata.score if obj.metadata.score is not None else None
                        item["distance"] = None  # Not applicable for hybrid search
                    results.append(item)

                return results

            except WeaviateBaseError as e:
                logger.error(f"Weaviate error in search_hybrid: {e}")
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
                raise

            except TimeoutError:
                raise

            except ValueError as e:
                logger.error(f"Value error in search_hybrid: {e}")
                if "collection" not in str(e).lower():
                    raise ValueError(f"Invalid search parameters: {str(e)}")
                raise

            except Exception as e:
                logger.error(f"Unexpected error in search_hybrid: {e}")
                raise ConnectionError(f"Hybrid search service error: {str(e)}")

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
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        area: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors using the default class name from settings."""
        from app.core.config import settings
        
        return await self.search_in_collection(
            collection_name=settings.weaviate_class_name,
            query_vector=query_vector,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            area=area
        )

    async def search_in_collection(
        self,
        collection_name: str,
        query_vector: List[float],
        area: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in a company-scoped collection with area filtering.

        Note: collection_name is the company identifier - each company has its own collection.
        """

        try:
            from app.core.config import settings

            # Use environment defaults if not provided
            actual_top_k = top_k if top_k is not None else settings.rag_top_k_results
            actual_similarity_threshold = similarity_threshold if similarity_threshold is not None else settings.rag_similarity_threshold

            # Validate inputs
            if not collection_name:
                raise ValueError("Collection name cannot be empty")
            if not query_vector:
                raise ValueError("Query vector cannot be empty")
            if not area:
                raise ValueError("Area is required")
            if actual_top_k <= 0:
                raise ValueError("top_k must be greater than 0")
            if similarity_threshold is not None and not (0.0 <= similarity_threshold <= 1.0):
                raise ValueError("Similarity threshold must be between 0.0 and 1.0")

            # Build area filter using v4 Filter class
            # Note: No company filter needed - collection itself is scoped to company
            # Include both the specific area AND Default area documents
            area_filter = (
                Filter.by_property("area").equal(area) |
                Filter.by_property("area").equal("Default")
            )

            filters = area_filter
            
            # Vector similarity search
            results = await self.search_by_vector(
                class_name=collection_name,
                vector=query_vector,
                top_k=actual_top_k,
                return_properties=["text", "doc_id", "chunk_id", "page_start", "page_end", "char_start", "char_end", "token_count"],
                filters=filters,
                include_distance=True
            )
            
        except ValueError:
            raise  # Re-raise validation errors
        except ConnectionError:
            raise  # Re-raise connection errors from search_by_vector
        except TimeoutError:
            raise  # Re-raise timeout errors from search_by_vector
        except Exception as e:
            logger.error(f"Unexpected error in search_in_collection: {e}")
            raise ConnectionError(f"Vector search service error: {str(e)}")

        if actual_similarity_threshold is not None:
            for i, r in enumerate(results):
                distance = r.get('distance', 1.0)
                similarity = 1.0 - distance if distance is not None else 0.0
            # Convert similarity_threshold to distance_threshold and filter
            distance_threshold = 1.0 - actual_similarity_threshold
            results = [r for r in results if r.get("distance", 1.0) <= distance_threshold]
        
        formatted_results = []
        for r in results:
            formatted_result = {
                "id": r["id"],
                "content": r["properties"].get("text", ""),
                "metadata": {
                    # Document metadata
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

    async def search_in_collection_hybrid(
        self,
        collection_name: str,
        query_text: str,
        query_vector: List[float],
        area: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        alpha: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Hybrid search (vector + BM25) in a company-scoped collection with area filtering.

        Note: collection_name is the company identifier - each company has its own collection.
        """

        try:
            from app.core.config import settings

            # Use environment defaults if not provided
            actual_top_k = top_k if top_k is not None else settings.rag_top_k_results
            actual_similarity_threshold = similarity_threshold if similarity_threshold is not None else settings.rag_similarity_threshold

            # Validate inputs
            if not collection_name:
                raise ValueError("Collection name cannot be empty")
            if not query_text:
                raise ValueError("Query text cannot be empty")
            if not query_vector:
                raise ValueError("Query vector cannot be empty")
            if not area:
                raise ValueError("Area is required")
            if actual_top_k <= 0:
                raise ValueError("top_k must be greater than 0")
            if similarity_threshold is not None and not (0.0 <= similarity_threshold <= 1.0):
                raise ValueError("Similarity threshold must be between 0.0 and 1.0")

            # Build area filter using v4 Filter class
            # Note: No company filter needed - collection itself is scoped to company
            # Include both the specific area AND Default area documents
            area_filter = (
                Filter.by_property("area").equal(area) |
                Filter.by_property("area").equal("Default")
            )

            filters = area_filter

            # HYBRID SEARCH (vector + BM25 keyword)
            results = await self.search_hybrid(
                class_name=collection_name,
                query_text=query_text,
                vector=query_vector,
                top_k=actual_top_k,
                return_properties=["text", "doc_id", "doc_title", "section_title", "section_path", "chunk_id", "page_start", "page_end", "char_start", "char_end", "token_count"],
                filters=filters,
                alpha=alpha,
                include_distance=True
            )

        except ValueError:
            raise  # Re-raise validation errors
        except ConnectionError:
            raise  # Re-raise connection errors from search_hybrid
        except TimeoutError:
            raise  # Re-raise timeout errors from search_hybrid
        except Exception as e:
            logger.error(f"Unexpected error in search_in_collection_hybrid: {e}")
            raise ConnectionError(f"Hybrid search service error: {str(e)}")

        if actual_similarity_threshold is not None:
            # For hybrid search, filter by score (higher = better)
            # Score is already normalized 0-1, so use threshold directly
            results = [r for r in results if r.get("score", 0.0) >= actual_similarity_threshold]

        formatted_results = []
        for r in results:
            # For hybrid search, use score directly (higher = better)
            score = r.get("score")
            distance = r.get("distance")

            formatted_result = {
                "id": r["id"],
                "content": r["properties"].get("text", ""),
                "metadata": {
                    # Document metadata
                    "doc_id": r["properties"].get("doc_id", ""),
                    "doc_title": r["properties"].get("doc_title", ""),
                    "section_title": r["properties"].get("section_title", ""),
                    "section_path": r["properties"].get("section_path", ""),
                    "chunk_id": r["properties"].get("chunk_id", ""),
                    "page_start": r["properties"].get("page_start"),
                    "page_end": r["properties"].get("page_end"),
                    "char_start": r["properties"].get("char_start"),
                    "char_end": r["properties"].get("char_end"),
                    "token_count": r["properties"].get("token_count"),
                    # Search metadata for hybrid search
                    "score": score,  # Hybrid score (higher = better)
                    "relevance_score": score if score is not None else 0.0,  # Use score directly
                    "search_type": "hybrid"
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
