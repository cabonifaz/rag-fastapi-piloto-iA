from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TypedDict

import asyncio
import logging
import weaviate
from weaviate.exceptions import WeaviateBaseError
from weaviate.classes.query import Filter

from app.domain.ports.vectorstore_port import VectorStorePort
from app.core.config import settings

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
            alpha: Balance between vector (1.0) and keyword (0.0) search. General 0.5 for balanced hybrid.
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
        return await self.search_in_collection(
            collection_name=settings.weaviate_class_name,
            query_vector=query_vector,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            area=area
        )

    async def search_in_collection(
        self,
        company_id: int,
        area_id: int,
        query_vector: List[float],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        general_area: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in a company-scoped collection with area filtering.

        Args:
            company_id: Company ID (will be formatted as "EMPR{company_id}" for collection name)
            area_id: Area ID (will be formatted as "AREA{area_id}" for filtering)
            query_vector: Vector for semantic search
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            general_area: Area ID for general/shared documents. If None, uses area_id.
        """

        try:
            # Format IDs with prefixes for Weaviate collection and area filtering
            collection_name = f"EMPR{company_id}"
            area = f"AREA{area_id}"
            general = f"AREA{general_area if general_area is not None else area_id}"

            # Use environment defaults if not provided
            actual_top_k = top_k if top_k is not None else settings.rag_top_k_results
            actual_similarity_threshold = similarity_threshold if similarity_threshold is not None else settings.rag_similarity_threshold

            # Validate inputs
            if company_id is None:
                raise ValueError("Company ID is required")
            if area_id is None:
                raise ValueError("Area ID is required")
            if not query_vector:
                raise ValueError("Query vector cannot be empty")
            if actual_top_k <= 0:
                raise ValueError("top_k must be greater than 0")
            if similarity_threshold is not None and not (0.0 <= similarity_threshold <= 1.0):
                raise ValueError("Similarity threshold must be between 0.0 and 1.0")

            # Build area filter using v4 Filter class
            # Note: No company filter needed - collection itself is scoped to company
            # Include both the specific area_id AND general area documents
            # area_id contains concatenated values like "AREA123"
            area_filter = (
                Filter.by_property("area_id").equal(area) |
                Filter.by_property("area_id").equal(general)
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
        company_id: int,
        area_id: int,
        query_text: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        alpha: Optional[float] = None,
        general_area: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Hybrid search (vector + BM25) in a company-scoped collection with area filtering.

        Args:
            company_id: Company ID (will be formatted as "EMPR{company_id}" for collection name)
            area_id: Area ID (will be formatted as "AREA{area_id}" for filtering)
            query_text: Text query for BM25 search
            query_vector: Vector for semantic search
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            alpha: Hybrid search weight (0.0 = pure BM25, 1.0 = pure vector)
            general_area: Area ID for general/shared documents. If None, uses area_id.
        """

        try:
            # Format IDs with prefixes for Weaviate collection and area filtering
            collection_name = f"EMPR{company_id}"
            area = f"AREA{area_id}"
            general = f"AREA{general_area if general_area is not None else area_id}"

            # Use environment defaults if not provided
            actual_top_k = top_k if top_k is not None else settings.rag_top_k_results
            actual_similarity_threshold = similarity_threshold if similarity_threshold is not None else settings.rag_similarity_threshold

            # Validate inputs
            if company_id is None:
                raise ValueError("Company ID is required")
            if area_id is None:
                raise ValueError("Area ID is required")
            if not query_text:
                raise ValueError("Query text cannot be empty")
            if not query_vector:
                raise ValueError("Query vector cannot be empty")
            if actual_top_k <= 0:
                raise ValueError("top_k must be greater than 0")
            if similarity_threshold is not None and not (0.0 <= similarity_threshold <= 1.0):
                raise ValueError("Similarity threshold must be between 0.0 and 1.0")

            # Build area filter using v4 Filter class
            # Note: No company filter needed - collection itself is scoped to company
            # Include both the specific area_id AND General area documents
            # area_id contains concatenated values like "AREA123"
            # area contains "General" for shared documents
            area_filter = (
                Filter.by_property("area_id").equal(area) |
                Filter.by_property("area_id").equal(general)
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

    async def delete_by_doc_ids(
        self,
        class_name: str,
        doc_ids: List[str],
        company_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete all objects with specific doc_ids from a collection.

        Args:
            class_name: Name of the collection (e.g., "EMPR123")
            doc_ids: List of document IDs to delete (e.g., ["123", "124", "125"])
            company_id: Optional company ID for additional filtering

        Returns:
            Dictionary with deletion results:
            - success: bool (True only if deleted_count > 0 and no errors)
            - deleted_count: int (number of objects deleted)
            - not_found: List[str] (doc_ids that were not found in Weaviate)
            - errors: List[str] (error messages if any)
        """
        def _delete_sync() -> Dict[str, Any]:
            try:
                collection = self._client.collections.get(class_name)

                deleted_count = 0
                errors = []
                not_found_docs = []

                for doc_id_int in doc_ids:
                    doc_id = f"CONOC-{doc_id_int}"
                    try:
                        # Build filter for doc_id
                        filter_condition = Filter.by_property("doc_id").equal(doc_id)

                        # Add company_id filter if provided
                        if company_id:
                            company_filter = Filter.by_property("company_id").equal(company_id)
                            filter_condition = filter_condition & company_filter

                        # STEP 1: Check if objects exist before attempting deletion
                        try:
                            check_query = collection.query.fetch_objects(
                                filters=filter_condition,
                                limit=1
                            )

                            objects_exist = len(check_query.objects) > 0

                            if not objects_exist:
                                warning_msg = f"No objects found with doc_id={doc_id} in collection {class_name}"
                                logger.warning(warning_msg)
                                not_found_docs.append(doc_id)
                                continue  # Skip deletion if no objects found

                        except Exception as check_error:
                            logger.warning(f"Error checking existence of doc_id={doc_id}: {check_error}")
                            # Continue with deletion attempt even if check fails

                        # STEP 2: Delete all objects matching the filter
                        result = collection.data.delete_many(
                            where=filter_condition
                        )

                        # Count successful deletions
                        objects_deleted = 0
                        if hasattr(result, 'successful') and result.successful:
                            objects_deleted = result.successful
                            deleted_count += result.successful
                        elif hasattr(result, 'matches') and result.matches:
                            objects_deleted = result.matches
                            deleted_count += result.matches

                        if objects_deleted > 0:
                            logger.info(f"Deleted {objects_deleted} objects with doc_id={doc_id} from collection {class_name}")
                        else:
                            warning_msg = f"No objects deleted for doc_id={doc_id} (may not exist)"
                            logger.warning(warning_msg)
                            not_found_docs.append(doc_id)

                    except WeaviateBaseError as e:
                        error_msg = f"Error deleting doc_id {doc_id}: {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                    except Exception as e:
                        error_msg = f"Unexpected error deleting doc_id {doc_id}: {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)

                # Check if nothing was deleted
                if deleted_count == 0 and len(doc_ids) > 0:
                    error_msg = f"No objects were deleted. Documents not found: {', '.join(not_found_docs)}"
                    logger.error(error_msg)
                    errors.append(error_msg)

                return {
                    "success": len(errors) == 0 and deleted_count > 0,
                    "deleted_count": deleted_count,
                    "not_found": not_found_docs,
                    "errors": errors
                }

            except WeaviateBaseError as e:
                logger.error(f"Weaviate error in delete_by_doc_ids: {e}")
                error_msg = str(e).lower()
                if "unauthorized" in error_msg or "authentication" in error_msg:
                    raise ConnectionError("Weaviate authentication failed")
                elif "not found" in error_msg or "does not exist" in error_msg:
                    raise ValueError(f"Collection '{class_name}' not found in Weaviate")
                elif "connection" in error_msg or "network" in error_msg:
                    raise ConnectionError("Cannot connect to Weaviate service")
                else:
                    raise ConnectionError(f"Weaviate service error: {str(e)}")

            except Exception as e:
                logger.error(f"Unexpected error in delete_by_doc_ids: {e}")
                raise ConnectionError(f"Delete service error: {str(e)}")

        return await asyncio.to_thread(_delete_sync)

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
