import numpy as np
from typing import Tuple, List, Dict, Optional, Any
import logging
import os
import threading
from pathlib import Path
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    PointIdsList,
)

from recmem.vector_store.base import SearchHit, VectorStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class QdrantStore(VectorStore):
    """
    Qdrant-based vector store with user-managed collections.

    Design Philosophy:
    - No internal collection state management
    - All operations require explicit collection_name
    - User is responsible for collection lifecycle
    - Supports multi-threading via collection isolation
    """

    # Default key for storing main content in payload
    DEFAULT_CONTENT_KEY = "text"

    def __init__(
        self,
        embedding_dim: int,
        path: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        on_disk: bool = True,
        exact_search: bool = True,
    ) -> None:
        """
        Initialize the Qdrant vector store.

        Args:
            embedding_dim: Dimension of embeddings
            path: Path for local Qdrant database (will be created if not exists)
            host: Host address for remote Qdrant server
            port: Port for remote Qdrant server
            url: Full URL for remote Qdrant server
            api_key: API key for remote Qdrant server
            on_disk: Whether to persist data to disk
            exact_search: Whether to use exact search
        """
        self._lock = threading.Lock()
        self.embedding_dim = embedding_dim
        self.on_disk = on_disk
        self.exact_search = exact_search

        # Initialize Qdrant client
        if host and port:
            params = {"host": host, "port": port}
            if api_key:
                params["api_key"] = api_key
            self.client = QdrantClient(**params)  # type: ignore
            self.is_local = False
            logger.info(f"Connected to remote Qdrant server at {host}:{port}")
        elif url:
            params = {"url": url}
            if api_key:
                params["api_key"] = api_key
            self.client = QdrantClient(**params)  # type: ignore
            self.is_local = False
            logger.info(f"Connected to remote Qdrant server at {url}")
        else:
            # Local Qdrant with persistent storage
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=path)
            self.is_local = True
            logger.info(f"Initialized local Qdrant at {path}")

        # Log existing collections
        self._log_existing_collections()

    def _log_existing_collections(self) -> None:
        """Log existing collections on startup."""
        try:
            collections = self.client.get_collections().collections
            if collections:
                collection_names = [col.name for col in collections]
                logger.info(
                    f"Found {len(collections)} existing collection(s): {collection_names}"
                )
            else:
                logger.info("No existing collections found")
        except Exception as e:
            logger.warning(f"Could not list collections: {e}")

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection to check

        Returns:
            True if collection exists, False otherwise
        """
        try:
            collections = self.client.get_collections().collections
            return any(col.name == collection_name for col in collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def create_collection(self, collection_name: str) -> None:
        """
        Create a new collection.

        Args:
            collection_name: Name of the collection to create

        Raises:
            ValueError: If collection already exists
        """
        try:
            if self.collection_exists(collection_name):
                raise ValueError(f"Collection '{collection_name}' already exists")

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=self.on_disk,
                ),
            )
            logger.info(f"Created new collection: {collection_name}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise

    def list_collections(self) -> List[str]:
        """
        List all available collections.

        Returns:
            List of collection names
        """
        try:
            collections = self.client.get_collections().collections
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    def add(
        self,
        *,
        embedding: List[float],
        payload: str,
        id: str,
        collection_name: str,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Upsert an embedding into the store, creating the collection if needed.

        Args:
            embedding: Vector embedding to add
            payload: Text payload stored under the main content key
            id: Unique identifier for the embedding
            collection_name: Name of the collection to add to. Will be created
                with the configured embedding dimension if it does not exist.
            extra_payload: Optional side metadata merged into the stored
                payload alongside the main content key.

        Raises:
            ValueError: If the embedding dimension does not match the store's
                configured embedding dimension.
        """
        if not self.collection_exists(collection_name):
            self.create_collection(collection_name)

        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expecting {self.embedding_dim}, got {len(embedding)}"
            )

        if extra_payload is None:
            extra_payload = {}

        point = PointStruct(
            id=id,
            vector=embedding,
            payload={self.DEFAULT_CONTENT_KEY: payload, **extra_payload},
        )

        self.client.upsert(collection_name=collection_name, points=[point])
        logger.debug(f"Added/updated embedding with id {id} in {collection_name}")

    def add_batch(
        self,
        *,
        embeddings: List[List[float]],
        payloads: List[str],
        ids: List[str],
        collection_name: str,
        extra_payloads: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """
        Upsert multiple points atomically in a single Qdrant call.

        All input lists must be the same length. An empty batch is a no-op.
        The collection is created if it does not yet exist.

        Raises:
            ValueError: If list lengths disagree or any embedding's dimension
                does not match the store's configured embedding dimension.
        """
        n = len(embeddings)
        if not (n == len(payloads) == len(ids)):
            raise ValueError(
                f"add_batch length mismatch: embeddings={len(embeddings)}, "
                f"payloads={len(payloads)}, ids={len(ids)}"
            )
        if extra_payloads is not None and len(extra_payloads) != n:
            raise ValueError(
                f"add_batch extra_payloads length {len(extra_payloads)} does "
                f"not match batch size {n}"
            )
        if n == 0:
            return

        for i, emb in enumerate(embeddings):
            if len(emb) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch at index {i}. "
                    f"Expecting {self.embedding_dim}, got {len(emb)}"
                )

        if not self.collection_exists(collection_name):
            self.create_collection(collection_name)

        points: List[PointStruct] = []
        for i in range(n):
            ep = extra_payloads[i] if extra_payloads is not None else None
            if ep is None:
                ep = {}
            points.append(
                PointStruct(
                    id=ids[i],
                    vector=embeddings[i],
                    payload={self.DEFAULT_CONTENT_KEY: payloads[i], **ep},
                )
            )

        self.client.upsert(collection_name=collection_name, points=points)
        logger.debug(f"Batch-added {n} points into {collection_name}")

    def list_memories(self, collection_name: str) -> List[str]:
        """
        List all memory texts stored in a collection.

        Args:
            collection_name: Name of the collection to list from

        Returns:
            List of memory texts

        Raises:
            ValueError: If collection does not exist
        """
        if not self.collection_exists(collection_name):
            return []

        try:
            points, _ = self.client.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )

            memories = [
                point.payload.get(self.DEFAULT_CONTENT_KEY, "") for point in points  # type: ignore
            ]
            return memories
        except Exception as e:
            logger.error(f"Error listing memories from {collection_name}: {e}")
            return []

    def search(
        self,
        *,
        q_embedding: List[float],
        top_k: int,
        collection_name: str,
    ) -> List[SearchHit]:
        """
        Search for similar embeddings, returning hits sorted by score desc.

        Raises:
            ValueError: If embedding dimension mismatches the collection.
        """
        if not self.collection_exists(collection_name):
            return []

        if len(q_embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(q_embedding)}"
            )

        results = self.client.query_points(
            collection_name=collection_name,
            query=q_embedding,
            limit=top_k,
            with_payload=True,
            search_params=models.SearchParams(exact=self.exact_search),
        )

        hits: List[SearchHit] = []
        for point in results.points:
            payload = dict(point.payload) if point.payload else {}
            content = payload.pop(self.DEFAULT_CONTENT_KEY, "")
            hits.append(
                SearchHit(
                    content=content,
                    score=point.score,
                    id=str(point.id),
                    extra_payload=payload,
                )
            )
        return hits

    def remove(self, ids: List[str], collection_name: str) -> None:
        """
        Remove embeddings by IDs.

        Args:
            ids: List of IDs to remove
            collection_name: Name of the collection to remove from

        Raises:
            ValueError: If collection does not exist
        """
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=ids),  # type: ignore
            )
            logger.debug(f"Removed {len(ids)} points from {collection_name}")
        except Exception as e:
            logger.warning(f"[{collection_name}] Error removing points: {e}")

    def reset(self, collection_name: str) -> None:
        """
        Reset a collection by deleting and recreating it.

        No-op if the collection does not exist.

        Args:
            collection_name: Name of the collection to reset

        Raises:
            Exception: Propagated from the underlying client if the
                delete-then-recreate sequence fails.
        """
        with self._lock:
            if not self.collection_exists(collection_name):
                return
            try:
                # Delete and recreate collection
                self.client.delete_collection(collection_name=collection_name)
                logger.info(f"Deleted collection: {collection_name}")

                self.create_collection(collection_name)
                logger.info(f"Reset collection: {collection_name}")
            except Exception as e:
                logger.error(f"[{collection_name}] Error resetting collection: {e}")
                raise

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection completely.

        Args:
            collection_name: Name of the collection to delete

        Raises:
            ValueError: If collection does not exist
        """
        if not self.collection_exists(collection_name):
            return

        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            raise

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection.

        Args:
            collection_name: Name of the collection to get info for

        Returns:
            Dictionary with collection information

        Raises:
            ValueError: If collection does not exist
        """
        if not self.collection_exists(collection_name):
            return {}

        try:
            info = self.client.get_collection(collection_name=collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {e}")
            return {}
