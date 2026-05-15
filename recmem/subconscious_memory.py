import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from recmem.embedding import Embedding, OpenAIEmbedding
from recmem.vector_store import QdrantStore, SearchHit, VectorStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()


class SubconsciousMemory:
    """
    SubconsciousMemory is a flat vector store of recent raw conversation snippets.

    It serves two roles:
    - A retrieval source at query time (via `retrieve_memory`).
    - The input pool for the consolidation flow that promotes recurring snippets
      into episodic memories (via `search` for gating + `add` / `remove`).

    `embedder` and `vector_store` may be injected for provider extensibility;
    when omitted, the OpenAI embedder and a local Qdrant store are used.
    """

    def __init__(
        self,
        embedding_dim: int,
        storage_path: Optional[str] = None,
        embedder: Optional[Embedding] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.embedder = embedder if embedder is not None else OpenAIEmbedding()
        if vector_store is not None:
            self.vec_store: VectorStore = vector_store
        else:
            if storage_path is None:
                storage_path = os.getenv("SUBCONSCIOUS_STORE") or ""
            self.vec_store = QdrantStore(
                embedding_dim=embedding_dim, path=storage_path, exact_search=True
            )

    def add(
        self,
        embedding: List[float],
        payload: str,
        id: str,
        conv_id: str,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.vec_store.add(
            embedding=embedding,
            payload=payload,
            id=id,
            collection_name=conv_id,
            extra_payload=extra_payload,
        )

    def remove(self, ids: List[str], conv_id: str) -> None:
        self.vec_store.remove(ids, conv_id)

    def reset(self, conv_id: str) -> None:
        self.vec_store.reset(conv_id)

    def search(
        self,
        *,
        q_embedding: List[float],
        top_k: int,
        conv_id: str,
    ) -> List[SearchHit]:
        """
        Low-level search returning rich `SearchHit` objects.

        Used by the consolidation gating logic, which needs ids and timestamps.
        For high-level retrieval use `retrieve_memory` instead.
        """
        return self.vec_store.search(
            q_embedding=q_embedding, top_k=top_k, collection_name=conv_id
        )

    def retrieve_memory(
        self,
        conv_id: str,
        top_k: int,
        query: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        if top_k <= 0:
            return []
        if query_embedding is None:
            if query is None:
                raise ValueError(
                    "retrieve_memory: either `query` or `query_embedding` must be provided."
                )
            try:
                query_embedding = self.embedder.embed(query)
            except Exception as e:
                logger.error(
                    f"[{conv_id}] Failed to embed query for subconscious retrieval: {e}. "
                    "Returning empty list."
                )
                return []
        hits = self.vec_store.search(
            q_embedding=query_embedding, top_k=top_k, collection_name=conv_id
        )
        results = [(h.content, h.score) for h in hits]
        if threshold is not None:
            results = [(c, s) for c, s in results if s > threshold]
        return results
