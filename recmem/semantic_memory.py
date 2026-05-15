from recmem.embedding import Embedding, OpenAIEmbedding
from recmem.vector_store import QdrantStore, VectorStore
import logging
import uuid
from typing import Optional, Dict, List, Any, Tuple
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()


class SemanticMemory:
    """
    SemanticMemory is a class that stores and searches for semantic memories.
    It uses a flat vector store to store the memories.
    Each memory piece is expected to be a concise string of a piece of factual information.

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
                storage_path = os.getenv("SEMANTIC_STORE") or ""
            if not storage_path:
                raise ValueError("SemanticMemory storage path is not set.")
            self.vec_store = QdrantStore(
                embedding_dim=embedding_dim, path=storage_path, exact_search=True
            )

    def add_memory(
        self, memory: str, conv_id: str, extra_payload: Optional[Dict[str, Any]] = None
    ) -> bool:
        mem_id: str = str(uuid.uuid4())
        try:
            embedding: List[float] = self.embedder.embed(memory)
        except Exception as e:
            logger.error(
                f"[{conv_id}] Failed to add semantic memory: embedding generation failed. Error: {e}"
                "Returning False."
            )
            return False
        self.vec_store.add(
            embedding=embedding,
            payload=memory,
            id=mem_id,
            collection_name=conv_id,
            extra_payload=extra_payload,
        )
        return True

    # TODO: extra_payload's type is wrong, fix later
    def add_memories(
        self,
        memories: List[str],
        conv_id: str,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if len(memories) == 0:
            return True
        try:
            memory_embeddings: List[List[float]] = self.embedder.embed_batch(memories)
        except Exception as e:
            logger.error(
                f"[{conv_id}] Failed to add semantic memories: batch embedding generation failed "
                f"for {len(memories)} memories. Error: {e}"
                "Returning False."
            )
            return False

        for memory, embedding in zip(memories, memory_embeddings):
            self.vec_store.add(
                embedding=embedding,
                payload=memory,
                id=str(uuid.uuid4()),
                collection_name=conv_id,
                extra_payload=extra_payload,
            )
        return True

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
                    f"[{conv_id}] Failed to search semantic memory: query embedding generation failed. "
                    f"Error: {e}"
                    "Returning empty list."
                )
                return []
        hits = self.vec_store.search(
            q_embedding=query_embedding, top_k=top_k, collection_name=conv_id
        )
        results = [(h.content, h.score) for h in hits]
        if threshold is not None:
            results = [(m, s) for m, s in results if s > threshold]
        return results

    def list_memories(self, conv_id: str) -> list[str]:
        return self.vec_store.list_memories(conv_id)

    def reset(self, conv_id: str) -> None:
        self.vec_store.reset(conv_id)
