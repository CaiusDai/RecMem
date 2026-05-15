from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SearchHit:
    """A single result from a vector store similarity search."""

    content: str
    score: float
    id: str
    extra_payload: Dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract base class for vector store providers.

    Implementations expose the operations RecMem actually needs: collection
    creation, point add / remove / reset, and similarity search returning
    `SearchHit` objects (instead of parallel-list tuples).
    """

    @abstractmethod
    def add(
        self,
        *,
        embedding: List[float],
        payload: str,
        id: str,
        collection_name: str,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upsert a single point into a collection.

        If the collection does not yet exist, implementations MUST create it
        before inserting. Callers rely on this because collection names are
        often per-conversation and not known at init time.

        `payload` is the main text content. `extra_payload` carries optional
        side metadata stored alongside it.
        """
        ...

    @abstractmethod
    def add_batch(
        self,
        *,
        embeddings: List[List[float]],
        payloads: List[str],
        ids: List[str],
        collection_name: str,
        extra_payloads: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """Upsert multiple points into a collection in a single atomic call.

        Parameters mirror `add` but each scalar becomes a list. All input lists
        MUST be the same length; `extra_payloads`, when provided, must also
        match. An empty batch is a no-op.

        Implementations MUST submit all points in one underlying call so that
        the operation is atomic at the store level — either every point is
        written, or none is. This is the key reason for this method to exist
        (callers use it to avoid partial writes on failure mid-loop).

        Like `add`, the collection is created if it does not yet exist.
        """
        ...

    @abstractmethod
    def remove(self, ids: List[str], collection_name: str) -> None:
        """Remove points by id from a collection."""
        ...

    @abstractmethod
    def reset(self, collection_name: str) -> None:
        """Remove all points from a collection (collection itself may persist)."""
        ...

    @abstractmethod
    def search(
        self,
        *,
        q_embedding: List[float],
        top_k: int,
        collection_name: str,
    ) -> List[SearchHit]:
        """Return up to `top_k` hits from a collection, sorted by score desc."""
        ...

    @abstractmethod
    def list_memories(self, collection_name: str) -> List[str]:
        """Return the text content of every point in a collection."""
        ...

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Whether a collection with this name has been created."""
        ...

    @abstractmethod
    def create_collection(self, collection_name: str) -> None:
        """Create a new collection. No-op or error if it already exists is implementation-defined."""
        ...
