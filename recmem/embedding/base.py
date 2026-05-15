from abc import ABC, abstractmethod
from typing import List


class Embedding(ABC):
    """Abstract base class for embedding providers.

    Implementations are responsible for converting text into dense vectors. They
    must report a consistent embedding dimension via the `dim` property so that
    vector stores can be initialized correctly.
    """

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return the embedding vector for a single text."""
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for a batch of texts, in input order."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """The dimensionality of vectors produced by this embedder."""
        ...
