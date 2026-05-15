import os
from typing import List, Optional

from openai import OpenAI
import tiktoken

import logging
import time

from recmem.embedding.base import Embedding

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenAIEmbedding(Embedding):
    # OpenAI embedding models have a max of 8191 tokens
    MAX_TOKENS = 8191

    def __init__(
        self, model: str = "text-embedding-3-small", embedding_dims: int = 1536
    ):

        self.model = model
        self.embedding_dims = embedding_dims

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = (
            os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._tokenizer: Optional[tiktoken.Encoding] = None

    @property
    def dim(self) -> int:
        return self.embedding_dims

    @property
    def tokenizer(self) -> tiktoken.Encoding:
        """Lazy load tokenizer only when needed."""
        if self._tokenizer is None:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def truncate_to_max_tokens(self, text: str) -> str:
        """Truncate text to fit within MAX_TOKENS."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.MAX_TOKENS:
            logger.warning(
                f"Text has {len(tokens)} tokens, truncating to {self.MAX_TOKENS}."
            )
            tokens = tokens[: self.MAX_TOKENS]
            text = self.tokenizer.decode(tokens)
        return text

    def embed(self, text: str) -> List[float]:
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.
        Returns:
            list: The embedding vector.
        Raises:
            RuntimeError: If all retry attempts fail.
        """
        text = text.replace("\n", " ")
        truncated = False
        last_error: Optional[Exception] = None
        for i in range(3):  # Retry 3 times
            try:
                embedding = (
                    self.client.embeddings.create(
                        input=[text],
                        model=self.model,
                        dimensions=self.embedding_dims,
                    )
                    .data[0]
                    .embedding
                )
                return embedding
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Error embedding text: {e}. Retrying ({i+1}/3)."
                )
                # Lazy truncation: only count and truncate tokens after first failure
                if not truncated:
                    text = self.truncate_to_max_tokens(text)
                    truncated = True
                else:
                    time.sleep(1 + i * 2)

        raise RuntimeError(
            f"Failed to embed text after 3 retries. Last error: {last_error}"
        )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in a single API call.

        Args:
            texts: List of texts to embed.
        Returns:
            List of embedding vectors in the same order as input texts.
        Raises:
            RuntimeError: If all retry attempts fail.
        """
        if not texts:
            return []

        # Preprocess texts
        processed_texts = [text.replace("\n", " ") for text in texts]
        truncated = False
        last_error: Optional[Exception] = None

        for i in range(3):  # Retry 3 times
            try:
                response = self.client.embeddings.create(
                    input=processed_texts,
                    model=self.model,
                    dimensions=self.embedding_dims,
                )
                # Sort by index to ensure correct order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in sorted_data]
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Error embedding batch: {e}. Retrying ({i+1}/3)."
                )
                # Lazy truncation: only truncate after first failure
                if not truncated:
                    processed_texts = [
                        self.truncate_to_max_tokens(text) for text in processed_texts
                    ]
                    truncated = True
                else:
                    time.sleep(1 + i * 2)

        raise RuntimeError(
            f"Failed to embed batch after 3 retries. Last error: {last_error}"
        )
