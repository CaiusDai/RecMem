import os

from typing import List, Tuple, Optional
from recmem.token_monitor import TokenMonitor, OPType
from recmem.embedding import Embedding, OpenAIEmbedding
from recmem.prompt import EPISODIC_GENERATION_PROMPT, EPISODIC_MERGE_PROMPT
from recmem.vector_store import QdrantStore, VectorStore
from recmem.llm import (
    ContentFilteredException,
    EmptyResponseException,
    JSONParseException,
    LLMClient,
    Message,
    OpenAIClient,
)
import uuid
from jinja2 import Template
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EpisodicMemory:
    def __init__(
        self,
        path: Optional[str] = None,
        embedder: Optional[Embedding] = None,
        llm_client: Optional[LLMClient] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        if vector_store is not None:
            self.vec_store: VectorStore = vector_store
        else:
            if path is None:
                path = os.getenv("EPISODIC_STORE") or ""
            self.vec_store = QdrantStore(
                embedding_dim=1536, path=path, exact_search=True
            )
        logger.info(f"Episodic memory storage path: {path}")

        self.llm_client: LLMClient = llm_client if llm_client is not None else OpenAIClient()
        self.embedder: Embedding = embedder if embedder is not None else OpenAIEmbedding()
        self.monitor: Optional[TokenMonitor] = None

    def set_monitor(self, monitor: TokenMonitor) -> None:
        self.monitor = monitor

    def add(
        self,
        conversation: str,
        raw_ids: List[str],
        conv_id: str,
    ) -> List[str]:
        template = Template(EPISODIC_GENERATION_PROMPT)
        episodic_prompt = template.render(
            raw_conversation=conversation,
        )
        try:
            _, response_json = self.llm_client.chat_completion_json(
                model=os.getenv("MODEL"),
                messages=[Message(role="system", content=episodic_prompt)],
                temperature=0.0,
                monitor=self.monitor,
                op_type=OPType.EPISODIC_GENERATION,
            )
        except ContentFilteredException as e:
            logger.warning(
                f"[{conv_id}] Content filtered during episodic memory generation: {e}. "
                f"Reason: {e.finish_reason}. Returning empty list."
            )
            return []
        except EmptyResponseException as e:
            logger.warning(
                f"[{conv_id}] Empty response during episodic memory generation: {e}. "
                f"Reason: {e.finish_reason}. Returning empty list."
            )
            return []
        except JSONParseException as e:
            logger.error(
                f"[{conv_id}] Error parsing episodic memory generation response: {e}. "
                f"Response: {e.raw_content}. Returning empty list."
            )
            return []
        except Exception as e:
            # OpenAI API errors (after all retries failed)
            logger.error(
                f"[{conv_id}] All retries failed for episodic memory generation: {e}. "
                "Returning empty list."
            )
            return []

        episodes = response_json.get("episodes", [])
        if len(episodes) == 0:
            logger.warning(
                f"[{conv_id}] No facts found in add memory response. Response: {response_json}"
            )
            return []

        # Embed every episode first so that any failure here leaves the vector
        # store untouched. Only after all embeddings succeed do we commit them
        # in one atomic add_batch call — this avoids partial-write ghosts that
        # the prior single-add loop could produce.
        episodic_memories: List[str] = []
        embeddings: List[List[float]] = []
        ids: List[str] = []
        extra_payloads: List[Optional[dict]] = []
        for episode in episodes:
            if not isinstance(episode, str):
                logger.warning(
                    f"[{conv_id}] Episodic memory generation failed: expected each episode "
                    f"to be string but received: {episodes}. Returning empty list."
                )
                return []
            try:
                embedding = self.embedder.embed(episode)
            except Exception as e:
                logger.error(
                    f"[{conv_id}] Error embedding episodic memory: {e}. Returning empty list."
                )
                return []
            episodic_memories.append(episode)
            embeddings.append(embedding)
            ids.append(str(uuid.uuid4()))
            extra_payloads.append(
                {"conversation": conversation, "raw_ids": raw_ids}
            )

        try:
            self.vec_store.add_batch(
                embeddings=embeddings,
                payloads=episodic_memories,
                ids=ids,
                collection_name=conv_id,
                extra_payloads=extra_payloads,
            )
        except Exception as e:
            logger.error(
                f"[{conv_id}] Failed to write episodic memories to vector store: {e}. "
                "Returning empty list."
            )
            return []

        return episodic_memories

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
                    f"[{conv_id}] Error embedding query: {e}. Returning empty list."
                )
                return []
        hits = self.vec_store.search(
            q_embedding=query_embedding, top_k=top_k, collection_name=conv_id
        )
        results = [(h.content, h.score) for h in hits]
        if threshold is not None:
            results = [(c, s) for c, s in results if s > threshold]
        return results

    def reset(self, conv_id: str) -> None:
        self.vec_store.reset(conv_id)

    # Return True if it is merged in, False otherwise
    def try_merge_new_memory(
        self,
        new_memory: str,
        new_memory_embedding: List[float],
        threshold: float,
        conv_id: str,
    ) -> Tuple[bool, str]:
        # Search from the episodic memory
        hits = self.vec_store.search(
            q_embedding=new_memory_embedding, top_k=1, collection_name=conv_id
        )
        if len(hits) == 0:
            return False, ""
        if hits[0].score < threshold:
            return False, ""
        past_memory: str = hits[0].content
        past_memory_id: str = hits[0].id
        template = Template(EPISODIC_MERGE_PROMPT)
        merge_prompt = template.render(
            new_memory=new_memory,
            past_memory=past_memory,
        )

        try:
            _, response_json = self.llm_client.chat_completion_json(
                model=os.getenv("MODEL"),
                messages=[Message(role="system", content=merge_prompt)],
                temperature=0.0,
                monitor=self.monitor,
                op_type=OPType.EPISODIC_MERGE,
            )
        except ContentFilteredException as e:
            logger.warning(
                f"[{conv_id}] Content filtered during memory merge: {e}. Skipping merge."
            )
            return False, ""
        except EmptyResponseException as e:
            logger.warning(
                f"[{conv_id}] Empty response during memory merge: {e}. Skipping merge."
            )
            return False, ""
        except JSONParseException as e:
            logger.error(
                f"[{conv_id}] Error parsing memory merge response: {e}. "
                f"Response: {e.raw_content}. Skipping merge."
            )
            return False, ""
        except Exception as e:
            # OpenAI API errors (after all retries failed)
            logger.error(
                f"[{conv_id}] All retries failed for memory merge: {e}. Skipping merge."
            )
            return False, ""

        should_merge = response_json.get("should_merge", "no")
        if should_merge.lower() == "yes":
            merged_memory = response_json.get("merged_memory")
            if merged_memory is None or merged_memory == "":
                logger.warning(
                    f"[{conv_id}] Merge decision is 'yes' but no merged_memory provided. "
                    "Skipping merge."
                )
                return False, ""
            try:
                # Use merged memory to replace the past memory
                self.vec_store.remove([past_memory_id], conv_id)
                memory_id = str(uuid.uuid4())
                embedding = self.embedder.embed(merged_memory)
                self.vec_store.add(
                    embedding=embedding,
                    payload=merged_memory,
                    id=memory_id,
                    collection_name=conv_id,
                )
            except Exception as e:
                logger.warning(
                    f"[{conv_id}] Error updating vector store during merge: {e}. "
                    "Skipping merge."
                )
                return False, ""
        else:
            return False, ""

        return True, merged_memory
