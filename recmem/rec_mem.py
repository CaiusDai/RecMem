from recmem.embedding import Embedding, OpenAIEmbedding
from recmem.episodic_memory import EpisodicMemory
from recmem.semantic_memory import SemanticMemory
from recmem.subconscious_memory import SubconsciousMemory
from recmem.vector_store import QdrantStore, VectorStore
from typing import Callable, List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from jinja2 import Template
from recmem.prompt import (
    SEMANTIC_EXTRACTION_PROMPT,
    ANSWER_PROMPT_LOCOMO,
    SEMANTIC_EXTRACTION_PROMPT_ABLATION,
)
import json
import os
from recmem.llm import LLMClient, Message, OpenAIClient
from recmem.token_monitor import TokenMonitor, OPType
import logging
import uuid


VectorStoreFactory = Callable[[str, Optional[str]], VectorStore]
"""Factory signature: (layer_name, storage_path) -> VectorStore.

`layer_name` is one of "subconscious" / "episodic" / "semantic" so factories
can route layers to different stores or apply layer-specific config.
`storage_path` is the path resolved from RecMemConfig / env vars.
"""


def _default_qdrant_factory(layer_name: str, storage_path: Optional[str]) -> VectorStore:
    return QdrantStore(
        embedding_dim=1536,
        path=storage_path or "",
        exact_search=True,
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class RecMemConfig:
    # General Setting
    min_consolidation_cnt: int
    min_relevant_score: float
    merge_with_epi_thresh: float

    # Retrieve phase topk
    retrieve_raw_topk: int
    retrieve_epi_topk: int
    semantic_memory_topk: int
    semantic_memory_threshold: float

    # conv_limit: Sample the first n conversations for testing
    conv_limit: Optional[int] = None

    disable_semantic_refinement: bool = False

    # Dataset storage path
    semantic_store: Optional[str] = None
    subconscious_store: Optional[str] = None
    episodic_store: Optional[str] = None


def format_memories(memories: List[str], prefix: str = "-") -> str:
    if not memories:
        return "None"
    return "\n".join(f"{prefix} {mem}" for mem in memories)


@dataclass
class MemorySearchResult:
    """Three-way memory retrieval result.

    Each field is a list of (content, score) tuples, sorted by score desc as
    returned by the underlying vector store.
    """
    subconscious: List[Tuple[str, float]]
    episodic: List[Tuple[str, float]]
    semantic: List[Tuple[str, float]]


class RecMem:
    def __init__(
        self,
        config: RecMemConfig,
        embedder: Optional[Embedding] = None,
        llm_client: Optional[LLMClient] = None,
        vector_store_factory: Optional[VectorStoreFactory] = None,
    ):
        # Configure storage paths
        semantic_path = config.semantic_store or os.getenv("SEMANTIC_STORE")
        subconscious_path = config.subconscious_store or os.getenv("SUBCONSCIOUS_STORE")
        episodic_path = config.episodic_store or os.getenv("EPISODIC_STORE")
        logger.info(f"Semantic memory storage path: {semantic_path}")
        logger.info(f"Subconscious memory storage path: {subconscious_path}")
        logger.info(f"Episodic memory storage path: {episodic_path}")

        # Resolve provider defaults — single shared embedder / llm_client across all layers
        self.embedder: Embedding = embedder if embedder is not None else OpenAIEmbedding()
        self.llm_client: LLMClient = llm_client if llm_client is not None else OpenAIClient()
        factory: VectorStoreFactory = vector_store_factory or _default_qdrant_factory

        # Initialize memory layers, sharing the embedder and llm_client where applicable
        self.subconscious_memory = SubconsciousMemory(
            embedding_dim=1536,
            storage_path=subconscious_path,
            embedder=self.embedder,
            vector_store=factory("subconscious", subconscious_path),
        )
        self.min_consolidation_cnt = config.min_consolidation_cnt
        self.min_relevant_score = config.min_relevant_score

        # Top-K Settings
        self.retrieve_raw_topk = config.retrieve_raw_topk
        self.gating_raw_topk = (
            self.min_consolidation_cnt * 2
        )  # How many raw memories are retrieved for gate checking
        self.retrieve_epi_topk = config.retrieve_epi_topk
        self.merge_with_epi_thresh = config.merge_with_epi_thresh
        self.conv_limit = config.conv_limit
        self.token_monitor: Optional[TokenMonitor] = None

        # Semantic Memory Part
        self.disable_semantic_refinement = config.disable_semantic_refinement
        self.semantic_memory_topk = config.semantic_memory_topk
        self.semantic_memory_threshold = config.semantic_memory_threshold
        self.semantic_memory = SemanticMemory(
            embedding_dim=1536,
            storage_path=semantic_path,
            embedder=self.embedder,
            vector_store=factory("semantic", semantic_path),
        )

        # Episodic Memory
        self.episodic_mem = EpisodicMemory(
            path=episodic_path,
            embedder=self.embedder,
            llm_client=self.llm_client,
            vector_store=factory("episodic", episodic_path),
        )
        logger.info(
            f"Parameters: min_consolidation_cnt = {self.min_consolidation_cnt}, min_relevant_score = {self.min_relevant_score},\
            gating_raw_topk = {self.gating_raw_topk}\
            retrieve_raw_topk = {self.retrieve_raw_topk}, retrieve_epi_topk = {self.retrieve_epi_topk},\
            semantic_memory_topk = {self.semantic_memory_topk}, semantic_memory_threshold = {self.semantic_memory_threshold},\
            merge_with_epi_thresh = {self.merge_with_epi_thresh}"
        )

    def enable_token_monitoring(self) -> None:
        if self.token_monitor is not None:
            return
        self.token_monitor = TokenMonitor()
        self.episodic_mem.set_monitor(self.token_monitor)

    def search_memories(
        self,
        question: str,
        conv_id: str,
        question_embedding: Optional[List[float]] = None,
    ) -> MemorySearchResult:
        """Run the three-way retrieval (subconscious + episodic + semantic).

        Uses the configured top-k / threshold from RecMemConfig. If embedding
        is not provided, embedder will embed it.

        Raises if embedding the question fails. Callers that want defensive
        behavior should wrap the call in try/except.
        """
        if question_embedding is None:
            question_embedding = self.embedder.embed(question)
        sub_results = self.subconscious_memory.retrieve_memory(
            conv_id=conv_id,
            top_k=self.retrieve_raw_topk,
            query_embedding=question_embedding,
        )
        epi_results = self.episodic_mem.retrieve_memory(
            conv_id=conv_id,
            top_k=self.retrieve_epi_topk,
            query_embedding=question_embedding,
        )
        sem_results = self.semantic_memory.retrieve_memory(
            conv_id=conv_id,
            top_k=self.semantic_memory_topk,
            query_embedding=question_embedding,
            threshold=self.semantic_memory_threshold,
        )
        return MemorySearchResult(
            subconscious=sub_results,
            episodic=epi_results,
            semantic=sem_results,
        )

    def answer_question(
        self,
        question: str,
        conv_id: str,
        question_embedding: Optional[List[float]] = None,
        answer_prompt: Optional[str] = None,
    ) -> str:
        """Answer a question using three-way memory retrieval + LLM.

        `answer_prompt` is a Jinja2 template string with variables
        {{episodic_memories}}, {{raw_memories}}, {{semantic_memories}}, and
        {{question}}. If None, defaults to `ANSWER_PROMPT_LOCOMO`. Callers
        targeting a specific benchmark or a custom answering style should
        inject their own template here.
        """
        template_src = answer_prompt if answer_prompt is not None else ANSWER_PROMPT_LOCOMO

        try:
            results = self.search_memories(question, conv_id, question_embedding)
        except Exception as e:
            logger.error(
                f"[{conv_id}] Failed to embed question: {e}. "
                "Returning default error message."
            )
            return "Not Answerable due to embedding error"

        sub_mems: List[str] = [c for c, _ in results.subconscious]
        epi_mems: List[str] = [c for c, _ in results.episodic]
        sem_mems: List[str] = [c for c, _ in results.semantic]

        rendered = Template(template_src).render(
            episodic_memories=format_memories(epi_mems),
            semantic_memories=format_memories(sem_mems),
            raw_memories=format_memories(sub_mems),
            question=question,
        )
        try:
            response = self.llm_client.chat_completion(
                model=os.getenv("MODEL"),
                messages=[Message(role="system", content=rendered)],
                temperature=0.0,
                monitor=self.token_monitor,
                op_type=OPType.ANSWER,
            )
        except Exception as e:
            logger.error(
                f"[{conv_id}] Failed to answer question after retries: {e}. "
                "Returning default error message."
            )
            return "Not Answerable due to LLM provider error"

        return response.content.strip()

    def _consolidate_memory(
        self,
        raw_mem_ids: List[str],
        raw_mem_contents: List[str],
        raw_timestamps: List[str],
        new_mem: str,
        new_mem_timestamp: str,
        conv_id: str,
    ) -> None:
        """
        Consolidate raw memories into episodic memories, then run semantic refinement
        to extract new semantic memories.
        After this function, related raw memories will be deleted from the vector store.
        New memory will not be added into the vector store.
        New episodic memory and semantic memories will be generated.

        Args:
            raw_mem_ids (List[str]): List of raw,related memory ids
            raw_mem_contents (List[str]): List of raw,related memory contents
            raw_timestamps (List[str]): List of raw,related memory timestamps
            new_mem (str): New memory content
            new_mem_timestamp (str): New memory timestamp
            conv_id (str): Conversation id, used for conversation isolation.
        """
        if len(raw_mem_ids) != len(raw_mem_contents) or len(raw_mem_ids) != len(
            raw_timestamps
        ):
            raise ValueError(
                f"[{conv_id}] Length of raw_mem_ids, raw_mem_contents, and raw_timestamps must be \
                the same during memory consolidation. Got(lengths) Raw IDs: {len(raw_mem_ids)}, \
                Raw Contents: {len(raw_mem_contents)}, Raw Timestamps: {len(raw_timestamps)}"
            )
        # Step 1: Sort the memories based on timestamp
        sorted_memories = sorted(
            zip(raw_timestamps, raw_mem_contents), key=lambda x: x[0]
        )
        # Step 2: Concatenate the memories into a single string, sorted from oldest to newest
        # Format:
        # -[Message Timestamp]: {timestamp} [Message]: {content}
        # -<Message 2>
        # ....
        cat_raw_memories = (
            "\n".join([f"- {mem[1]}" for mem in sorted_memories]) + f"\n- {new_mem}"
        )

        episodic_memories: List[str] = self.episodic_mem.add(
            conversation=cat_raw_memories,
            conv_id=conv_id,
            raw_ids=raw_mem_ids,
        )
        if len(episodic_memories) == 0:
            logger.warning(
                f"[{conv_id}] Episodic memory generation failed. Skipping! No memory get deleted."
            )
            return
        # Step 3: Update Semantic Memory Store
        if not self.disable_semantic_refinement:
            for episodic_memory in episodic_memories:
                semantic_memories: List[str] = self.generate_semantic_memories(
                    episodic_memory,
                    cat_raw_memories,
                    is_new_episodic=True,
                    conv_id=conv_id,
                )
                if len(semantic_memories) == 0:
                    logger.warning(
                        f"[{conv_id}] Semantic memory generation failed.Skipping!(Memories may still get deleted.)"
                    )
                    continue
                extra_payload = {"source": f"{episodic_memory}"}
                self.semantic_memory.add_memories(
                    semantic_memories, conv_id, extra_payload=extra_payload
                )
        else:
            semantic_memories: List[str] = self.direct_semantic_extraction(
                raw_conv=cat_raw_memories,
                conv_id=conv_id,
            )
            if len(semantic_memories) == 0:
                logger.warning(
                    f"[{conv_id}] Semantic memory generation failed.Skipping!(Memories may still get deleted.)"
                )
            self.semantic_memory.add_memories(semantic_memories, conv_id)
        # Step 4: Remove the consolidated memories from subconscious memory store
        if self.token_monitor is not None:
            self.token_monitor.record_merging(len(raw_mem_ids))
        if len(raw_mem_ids) > 0:
            self.subconscious_memory.remove(raw_mem_ids, conv_id)

    def add_memory(
        self,
        message: str,
        timestamp: str,
        conv_id: str,
    ) -> None:
        if self.token_monitor is not None:
            self.token_monitor.record_message()
        conv = f"[Message Timestamp]: {timestamp} [Message]: {message}"
        conv_embedding = self.embedder.embed(conv)
        # Step 1: Search from the episodic memory, to see if related memories have been
        # consolidated before. If so, merge in episodes.
        decide_to_merge, new_episodic = self.episodic_mem.try_merge_new_memory(
            new_memory=conv,
            new_memory_embedding=conv_embedding,
            threshold=self.merge_with_epi_thresh,
            conv_id=conv_id,
        )
        if decide_to_merge:
            if self.token_monitor is not None:
                self.token_monitor.record_merging(1)
            if not self.disable_semantic_refinement:
                semantic_memories: List[str] = self.generate_semantic_memories(
                    episodic_memory=new_episodic,
                    raw_conv=conv,
                    is_new_episodic=False,
                    conv_id=conv_id,
                )
            else:
                semantic_memories: List[str] = self.direct_semantic_extraction(
                    raw_conv=conv,
                    conv_id=conv_id,
                )
            self.semantic_memory.add_memories(
                semantic_memories,
                conv_id,
                extra_payload={"source": f"{conv}-{new_episodic}"},
            )
            return

        # Step 2: Related topic has not been discussed before. Trigger gating mechanism
        # to see if we've accumulated enough experience of our new information
        hits = self.subconscious_memory.search(
            q_embedding=conv_embedding,
            top_k=self.gating_raw_topk,
            conv_id=conv_id,
        )
        relevant_vec_ids: List[str] = []
        relevant_vec_contents: List[str] = []
        relevant_vec_timestamps: List[str] = []
        for hit in hits:
            if hit.score > self.min_relevant_score:
                relevant_vec_ids.append(hit.id)
                relevant_vec_contents.append(hit.content)
                relevant_vec_timestamps.append(hit.extra_payload["timestamp"])
        # Update counters for all similar vectors and new memory
        relevant_count: int = len(relevant_vec_ids) + 1

        if relevant_count >= self.min_consolidation_cnt:
            self._consolidate_memory(
                raw_mem_ids=relevant_vec_ids,
                raw_mem_contents=relevant_vec_contents,
                raw_timestamps=relevant_vec_timestamps,
                new_mem=conv,
                new_mem_timestamp=timestamp,
                conv_id=conv_id,
            )
        else:
            # Step 3: Add the new memory to the vector store
            memory_id = str(uuid.uuid4())
            self.subconscious_memory.add(
                embedding=conv_embedding,
                payload=conv,
                id=memory_id,
                conv_id=conv_id,
                extra_payload={"timestamp": timestamp},
            )

    def generate_semantic_memories(
        self,
        episodic_memory: str,
        raw_conv: str,
        conv_id: str,
        is_new_episodic: bool,
    ) -> List[str]:
        if self.disable_semantic_refinement:
            raise ValueError(
                "Semantic refinement is not enabled. Use direct semantic extraction instead."
            )
        # Step 1 : Try to get the facts in old semantic memories
        relevant_results = self.semantic_memory.retrieve_memory(
            conv_id=conv_id,
            top_k=self.semantic_memory_topk,
            query=episodic_memory,
            threshold=self.semantic_memory_threshold,
        )
        relevant_facts: List[str] = [m for m, _ in relevant_results]
        # Step 2: Try to distill more related facts about this episodic memory
        # Need to let episodic memory also provide the raw reference back
        template = Template(SEMANTIC_EXTRACTION_PROMPT)
        op_type = (
            OPType.SEMANTIC_EXTRACTION
            if is_new_episodic
            else OPType.SEMANTIC_EXTRACTION_DURING_MERGE
        )
        extraction_prompt = template.render(
            episodic_memory=episodic_memory,
            raw_reference=raw_conv,
            old_semantic_memories=relevant_facts,
        )

        try:
            _, response_json = self.llm_client.chat_completion_json(
                model=os.getenv("MODEL"),
                messages=[Message(role="system", content=extraction_prompt)],
                temperature=0.0,
                monitor=self.token_monitor,
                op_type=op_type,
            )
        except Exception as e:
            logger.error(
                f"[{conv_id}] Failed to generate semantic memories after retries: {e}. "
                "Returning empty list."
            )
            return []

        facts: List[str] = response_json.get("facts", [])
        if len(facts) == 0:
            logger.error(
                f"[{conv_id}] No 'facts' field in semantic extraction response. "
                f"Response: {response_json}. Returning empty list."
            )
            return []
        return facts

    def direct_semantic_extraction(
        self,
        raw_conv: str,
        conv_id: str,
    ) -> List[str]:
        template = Template(SEMANTIC_EXTRACTION_PROMPT_ABLATION)
        extraction_prompt = template.render(
            raw_reference=raw_conv,
        )
        try:
            _, response_json = self.llm_client.chat_completion_json(
                model=os.getenv("MODEL"),
                messages=[Message(role="system", content=extraction_prompt)],
                temperature=0.0,
                monitor=self.token_monitor,
                op_type=OPType.SEMANTIC_EXTRACTION,
            )
        except Exception as e:
            logger.error(
                f"[{conv_id}] Failed to generate semantic memories after retries: {e}. "
                "Returning empty list."
            )
            return []
        facts: List[str] = response_json.get("facts", [])
        if len(facts) == 0:
            logger.error(
                f"[{conv_id}] No 'facts' field in semantic extraction response. "
                f"Response: {response_json}. Returning empty list."
            )
            return []
        return facts

    def reset(self, conv_id: str) -> None:
        self.subconscious_memory.reset(conv_id)
        self.episodic_mem.reset(conv_id)
        self.semantic_memory.reset(conv_id)
