# type: ignore
import argparse
import os
from dotenv import load_dotenv
from recmem.embedding import OpenAIEmbedding
from recmem.llm import OpenAIClient
from recmem.prompt import ANSWER_PROMPT_LME, ANSWER_PROMPT_LOCOMO
from recmem.rec_mem import RecMem, RecMemConfig
from recmem.vector_store import QdrantStore, VectorStore
import concurrent.futures
import json
import threading
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import time
from evaluation.metrics.llm_judge import evaluate_llm_judge, evaluate_llm_judge_lme
from evaluation.metrics.utils import calculate_bleu_scores, calculate_metrics
from tqdm import tqdm
from evaluation.conv_loader import ConvLoader, Conversation, Question
import logging
import pandas as pd
import random


def make_qdrant_store(layer_name: str, storage_path: Optional[str]) -> VectorStore:
    """Build the local Qdrant-backed `VectorStore` used for one memory layer.

    `layer_name` is one of "subconscious" / "episodic" / "semantic" — kept here
    so external users can route different layers to different backends if they
    want to. Swap the body of this function (or replace it entirely) to plug in
    Chroma, Pinecone, Weaviate, etc.
    """
    return QdrantStore(
        embedding_dim=1536,
        path=storage_path or "",
        exact_search=True,
    )

# Configure logging to output to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
for logger_name in ["httpx", "urllib3", "openai", "httpcore", "requests"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


load_dotenv()


def add_arguments():
    parser = argparse.ArgumentParser(description="Run memory experiments")
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of top memories to retrieve"
    )
    parser.add_argument(
        "--min_consolidation_cnt",
        type=int,
        default=6,
        help="Recurrence Count Threshold",
    )
    parser.add_argument(
        "--min_relevant_score",
        type=float,
        default=0.7,
        help="Similarity Threshold",
    )
    parser.add_argument(
        "--retrieve_raw_topk",
        type=int,
        default=10,
        help="Number of subconscious memories to retrieve",
    )
    parser.add_argument(
        "--retrieve_epi_topk",
        type=int,
        default=5,
        help="Number of episodic memories to retrieve",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="Model to use",
    )
    parser.add_argument(
        "--merge_with_epi_thresh",
        type=float,
        default=0.75,
        help="Threshold to merge new memory with episodic memory",
    )
    parser.add_argument(
        "--enable_stat",
        action="store_true",
        help="Whether to enable token monitoring and saving ratio counting",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Whether to evaluate only",
    )
    parser.add_argument(
        "--conv_limit",
        type=int,
        default=None,
        help="Number of conversations to sample for testing",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="[LME only] Number of questions to sample per question type. "
        "Since LME has one question per conversation, this samples sample_size "
        "conversations from each of the 6 question types (0-5).",
    )
    parser.add_argument(
        "--disable_semantic_refinement",
        action="store_true",
        help="Whether to disable semantic refinement",
    )
    parser.add_argument(
        "--semantic_memory_topk",
        type=int,
        default=10,
        help="Number of top memories to retrieve from semantic memory",
    )
    parser.add_argument(
        "--semantic_memory_threshold",
        type=float,
        default=0,
        help="Threshold to consider a memory as semantic memory",
    )
    parser.add_argument(
        "--question_max_workers",
        type=int,
        default=5,
        help="Maximum number of parallel workers for answering questions",
    )
    parser.add_argument(
        "--conv_workers",
        type=int,
        default=20,
        help="Maximum number of parallel workers for processing conversations",
    )
    parser.add_argument(
        "--search_only",
        action="store_true",
        help="Whether to search only",
    )
    parser.add_argument(
        "--semantic_store",
        type=str,
        default=None,
        help="Path to the semantic store",
    )
    parser.add_argument(
        "--subconscious_store",
        type=str,
        default=None,
        help="Path to the subconscious store",
    )
    parser.add_argument(
        "--episodic_store",
        type=str,
        default=None,
        help="Path to the episodic store",
    )
    parser.add_argument(
        "--bench",
        type=str,
        required=True,
        choices=["locomo", "longmemeval_s"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4o-mini",
        help="Model used by the LLM judge during evaluation",
    )
    return parser.parse_args()


def process_item(item_data, bench: str, judge_model: str):
    k, v = item_data
    local_results = defaultdict(list)

    for item in v:
        gt_answer = str(item["answer"])
        pred_answer = str(item["response"])
        category = int(item["category"])
        question = str(item["question"])

        if bench == "locomo":
            # Skip category 5
            if category == 5:
                continue
            llm_score = evaluate_llm_judge(
                question, gt_answer, pred_answer, model=judge_model
            )
        else:
            answerable = item["answerable"]
            llm_score = evaluate_llm_judge_lme(
                question, category, gt_answer, pred_answer, answerable,
                model=judge_model,
            )

        metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)

        local_results[k].append(
            {
                "question": question,
                "answer": gt_answer,
                "response": pred_answer,
                "category": category,
                "bleu_score": bleu_scores["bleu1"],
                "f1_score": metrics["f1"],
                "llm_score": llm_score,
            }
        )

    return local_results


def generate_eval_summary(eval_path: str, save_path: str, category_mapping: dict):
    """
    Generate evaluation scores from evaluation metrics data and save results to file.

    Args:
        eval_path: Path to the evaluation metrics JSON file
        save_path: Path to save the results (without extension, will save as .txt and .json)
    """
    logger.info(f"Loading evaluation metrics from {eval_path}")

    # Load the evaluation metrics data
    with open(eval_path, "r") as f:
        data = json.load(f)

    # Flatten the data into a list of question items
    all_items = []
    for key in data:
        all_items.extend(data[key])

    # Convert to DataFrame
    df = pd.DataFrame(all_items)

    # Convert category to numeric type
    df["category"] = pd.to_numeric(df["category"])

    # Calculate mean scores by category
    result = (
        df.groupby("category")
        .agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"})
        .round(4)
    )

    # Add count of questions per category
    result["count"] = df.groupby("category").size()

    # Create a version with category names for display
    result_with_names = result.copy()
    result_with_names.index = result_with_names.index.map(
        lambda x: f"{category_mapping.get(x, f'unknown_{x}')} ({x})"
    )

    # Calculate overall means
    overall_means = df.agg(
        {"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}
    ).round(4)

    # Prepare results for saving - use original numeric categories for JSON
    results_dict = {
        "category_scores": result.to_dict(),
        "category_mapping": category_mapping,
        "overall_scores": overall_means.to_dict(),
        "total_questions": len(all_items),
    }

    # Create save directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save results as JSON
    json_path = save_path + ".json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    # Save results as human-readable text
    txt_path = save_path + ".txt"
    with open(txt_path, "w") as f:
        f.write("Evaluation Results\n")
        f.write("==================\n\n")

        f.write("Mean Scores Per Category:\n")
        f.write("-" * 60 + "\n")
        f.write(result_with_names.to_string())
        f.write("\n\n")

        f.write("Overall Mean Scores:\n")
        f.write("-" * 20 + "\n")
        for metric, score in overall_means.items():
            f.write(f"{metric}: {score}\n")
        f.write(f"\nTotal Questions: {len(all_items)}\n")

        f.write("\nCategory Mapping:\n")
        f.write("-" * 20 + "\n")
        for num, name in category_mapping.items():
            f.write(f"{num}: {name}\n")

    # Also print to console
    print("Mean Scores Per Category:")
    print(result_with_names)
    print("\nOverall Mean Scores:")
    print(overall_means)

    logger.info(f"Results saved to {json_path} and {txt_path}")

    return results_dict


def eval_experiment(input_file: str, output_file: str, bench: str, judge_model: str):
    logger.info(f"Using judge model: {judge_model}")
    with open(input_file, "r") as f:
        data = json.load(f)

    results = defaultdict(list)
    results_lock = threading.Lock()

    # Use ThreadPoolExecutor with specified workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = [
            executor.submit(process_item, item_data, bench, judge_model)
            for item_data in data.items()
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            mininterval=10,
        ):
            local_results = future.result()
            with results_lock:
                for k, items in local_results.items():
                    results[k].extend(items)

    # Save results to JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")

    summary_path = output_file[:-5] + "_summary"
    if bench == "locomo":
        category_mapping = {
            1: "multihop",
            2: "temporal",
            3: "open_domain",
            4: "singlehop",
            5: "adversarial",
        }
    elif bench == "longmemeval_s":
        category_mapping = {
            0: "single-session-user",
            1: "single-session-assistant",
            2: "single-session-preference",
            3: "temporal-reasoning",
            4: "knowledge-update",
            5: "multi-session",
        }
    else:
        raise ValueError(f"Invalid benchmark name: {bench}")

    generate_eval_summary(output_file, summary_path, category_mapping)


def process_one_conversation(
    rec_mem: RecMem,
    conv_idx: int,
    chat_history: Conversation,
    conv_questions: List[Question],
    max_workers: int,
    search_only: bool,
    bench: str,
) -> List[Dict]:
    """
    Process a single conversation: add memories and answer questions.

    Args:
        rec_mem: RecMem instance to use
        conv_idx: Conversation index
        chat_history: The conversation data
        conv_questions: List of questions for this conversation
        max_workers: Maximum number of parallel workers for answering questions

    Returns:
        List of results containing questions, answers, and responses
    """
    if bench == "locomo":
        conv_id = f"locomo_conv_{conv_idx}"
    elif bench == "longmemeval_s":
        conv_id = f"longmemeval_s_conv_{conv_idx}"
    else:
        raise ValueError(f"Invalid benchmark name: {bench}")
    if not search_only:
        messages: List[str] = []
        timestamps: List[str] = []

        # Formulate messages and timestamps.
        for session in chat_history.sessions:
            if bench == "locomo":
                timestamp = ConvLoader.locomo_time_to_datetime(
                    session.date_time
                ).isoformat()
                # Locomo: dual-user dialogue, simple pairwise grouping is fine
                for i in range(0, len(session.messages), 2):
                    message_a = session.messages[i]
                    if i + 1 < len(session.messages):
                        message_b = session.messages[i + 1]
                        messages.append(
                            message_a.to_string()
                            + "\n"
                            + message_b.to_string()
                        )
                    else:
                        messages.append(message_a.to_string())
                    timestamps.append(timestamp)
            else:
                # LongMemEval: user-assistant dialogue
                # Group messages as: user+assistant pairs, or standalone assistant messages
                # Avoid grouping assistant+user which would split a dialogue turn
                timestamp = session.date_time
                i = 0
                while i < len(session.messages):
                    message_a = session.messages[i]
                    if message_a.speaker == "user" and i + 1 < len(session.messages):
                        # User message followed by another message - check if it's assistant
                        message_b = session.messages[i + 1]
                        if message_b.speaker == "assistant":
                            # Complete dialogue turn: user + assistant
                            messages.append(
                                message_a.to_string()
                                + "\n"
                                + message_b.to_string()
                            )
                            timestamps.append(timestamp)
                            i += 2
                            continue
                    # Standalone message (assistant, or user not followed by assistant)
                    messages.append(message_a.to_string())
                    timestamps.append(timestamp)
                    i += 1

        # Initialize conversation
        rec_mem.reset(conv_id)
        start_time = time.time()
        # Adding memory phase (must be sequential)
        for i in tqdm(
            range(len(messages)),
            desc=f"Adding memories (conv {conv_idx})",
            leave=False,
            disable=True,
        ):
            rec_mem.add_memory(messages[i], timestamps[i], conv_id)
            if i % 20 == 0 and i > 0:
                pass_time = time.time() - start_time
                eto = pass_time / i * len(messages)
                time_left = eto - pass_time
                time_left_str = f"{time_left // 60}m{(time_left % 60):.2f}s"
                logger.info(
                    f"[{conv_idx}] Finished {i / len(messages) * 100 :.2f}%, Time left to add memories: {time_left_str}"
                )
        end_time = time.time()
        logger.info(
            f"[{conv_idx}] Time taken to add memories: {(end_time - start_time) // 60}m{(end_time - start_time) % 60:.2f}s"
        )

    # Pick the answer prompt by bench explicitly — no implicit defaults, so a
    # reader of this file knows exactly which template each bench uses.
    if bench == "locomo":
        answer_prompt = ANSWER_PROMPT_LOCOMO
    elif bench == "longmemeval_s":
        answer_prompt = ANSWER_PROMPT_LME
    else:
        raise ValueError(f"Invalid benchmark name: {bench}")

    # Retrieving memory phase (parallel processing)
    def process_question(q_idx_item):
        """Process a single question and return result with index."""
        q_idx, item = q_idx_item
        answerable = item.answerable
        question: str = item.content
        answer: str = item.answer
        category: int = item.category
        if item.date is not None:
            question = question + f"\nThe date of the question is {item.date}"
        response: str = rec_mem.answer_question(
            question, conv_id, answer_prompt=answer_prompt
        )

        return {
            "index": q_idx,
            "question": question,
            "answer": answer,
            "category": category,
            "response": response,
            "answerable": answerable,
        }

    # Pre-allocate results list to maintain order
    results = [None] * len(conv_questions)

    # Use ThreadPoolExecutor for parallel question answering
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_question, (q_idx, item))
            for q_idx, item in enumerate(conv_questions)
        ]

        # Collect results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            desc=f"Answering questions (conv {conv_idx})",
            leave=False,
            total=len(futures),
            mininterval=5,
            maxinterval=10,
        ):
            result = future.result()
            idx = result.pop("index")
            results[idx] = result

    return results


def sample_lme_by_question_type(
    convs: List[Conversation],
    questions: List[List[Question]],
    sample_size: int,
    seed: int = 42,
) -> Tuple[List[Conversation], List[List[Question]]]:
    """
    Sample LME conversations by question type.

    For LongMemEval, each conversation has exactly one question. This function
    samples `sample_size` questions from each question type (category 0-5),
    resulting in up to 6 * sample_size conversations total.

    Args:
        convs: List of conversations
        questions: List of questions (each is a list with one Question for LME)
        sample_size: Number of questions to sample per question type
        seed: Random seed for reproducibility

    Returns:
        Tuple of (sampled_convs, sampled_questions)
    """
    random.seed(seed)

    # Group conversations by question category
    # For LME, each conversation has exactly one question
    category_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, q_list in enumerate(questions):
        if q_list:  # Should always have exactly one question for LME
            category = q_list[0].category
            category_to_indices[category].append(idx)

    # Sample from each category
    sampled_indices = []
    for category in sorted(category_to_indices.keys()):
        indices = category_to_indices[category]
        if len(indices) <= sample_size:
            # Take all if not enough samples
            sampled_indices.extend(indices)
            logger.info(
                f"Category {category}: only {len(indices)} samples available, "
                f"taking all (requested {sample_size})"
            )
        else:
            # Random sample
            sampled = random.sample(indices, sample_size)
            sampled_indices.extend(sampled)
            logger.info(
                f"Category {category}: sampled {sample_size} from {len(indices)}"
            )

    # Sort indices to maintain some order
    sampled_indices.sort()

    # Build sampled lists
    sampled_convs = [convs[i] for i in sampled_indices]
    sampled_questions = [questions[i] for i in sampled_indices]

    logger.info(f"Total sampled: {len(sampled_convs)} conversations")
    return sampled_convs, sampled_questions


def process_all_conversations(
    rec_mem: RecMem,
    data_path: str,
    output_file_path: str,
    bench: str,
    conv_workers: int,
    conv_limit: int = None,
    sample_size: int = None,
    question_max_workers: int = 5,
    search_only: bool = False,
) -> None:
    """
    Process all conversations in the dataset.

    Args:
        rec_mem: RecMem instance to use
        data_path: Path to the dataset
        output_file_path: Path to save results
        bench: Benchmark name.  [locomo, longmemeval_s]
        conv_workers: Maximum number of parallel workers for processing conversations
        conv_limit: Optional limit on number of conversations to process
        sample_size: [LME only] Number of questions to sample per question type
        question_max_workers: Maximum number of parallel workers for answering questions
    """
    if bench == "locomo":
        convs, questions = ConvLoader.load_locomo(data_path)
    elif bench == "longmemeval_s":
        convs, questions = ConvLoader.load_longmemeval_s(data_path)
    else:
        raise ValueError(f"Invalid benchmark name: {bench}")
    logger.info(f"Benchmarking {bench} dataset, loading from {data_path}")

    # Apply sample_size for LME (takes precedence over conv_limit for LME)
    if sample_size is not None:
        if bench == "longmemeval_s":
            convs, questions = sample_lme_by_question_type(
                convs, questions, sample_size
            )
            logger.info(f"Sampled {sample_size} questions per type for LME")
        else:
            logger.warning(
                f"sample_size is only supported for longmemeval_s, ignoring for {bench}"
            )
            # Fall through to conv_limit if specified
            if conv_limit is not None:
                convs = convs[:conv_limit]
                questions = questions[:conv_limit]
                logger.info(f"Limited to {conv_limit} conversations")
    elif conv_limit is not None:
        convs = convs[:conv_limit]
        questions = questions[:conv_limit]
        logger.info(f"Limited to {conv_limit} conversations")
    # convs = convs[-10:]
    # questions = questions[-10:]
    FINAL_RESULTS = {}
    results_lock = threading.Lock()
    stats_file_path = output_file_path.replace(".json", "_token_stats.json")
    os.makedirs(os.path.dirname(stats_file_path), exist_ok=True)

    # For parallel processing
    def process_conv_wrapper(conv_data):
        conv_idx, chat_history, conv_questions = conv_data
        results = process_one_conversation(
            rec_mem=rec_mem,
            conv_idx=conv_idx,
            chat_history=chat_history,
            conv_questions=conv_questions,
            max_workers=question_max_workers,
            search_only=search_only,
            bench=bench,
        )
        return conv_idx, results

    with concurrent.futures.ThreadPoolExecutor(max_workers=conv_workers) as executor:
        conv_data_list = [
            (conv_idx, convs[conv_idx], questions[conv_idx])
            for conv_idx in range(len(convs))
        ]

        futures = [
            executor.submit(process_conv_wrapper, conv_data)
            for conv_data in conv_data_list
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            desc="Processing conversations",
            total=len(futures),
            mininterval=5,
            maxinterval=10,
        ):
            conv_idx, results = future.result()
            with results_lock:
                FINAL_RESULTS[str(conv_idx)] = results

    # Write all results
    logger.info(f"Writing results to {output_file_path}")
    with open(output_file_path, "w") as f:
        json.dump(FINAL_RESULTS, f, indent=4)

    # Save token stats if monitoring is enabled
    if rec_mem.token_monitor is not None:
        rec_mem.token_monitor.save_stats(stats_file_path)
        logger.info(f"Token statistics saved to {stats_file_path}")


def main():
    args = add_arguments()
    # Configure path related configurations
    bench = args.bench
    if bench == "locomo":
        score_output_dir = os.getenv("LOCOMO_SCORE_PATH")
        data_path = os.getenv("LOCOMO_PATH")
    elif bench == "longmemeval_s":
        score_output_dir = os.getenv("LONGMEMEVAL_S_SCORE_PATH")
        data_path = os.getenv("LONGMEMEVAL_S_PATH")
    else:
        raise ValueError(f"Invalid benchmark name: {bench}")
    if args.eval_only:
        answer_file_name = args.output_file.split("/")[-1]
        score_file_name = score_output_dir + "/" + answer_file_name
        logger.info(
            f"Evaluating experiment. Source file: {args.output_file}, Score file: {score_file_name}"
        )
        eval_experiment(args.output_file, score_file_name, bench, args.judge_model)
        return

    os.environ["MODEL"] = args.model
    logger.info(f"Using model: {os.getenv('MODEL')}")
    logger.info(f"Enable token monitoring: {args.enable_stat}")
    logger.info(f"Disable semantic refinement: {args.disable_semantic_refinement}")
    logger.info(f"Question max workers: {args.question_max_workers}")
    logger.info(f"Search only: {args.search_only}")
    if args.sample_size is not None:
        logger.info(f"Sample size per question type (LME only): {args.sample_size}")

    cfg = RecMemConfig(
        min_consolidation_cnt=args.min_consolidation_cnt,
        min_relevant_score=args.min_relevant_score,
        retrieve_raw_topk=args.retrieve_raw_topk,
        retrieve_epi_topk=args.retrieve_epi_topk,
        merge_with_epi_thresh=args.merge_with_epi_thresh,
        conv_limit=args.conv_limit,
        disable_semantic_refinement=args.disable_semantic_refinement,
        semantic_memory_topk=args.semantic_memory_topk,
        semantic_memory_threshold=args.semantic_memory_threshold,
        semantic_store=args.semantic_store,
        subconscious_store=args.subconscious_store,
        episodic_store=args.episodic_store,
    )

    # Provider injection — explicit so users can see (and swap) what backs each layer.
    # Replace any of these three with your own `Embedding` / `LLMClient` /
    # `VectorStoreFactory` to change providers without touching the library code.
    embedder = OpenAIEmbedding()
    llm_client = OpenAIClient()
    rec_mem = RecMem(
        config=cfg,
        embedder=embedder,
        llm_client=llm_client,
        vector_store_factory=make_qdrant_store,
    )

    if args.enable_stat:
        rec_mem.enable_token_monitoring()

    process_all_conversations(
        rec_mem=rec_mem,
        data_path=data_path,
        output_file_path=args.output_file,
        conv_workers=args.conv_workers,
        conv_limit=args.conv_limit,
        sample_size=args.sample_size,
        question_max_workers=args.question_max_workers,
        search_only=args.search_only,
        bench=bench,
    )

    # Eval section
    answer_file_name = args.output_file.split("/")[-1]
    score_file_name = score_output_dir + "/" + answer_file_name
    logger.info(
        f"Evaluating experiment. Source file: {args.output_file}, Score file: {score_file_name}"
    )
    eval_experiment(args.output_file, score_file_name, bench, args.judge_model)


if __name__ == "__main__":
    main()
