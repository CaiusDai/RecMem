import json
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


@dataclass
class TokenUsage:
    """Provider-neutral token usage report for a single LLM call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OPType(Enum):
    """
    Operation types that will consume tokens.
    """

    EPISODIC_GENERATION = 1
    ITERATIVE_FILTER = 2
    ANSWER = 3
    SEMANTIC_EXTRACTION = 4
    EPISODIC_MERGE = 5
    SEMANTIC_EXTRACTION_DURING_MERGE = 6


# Note: Thread safe class
class TokenMonitor:
    def __init__(self) -> None:
        self.total_tokens_used = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # Total messages passed into SAMem
        self.total_messages = 0
        # How many messages are removed from subconscious store
        self.total_merged_messages = 0
        self.total_questions = 0
        self.total_op_tokens: Dict[OPType, Dict[str, int]] = {
            op_type: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            for op_type in OPType
        }
        self._lock = threading.Lock()

    def record_message(self, count: int = 1) -> None:
        with self._lock:
            self.total_messages += count

    def record_merging(self, raw_count: int) -> None:
        with self._lock:
            self.total_merged_messages += raw_count

    def record_usage(
        self,
        usage: TokenUsage,
        op_type: Optional[OPType] = None,
    ) -> None:
        """Record a normalized token usage report. Thread-safe."""
        with self._lock:
            self.total_tokens_used += usage.total_tokens
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            if op_type is not None:
                op_totals = self.total_op_tokens[op_type]
                op_totals["prompt_tokens"] += usage.prompt_tokens
                op_totals["completion_tokens"] += usage.completion_tokens
                op_totals["total_tokens"] += usage.total_tokens

    def record_usage_from_dict(
        self,
        tokens: Dict[str, int],
        op_type: Optional[OPType] = None,
    ) -> None:
        """Convenience wrapper for callers that already have a token-count dict."""
        prompt_tokens = tokens.get("prompt_tokens", 0)
        completion_tokens = tokens.get("completion_tokens", 0)
        total_tokens = tokens.get("total_tokens", prompt_tokens + completion_tokens)
        self.record_usage(
            TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            op_type=op_type,
        )

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            op_tokens_serializable = {
                op_type.name: {
                    "prompt_tokens": tokens["prompt_tokens"],
                    "completion_tokens": tokens["completion_tokens"],
                    "total_tokens": tokens["total_tokens"],
                    "percentage": (
                        (tokens["total_tokens"] / self.total_tokens_used * 100)
                        if self.total_tokens_used > 0
                        else 0.0
                    ),
                }
                for op_type, tokens in self.total_op_tokens.items()
            }

            average_message_save_ratio = (
                self.total_merged_messages / self.total_messages
                if self.total_messages > 0
                else 0
            )

            return {
                "total_tokens_used": self.total_tokens_used,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_messages": self.total_messages,
                "total_merged_messages": self.total_merged_messages,
                "total_questions": self.total_questions,
                "average_message_save_ratio": average_message_save_ratio,
                "op_tokens": op_tokens_serializable,
            }

    def save_stats(self, file_path: str) -> None:
        stats = self.get_summary()
        with open(file_path, "w") as f:
            json.dump(stats, f, indent=4)

    def print_summary(self) -> None:
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("TOKEN USAGE SUMMARY")
        print("=" * 60)
        print(f"Total tokens used: {summary['total_tokens_used']:,}")
        print(f"Prompt tokens: {summary['total_prompt_tokens']:,}")
        print(f"Completion tokens: {summary['total_completion_tokens']:,}")
        print(f"Total messages recorded: {summary['total_messages']:,}")
        print(f"Total questions answered: {summary['total_questions']:,}")
        print(f"Total merged messages: {summary['total_merged_messages']:,}")
        print(f"Message save ratio: {summary['average_message_save_ratio'] * 100:.2f}%")

        # Print operation-specific token usage
        if "op_tokens" in summary and summary["op_tokens"]:
            print("\n" + "-" * 60)
            print("TOKEN USAGE BY OPERATION TYPE")
            print("-" * 60)

            # Sort by total tokens (descending)
            sorted_ops = sorted(
                summary["op_tokens"].items(),
                key=lambda x: x[1]["total_tokens"],
                reverse=True,
            )

            for op_name, op_stats in sorted_ops:
                total = op_stats["total_tokens"]
                percentage = op_stats["percentage"]
                prompt = op_stats["prompt_tokens"]
                completion = op_stats["completion_tokens"]

                # Format operation name for better readability
                formatted_name = op_name.replace("_", " ").title()

                if total > 0:
                    print(f"\n{formatted_name}:")
                    print(f"  Total tokens: {total:,} ({percentage:.2f}%)")
                    print(f"  Prompt tokens: {prompt:,}")
                    print(f"  Completion tokens: {completion:,}")
                else:
                    print(f"\n{formatted_name}: 0 tokens (0.00%)")

        print("=" * 60)
