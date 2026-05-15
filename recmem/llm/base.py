import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from json_repair import repair_json

if TYPE_CHECKING:
    from recmem.token_monitor import OPType, TokenMonitor, TokenUsage

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single chat message in a normalized, provider-neutral shape."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    """Normalized chat completion response, decoupled from any provider's SDK types."""

    content: str
    finish_reason: str  # "stop" | "length" | "content_filter" | "tool_calls" | ...
    usage: Optional["TokenUsage"]


class ContentFilteredException(Exception):
    """Raised when content is filtered by provider moderation."""

    def __init__(
        self,
        message: str = "Content was filtered by provider moderation",
        finish_reason: str = "content_filter",
    ):
        super().__init__(message)
        self.finish_reason = finish_reason


class EmptyResponseException(Exception):
    """Raised when response content is empty or missing."""

    def __init__(
        self,
        message: str = "Response content is empty",
        finish_reason: Optional[str] = None,
    ):
        super().__init__(message)
        self.finish_reason = finish_reason


class JSONParseException(Exception):
    """Raised when JSON parsing of a response fails (even after repair)."""

    def __init__(self, message: str, raw_content: str):
        super().__init__(message)
        self.raw_content = raw_content


class LLMClient(ABC):
    """Abstract base class for chat-completion providers.

    Implementations map this normalized API to and from a vendor-specific SDK.
    Callers always work with `Message` / `LLMResponse` / `TokenUsage` and never
    see provider-native types.
    """

    @abstractmethod
    def chat_completion(
        self,
        *,
        model: str,
        messages: List[Message],
        temperature: float = 0.0,
        json_mode: bool = False,
        monitor: Optional["TokenMonitor"] = None,
        op_type: Optional["OPType"] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> LLMResponse:
        """Run a chat completion and return a normalized response.

        If `monitor` is provided and the response carries usage information, the
        implementation MUST call `monitor.record_usage(usage, op_type)`.

        Implementations MUST raise `ContentFilteredException` /
        `EmptyResponseException` for the corresponding failure modes.
        """
        ...

    def chat_completion_json(
        self,
        *,
        model: str,
        messages: List[Message],
        temperature: float = 0.0,
        monitor: Optional["TokenMonitor"] = None,
        op_type: Optional["OPType"] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Tuple[LLMResponse, Dict[str, Any]]:
        """Run a chat completion in JSON mode and return (response, parsed_dict).

        Default implementation calls `chat_completion(json_mode=True)` then
        parses with `json_repair` fallback. Subclasses may override to use a
        provider-native JSON tool.
        """
        response = self.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            json_mode=True,
            monitor=monitor,
            op_type=op_type,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            try:
                repaired = repair_json(response.content, return_objects=True)
                logger.warning(
                    f"Fixing Broken JSON: {response.content}\nRepaired JSON: {repaired}"
                )
                if not isinstance(repaired, dict):
                    raise JSONParseException(
                        f"Repaired JSON is not a dict, got {type(repaired).__name__}",
                        raw_content=response.content,
                    )
                parsed = repaired
            except JSONParseException:
                raise
            except Exception as e:
                raise JSONParseException(
                    f"Failed to parse/repair JSON: {e}",
                    raw_content=response.content,
                )
        return response, parsed
