import os
import logging
import time
from typing import List, Optional

from openai import OpenAI

from recmem.llm.base import (
    ContentFilteredException,
    EmptyResponseException,
    LLMClient,
    LLMResponse,
    Message,
)
from recmem.token_monitor import OPType, TokenMonitor, TokenUsage

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """OpenAI implementation of `LLMClient`.

    Maps the normalized `Message` / `LLMResponse` shape to and from the OpenAI
    Python SDK. Token usage is reported via `TokenMonitor.record_usage` using a
    `TokenUsage` dataclass; the OpenAI native types do not leak past this file.

    Native exception types from `openai.*Error` are not wrapped and will surface
    to the caller as-is for retryable network / auth errors.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        resolved_base_url = (
            base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
        )
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=resolved_base_url,
        )

    def chat_completion(
        self,
        *,
        model: str,
        messages: List[Message],
        temperature: float = 0.0,
        json_mode: bool = False,
        monitor: Optional[TokenMonitor] = None,
        op_type: Optional[OPType] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> LLMResponse:
        if max_retries <= 0:
            raise ValueError("max_retries must be greater than 0")
        if retry_delay <= 0:
            raise ValueError("retry_delay must be greater than 0")

        openai_messages = [{"role": m.role, "content": m.content} for m in messages]
        kwargs = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(retry_delay * (attempt + 1))
                continue

            if not response.choices:
                raise EmptyResponseException("No choices in response")

            choice = response.choices[0]

            if choice.finish_reason == "content_filter":
                raise ContentFilteredException(
                    "Content was filtered by OpenAI moderation",
                    finish_reason=choice.finish_reason,
                )

            if choice.message.content is None:
                raise EmptyResponseException(
                    f"Response content is None, finish_reason: {choice.finish_reason}",
                    finish_reason=choice.finish_reason,
                )

            usage: Optional[TokenUsage] = None
            if response.usage is not None:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
            elif monitor is not None:
                logger.warning(
                    "Response usage is None but token monitor is not None. This is unexpected."
                )

            if monitor is not None and usage is not None:
                monitor.record_usage(usage, op_type)

            return LLMResponse(
                content=choice.message.content,
                finish_reason=choice.finish_reason or "stop",
                usage=usage,
            )

        raise last_exception  # type: ignore[misc]
