import logging
import os
import random
import re
import time
from collections.abc import Callable
from typing import TypeVar

import groq
import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, ValidationError

from coding_agent.config import DEFAULT_MODEL, LLM_TIMEOUT_SECONDS, MAX_PROMPT_CHARS, get_api_key
from coding_agent.errors import (
    AgentError,
    ConfigError,
    LLMError,
    LLMTimeoutError,
    PromptTooLargeError,
    RateLimitError,
)

logger = logging.getLogger("coding_agent.llm")

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

# Transient-failure policy: total tries per call, exponential backoff with a cap.
MAX_LLM_ATTEMPTS = 4
BASE_DELAY = 2.0   # seconds before the 2nd try; doubles each time
MAX_DELAY = 30.0   # longest we will sleep; a rate-limit wait beyond this fails fast
_sleep = time.sleep          # tests replace these two so nothing really waits
_jitter = random.uniform

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_RECOVERABLE_MARKERS = ("tool_use_failed", "json_validate_failed", "failed_generation")

RETRY_NOTE = (
    "Your previous reply was rejected because it was not a valid structured answer "
    "(for example, it tried to call a tool). You have no tools and cannot open files. "
    "Reply ONLY with the requested structured answer, using the information already provided."
)


def get_llm(model_name: str | None = None, temperature: float = 0.0) -> ChatGroq:
    """The single place where LLM clients are created."""
    return ChatGroq(
        model=model_name or DEFAULT_MODEL,
        temperature=temperature,
        api_key=get_api_key(),
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=0,  # we retry ourselves (see call_with_backoff) so waits are visible
    )


def structured(llm, schema: type[T]):
    """A runnable whose .invoke(messages) returns a validated `schema` instance.

    Set STRUCTURED_METHOD in .env (e.g. json_schema) to use a different method.
    """
    method = os.getenv("STRUCTURED_METHOD")
    if method:
        return llm.with_structured_output(schema, method=method)
    return llm.with_structured_output(schema)


def _is_recoverable(exc: Exception) -> bool:
    if isinstance(exc, (ValidationError, OutputParserException)):
        return True
    return any(marker in str(exc) for marker in _RECOVERABLE_MARKERS)


# ---- resilience: classify, back off, translate -----------------------------
def _status(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _classify(exc: Exception) -> str | None:
    """'rate_limit' | 'timeout' | 'transient' for retryable failures, else None."""
    if isinstance(exc, groq.RateLimitError) or _status(exc) == 429:
        return "rate_limit"
    if isinstance(exc, (groq.APITimeoutError, httpx.TimeoutException, TimeoutError)):
        return "timeout"
    if isinstance(exc, (groq.APIConnectionError, httpx.TransportError, ConnectionError)):
        return "transient"
    if isinstance(exc, groq.InternalServerError) or (_status(exc) or 0) >= 500:
        return "transient"
    return None


def _fatal(exc: Exception) -> AgentError | None:
    """Non-retryable service errors that deserve a clear message."""
    status = _status(exc)
    if status in (401, 403):
        return ConfigError(
            f"Groq rejected the API key (HTTP {status}). Check GROQ_API_KEY in your .env file."
        )
    if status == 413:
        return PromptTooLargeError(
            "The request is too large for this model's limits (HTTP 413)."
        )
    return None


_DURATION = re.compile(
    r"try again in\s+(?:(?P<h>\d+)h)?\s*(?:(?P<m>\d+)m)?\s*(?:(?P<s>\d+(?:\.\d+)?)s)?",
    re.IGNORECASE,
)


def _retry_after(exc: Exception) -> float | None:
    """Seconds the service asked us to wait: Retry-After header, else 'try again in 1m5s'."""
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = _DURATION.search(str(exc))
    if match and any(match.group(g) for g in ("h", "m", "s")):
        return (
            int(match.group("h") or 0) * 3600
            + int(match.group("m") or 0) * 60
            + float(match.group("s") or 0)
        )
    return None


def _human_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _give_up(kind: str, exc: Exception, attempts: int, wait: float | None = None) -> LLMError:
    if kind == "rate_limit":
        when = f" Try again in about {_human_duration(wait)}." if wait else ""
        return RateLimitError(f"Groq's rate limit was reached.{when}")
    if kind == "timeout":
        return LLMTimeoutError(
            f"The model did not answer within {LLM_TIMEOUT_SECONDS:.0f}s "
            f"(tried {attempts} time{'s' if attempts != 1 else ''})."
        )
    return LLMError(
        f"The model service could not be reached after {attempts} tries: "
        f"{type(exc).__name__}."
    )


def call_with_backoff(fn: Callable[[], R], what: str = "model call") -> R:
    """Run fn(), retrying transient failures with exponential backoff.

    Rate limits honour the service's Retry-After; a wait longer than MAX_DELAY (for
    example a daily-limit reset) fails fast instead of freezing the run. Anything we
    do not recognise is re-raised untouched so callers can handle it.
    """
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:
            kind = _classify(exc)
            if kind is None:
                fatal = _fatal(exc)
                if fatal is not None:
                    raise fatal from exc
                raise
            hinted = _retry_after(exc) if kind == "rate_limit" else None
            if kind == "rate_limit" and hinted is not None and hinted > MAX_DELAY:
                raise _give_up(kind, exc, attempt, hinted) from exc
            if attempt == MAX_LLM_ATTEMPTS:
                raise _give_up(kind, exc, attempt, hinted) from exc
            backoff = min(MAX_DELAY, BASE_DELAY * 2 ** (attempt - 1))
            wait = (hinted if hinted is not None else backoff) + _jitter(0, backoff * 0.25)
            logger.warning(
                "%s: %s, retrying in %.1fs (attempt %d of %d)",
                what, {"rate_limit": "rate limited", "timeout": "timed out",
                       "transient": "service error"}[kind],
                wait, attempt + 1, MAX_LLM_ATTEMPTS,
            )
            _sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


def _message_chars(messages) -> int:
    return sum(len(str(getattr(m, "content", m))) for m in messages)


def check_prompt_size(messages) -> None:
    """Fail early, with a clear message, instead of sending a request that will be refused."""
    size = _message_chars(messages)
    if size > MAX_PROMPT_CHARS:
        raise PromptTooLargeError(
            f"The prompt is {size:,} characters; the limit is {MAX_PROMPT_CHARS:,}."
        )


def invoke_structured(llm, schema: type[T], messages) -> T:
    """Structured call: size check, backoff on transient errors, and one corrective retry
    for malformed or tool-call replies (a different failure from a transient one)."""
    check_prompt_size(messages)
    runnable = structured(llm, schema)
    try:
        return call_with_backoff(lambda: runnable.invoke(messages), schema.__name__)
    except Exception as exc:
        if not _is_recoverable(exc):
            raise
    try:
        return call_with_backoff(
            lambda: runnable.invoke([*messages, HumanMessage(content=RETRY_NOTE)]),
            schema.__name__,
        )
    except Exception as exc:
        if _is_recoverable(exc):
            raise AgentError(
                "The model could not produce a valid structured answer after a retry. "
                "Run the task again or switch models."
            ) from exc
        raise


def invoke_text(llm, messages) -> str:
    """Plain-text call with the same size check and backoff; returns cleaned text."""
    check_prompt_size(messages)
    reply = call_with_backoff(lambda: llm.invoke(messages), "text call")
    return clean_text(reply.content)


def clean_text(content) -> str:
    """Normalize model output to plain text and strip <think> reasoning blocks."""
    if isinstance(content, list):
        content = "".join(
            part if isinstance(part, str) else part.get("text", "") for part in content
        )
    return _THINK_RE.sub("", str(content)).strip()