import os
import re
from typing import TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, ValidationError

from coding_agent.config import DEFAULT_MODEL, get_api_key
from coding_agent.errors import AgentError

T = TypeVar("T", bound=BaseModel)

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


def invoke_structured(llm, schema: type[T], messages) -> T:
    """Structured call with one corrective retry for malformed or tool-call replies."""
    runnable = structured(llm, schema)
    try:
        return runnable.invoke(messages)
    except Exception as exc:
        if not _is_recoverable(exc):
            raise
    try:
        return runnable.invoke([*messages, HumanMessage(content=RETRY_NOTE)])
    except Exception as exc:
        if _is_recoverable(exc):
            raise AgentError(
                "The model could not produce a valid structured answer after a retry. "
                "Run the task again or switch models."
            ) from exc
        raise


def clean_text(content) -> str:
    """Normalize model output to plain text and strip <think> reasoning blocks."""
    if isinstance(content, list):
        content = "".join(
            part if isinstance(part, str) else part.get("text", "") for part in content
        )
    return _THINK_RE.sub("", str(content)).strip()