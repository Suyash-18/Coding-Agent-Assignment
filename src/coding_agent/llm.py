import os
import re
from typing import TypeVar

from langchain_groq import ChatGroq
from pydantic import BaseModel

from coding_agent.config import DEFAULT_MODEL, get_api_key

T = TypeVar("T", bound=BaseModel)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def get_llm(model_name: str | None = None, temperature: float = 0.0) -> ChatGroq:
    """The single place where LLM clients are created."""
    return ChatGroq(
        model=model_name or DEFAULT_MODEL,
        temperature=temperature,
        api_key=get_api_key(),
    )


def structured(llm, schema: type[T]):
    """A runnable whose .invoke(messages) returns a validated `schema` instance.

    Set STRUCTURED_METHOD in .env (e.g. json_schema) if a model needs a
    different structured-output method than the default.
    """
    method = os.getenv("STRUCTURED_METHOD")
    if method:
        return llm.with_structured_output(schema, method=method)
    return llm.with_structured_output(schema)


def clean_text(content) -> str:
    """Normalize model output to plain text and strip <think> reasoning blocks."""
    if isinstance(content, list):
        content = "".join(
            part if isinstance(part, str) else part.get("text", "") for part in content
        )
    return _THINK_RE.sub("", str(content)).strip()